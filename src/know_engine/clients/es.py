"""Elasticsearch chunk index.

Reads and writes go through an alias; the physical index carries a version suffix so it
can be rebuilt and swapped later (stage 4). Document ``_id`` is the chunk_id, which makes
indexing idempotent.
"""

from functools import lru_cache

from elasticsearch import Elasticsearch, helpers

from know_engine.config import get_settings


@lru_cache
def get_es() -> Elasticsearch:
    return Elasticsearch(get_settings().es_url, request_timeout=60)


# Single local node: replicas could never be assigned and would keep the cluster yellow.
INDEX_SETTINGS = {"number_of_shards": 1, "number_of_replicas": 0}


def index_mapping(dims: int) -> dict:
    return {
        "properties": {
            "chunk_id": {"type": "keyword"},
            "text_hash": {"type": "keyword"},
            "doc_id": {"type": "long"},
            "version_id": {"type": "long"},
            "parent_chunk_id": {"type": "keyword"},
            "file_name": {"type": "keyword"},
            "text": {"type": "text", "analyzer": "ik_max_word", "search_analyzer": "ik_smart"},
            "vector": {"type": "dense_vector", "dims": dims, "index": True, "similarity": "cosine"},
            "metadata": {"type": "object", "enabled": False},
        }
    }


_checked_mapping: list[bool] = []


def ensure_index() -> str:
    """Create ``<alias>-v1`` behind the alias if the alias does not exist yet."""
    es, s = get_es(), get_settings()
    alias = s.es_index_alias
    if not es.indices.exists_alias(name=alias):
        physical = f"{alias}-v1"
        es.indices.create(index=physical, settings=INDEX_SETTINGS, mappings=index_mapping(s.embedding_dims),
                          aliases={alias: {}})
    elif not _checked_mapping:
        # Add fields introduced after the index was created (adding a field is allowed in place).
        es.indices.put_mapping(index=alias, properties={"text_hash": {"type": "keyword"}})
        _checked_mapping.append(True)
    return alias


def find_vectors(text_hashes: list[str]) -> dict[str, list[float]]:
    """Vectors already indexed for these text hashes (any document or version)."""
    if not text_hashes:
        return {}
    hits = get_es().search(index=ensure_index(), size=len(text_hashes) * 2, source=["text_hash", "vector"],
                           query={"terms": {"text_hash": text_hashes}})["hits"]["hits"]
    return {h["_source"]["text_hash"]: h["_source"]["vector"] for h in hits}


def bulk_upsert(docs: list[dict]) -> None:
    """Index chunk documents; each dict must contain ``chunk_id``."""
    alias = ensure_index()
    actions = ({"_op_type": "index", "_index": alias, "_id": d["chunk_id"], "_source": d} for d in docs)
    _, errors = helpers.bulk(get_es(), actions, raise_on_error=False, refresh="wait_for")
    if errors:
        raise RuntimeError(f"ES 写入失败 {len(errors)} 条，例如: {errors[0]}")


def delete_by_query(query: dict) -> int:
    alias = ensure_index()
    result = get_es().delete_by_query(index=alias, query=query, refresh=True, conflicts="proceed")
    return result["deleted"]


def delete_version(doc_id: int, version_id: int) -> int:
    return delete_by_query({"bool": {"filter": [{"term": {"doc_id": doc_id}}, {"term": {"version_id": version_id}}]}})


def delete_other_versions(doc_id: int, keep_version_id: int) -> int:
    return delete_by_query({
        "bool": {"filter": [{"term": {"doc_id": doc_id}}], "must_not": [{"term": {"version_id": keep_version_id}}]}
    })


def delete_documents(doc_ids: list[int]) -> int:
    return delete_by_query({"terms": {"doc_id": doc_ids}})


def count_version(doc_id: int, version_id: int) -> int:
    alias = ensure_index()
    query = {"bool": {"filter": [{"term": {"doc_id": doc_id}}, {"term": {"version_id": version_id}}]}}
    return get_es().count(index=alias, query=query)["count"]
