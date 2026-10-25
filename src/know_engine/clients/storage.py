"""MinIO object storage.

Objects are referenced in the database as ``minio://<bucket>/<object>`` URIs.
"""

import io
from functools import lru_cache

from minio import Minio

from know_engine.config import get_settings

URI_PREFIX = "minio://"


@lru_cache
def get_minio() -> Minio:
    s = get_settings()
    client = Minio(s.minio_endpoint, access_key=s.minio_access_key, secret_key=s.minio_secret_key, secure=s.minio_secure)
    if not client.bucket_exists(s.minio_bucket):
        client.make_bucket(s.minio_bucket)
    return client


def to_uri(object_name: str) -> str:
    return f"{URI_PREFIX}{get_settings().minio_bucket}/{object_name}"


def object_name_from_uri(uri: str) -> str:
    bucket = get_settings().minio_bucket
    prefix = f"{URI_PREFIX}{bucket}/"
    if not uri.startswith(prefix):
        raise ValueError(f"不是本系统的对象存储地址: {uri}")
    return uri[len(prefix):]


def put_bytes(object_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload and return the object's URI. Same name overwrites, so retries are safe."""
    get_minio().put_object(
        get_settings().minio_bucket, object_name, io.BytesIO(data), length=len(data), content_type=content_type
    )
    return to_uri(object_name)


def get_bytes(uri: str) -> bytes:
    response = get_minio().get_object(get_settings().minio_bucket, object_name_from_uri(uri))
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def remove_prefix(prefix: str) -> int:
    """Delete every object under a prefix; returns how many were deleted."""
    client, bucket = get_minio(), get_settings().minio_bucket
    names = [o.object_name for o in client.list_objects(bucket, prefix=prefix, recursive=True)]
    for name in names:
        client.remove_object(bucket, name)
    return len(names)
