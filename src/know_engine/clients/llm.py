"""Model API clients (OpenAI-compatible): chat, vision and embeddings."""

import base64
import io
from functools import lru_cache

from openai import OpenAI
from PIL import Image

from know_engine.config import get_settings
from know_engine.prompts import load_prompt

EMBED_BATCH_SIZE = 32
IMAGE_MAX_SIDE = 1024


@lru_cache
def get_client() -> OpenAI:
    s = get_settings()
    if not s.llm_base_url.startswith(("http://", "https://")):
        raise ValueError("LLM_BASE_URL 必须是 http(s):// 开头的地址，请检查 .env")
    # The model API rate-limits bursts (HTTP 429); the SDK retries with exponential backoff
    # and honours Retry-After, so give it enough attempts to ride out a burst.
    return OpenAI(base_url=s.llm_base_url, api_key=s.llm_api_key, timeout=60, max_retries=6)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed in batches; output order matches input order."""
    s = get_settings()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        response = get_client().embeddings.create(model=s.embedding_model, input=batch)
        data = sorted(response.data, key=lambda d: d.index)
        vectors.extend(d.embedding for d in data)
    for v in vectors:
        if len(v) != s.embedding_dims:
            raise ValueError(f"向量维度 {len(v)} 与 EMBEDDING_DIMS={s.embedding_dims} 不一致")
    return vectors


def _to_data_url(image_bytes: bytes) -> str:
    """Shrink the long side to IMAGE_MAX_SIDE and encode as a PNG data URL.

    Images live in local MinIO, which the model API cannot reach, so they are sent inline.
    """
    image = Image.open(io.BytesIO(image_bytes))
    image.thumbnail((IMAGE_MAX_SIDE, IMAGE_MAX_SIDE))
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def describe_image(image_bytes: bytes) -> str | None:
    """Return a one-line description, or None when no vision model is configured."""
    s = get_settings()
    if not s.vision_model:
        return None
    response = get_client().chat.completions.create(
        model=s.vision_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": load_prompt("image-description-user-prompt.txt")},
                {"type": "image_url", "image_url": {"url": _to_data_url(image_bytes)}},
            ],
        }],
    )
    text = (response.choices[0].message.content or "").strip()
    # Keep the Markdown image tag on one line.
    return " ".join(text.split()) or None
