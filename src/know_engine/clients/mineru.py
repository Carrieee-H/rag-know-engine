"""MinerU parsing service (mineru-api 4.x), used for PDF and DOCX."""

import httpx

from know_engine.config import get_settings

PARSE_TIMEOUT_SECONDS = 1800


def parse_to_zip(file_name: str, data: bytes, tier: str = "standard") -> bytes:
    """Convert a document with MinerU and return the result zip (``markdown.md`` plus any images)."""
    response = httpx.post(
        f"{get_settings().mineru_url}/v1/file_parse",
        params={
            "response_format": "zip",
            "tier": tier,
            "filename": file_name,
            "wait_timeout": PARSE_TIMEOUT_SECONDS,
        },
        content=data,
        headers={"Content-Type": "application/octet-stream"},
        timeout=PARSE_TIMEOUT_SECONDS + 60,
    )
    if response.status_code == 202:
        raise RuntimeError(f"MinerU 解析超过 {PARSE_TIMEOUT_SECONDS} 秒仍未完成")
    if response.status_code != 200:
        raise RuntimeError(f"MinerU 解析失败: HTTP {response.status_code}, {response.text[:500]}")
    return response.content
