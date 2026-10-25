"""Convert an uploaded file into the text that splitting works on.

- Markdown: image tags get a description from the vision model (only http(s) images can be fetched).
- PDF / DOCX: MinerU produces Markdown plus images; images go to MinIO and get descriptions.
- TXT / XLSX / CSV: no conversion, the raw file is used as is.

Returns the MinIO URI of the converted file.
"""

import io
import posixpath
import re
import zipfile

import httpx
import structlog

from know_engine.clients import storage
from know_engine.clients.llm import describe_image
from know_engine.clients.mineru import parse_to_zip

log = structlog.get_logger()

IMAGE_TAG = re.compile(r"!\[(.*?)\]\(([^)\s]+)\)")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
MINERU_EXTENSIONS = (".pdf", ".docx", ".doc")
RAW_EXTENSIONS = (".txt", ".xlsx", ".xls", ".csv")
MARKDOWN_EXTENSIONS = (".md", ".markdown")
SUPPORTED_EXTENSIONS = MINERU_EXTENSIONS + RAW_EXTENSIONS + MARKDOWN_EXTENSIONS


def extension(file_name: str) -> str:
    return posixpath.splitext(file_name.lower())[1]


def check_supported(file_name: str) -> None:
    if extension(file_name) not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型: {file_name}，支持 {', '.join(SUPPORTED_EXTENSIONS)}")


def raw_object_name(doc_id: int, version_id: int, file_name: str) -> str:
    return f"raw/{doc_id}/{version_id}/{file_name}"


def converted_prefix(doc_id: int, version_id: int) -> str:
    # Per version, so a new version never overwrites the previous one's files.
    return f"converted/{doc_id}/{version_id}/"


def _describe_or_keep(alt: str, image_bytes: bytes | None) -> str:
    if image_bytes is None:
        return alt
    try:
        return describe_image(image_bytes) or alt
    except Exception as e:  # a failed description must not fail the whole document
        log.warning("image_description_failed", error=str(e))
        return alt


def _enrich_markdown(markdown: str, images: dict[str, tuple[bytes, str]]) -> str:
    """Rewrite image tags as ``![description](uri)``.

    ``images`` maps a file name to (bytes, MinIO URI) for images that came with the document.
    Other http(s) images are downloaded for the description and keep their URL.
    """

    def replace(match: re.Match) -> str:
        alt, path = match.group(1), match.group(2)
        name = posixpath.basename(path)
        if name in images:
            image_bytes, uri = images[name]
            return f"![{_describe_or_keep(alt, image_bytes)}]({uri})"
        image_bytes = None
        if path.startswith(("http://", "https://")):
            try:
                image_bytes = httpx.get(path, timeout=30, follow_redirects=True).raise_for_status().content
            except Exception as e:
                log.warning("image_download_failed", url=path, error=str(e))
        return f"![{_describe_or_keep(alt, image_bytes)}]({path})"

    return IMAGE_TAG.sub(replace, markdown)


def _convert_with_mineru(data: bytes, file_name: str, prefix: str) -> str:
    # MinerU sees an ASCII name; the original name stays in our database.
    with zipfile.ZipFile(io.BytesIO(parse_to_zip(f"upload{extension(file_name)}", data))) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]
        md_names = [n for n in names if n.lower().endswith(".md")]
        if not md_names:
            raise RuntimeError("MinerU 结果中没有 Markdown 文件")
        images: dict[str, tuple[bytes, str]] = {}
        for name in names:
            if name.lower().endswith(IMAGE_EXTENSIONS):
                image_bytes = archive.read(name)
                base = posixpath.basename(name)
                uri = storage.put_bytes(f"{prefix}images/{base}", image_bytes, "image/" + extension(base).lstrip("."))
                images[base] = (image_bytes, uri)
        markdown = archive.read(md_names[0]).decode("utf-8")
    return storage.put_bytes(f"{prefix}content.md", _enrich_markdown(markdown, images).encode(), "text/markdown")


def convert(data: bytes, file_name: str, doc_id: int, version_id: int, raw_uri: str) -> str:
    ext = extension(file_name)
    prefix = converted_prefix(doc_id, version_id)
    if ext in RAW_EXTENSIONS:
        return raw_uri
    if ext in MARKDOWN_EXTENSIONS:
        markdown = _enrich_markdown(data.decode("utf-8", errors="replace"), {})
        return storage.put_bytes(f"{prefix}content.md", markdown.encode(), "text/markdown")
    if ext in MINERU_EXTENSIONS:
        return _convert_with_mineru(data, file_name, prefix)
    raise ValueError(f"不支持的文件类型: {file_name}")


def converted_file_name(original_name: str, converted_uri: str) -> str:
    """Name the splitter should see: tables keep their extension, everything else is Markdown/text."""
    return posixpath.basename(storage.object_name_from_uri(converted_uri)) or original_name
