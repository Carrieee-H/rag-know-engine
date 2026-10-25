from datetime import date

from sqlalchemy import BigInteger, Date, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from know_engine.constants import DocumentStatus, SegmentStatus
from know_engine.models.base import AuditMixin, Base, bigint_pk


class KnowledgeDocument(AuditMixin, Base):
    __tablename__ = "knowledge_document"

    doc_id: Mapped[int] = bigint_pk()
    doc_title: Mapped[str] = mapped_column(String(1024))
    expire_date: Mapped[date | None] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(32), default=DocumentStatus.UPLOADED)
    description: Mapped[str | None] = mapped_column(String(512))
    extension: Mapped[dict | None] = mapped_column(JSONB)
    # Points at knowledge_document_version.version_id. No FK: the two tables reference each other.
    current_version_id: Mapped[int | None] = mapped_column(BigInteger)

    __table_args__ = (
        Index(None, "status", "doc_id"),
        Index(None, "created_at"),
    )


class KnowledgeDocumentVersion(AuditMixin, Base):
    __tablename__ = "knowledge_document_version"

    version_id: Mapped[int] = bigint_pk()
    doc_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("knowledge_document.doc_id", ondelete="CASCADE"), index=True
    )
    version: Mapped[str] = mapped_column(String(32), server_default="1.0.0")
    doc_url: Mapped[str | None] = mapped_column(String(2048))
    converted_doc_url: Mapped[str | None] = mapped_column(String(2048))
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32), default=DocumentStatus.UPLOADED)
    upload_user: Mapped[str | None] = mapped_column(String(255))
    changelog: Mapped[str | None] = mapped_column(String(1024))
    # New: worker retry bookkeeping
    attempts: Mapped[int] = mapped_column(server_default="0")
    last_error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        UniqueConstraint("doc_id", "version"),
        # The worker polls by status in updated_at order.
        Index(None, "status", "updated_at"),
    )


class KnowledgeSegment(AuditMixin, Base):
    __tablename__ = "knowledge_segment"

    id: Mapped[int] = bigint_pk()
    text: Mapped[str] = mapped_column(Text)
    # sha256 of (embedding model, text): lets a new version reuse vectors of unchanged chunks.
    text_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    chunk_id: Mapped[str] = mapped_column(String(64), unique=True)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSONB)
    document_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("knowledge_document.doc_id", ondelete="CASCADE")
    )
    document_version: Mapped[int] = mapped_column(
        BigInteger,
        # Explicit name: the generated one exceeds PostgreSQL's 63-character limit.
        ForeignKey(
            "knowledge_document_version.version_id",
            name="fk_knowledge_segment_document_version",
            ondelete="CASCADE",
        ),
        index=True,
    )
    chunk_order: Mapped[int]
    embedding_id: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), default=SegmentStatus.STORED, index=True)
    skip_embedding: Mapped[bool] = mapped_column(server_default="false")

    __table_args__ = (
        UniqueConstraint("document_version", "chunk_order"),
        Index(None, "document_id", "chunk_order"),
        Index(None, "document_id", "status", "skip_embedding"),
    )
