from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Index, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from know_engine.models.base import Base, bigint_pk


class IngestStageLog(Base):
    """One row per pipeline stage run (convert / split / quality / embed): the basis for
    throughput and latency statistics and for tracing a version's history."""

    __tablename__ = "ingest_stage_log"

    id: Mapped[int] = bigint_pk()
    doc_id: Mapped[int] = mapped_column(BigInteger)
    version_id: Mapped[int] = mapped_column(BigInteger)
    stage: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))  # SUCCESS / FAILED
    duration_ms: Mapped[int] = mapped_column(BigInteger)
    items: Mapped[int] = mapped_column(server_default="0")  # chunks produced / embedded
    detail: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index(None, "version_id"),
        Index(None, "stage", "created_at"),
    )
