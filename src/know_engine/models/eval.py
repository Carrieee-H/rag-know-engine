from sqlalchemy import BigInteger, ForeignKey, Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from know_engine.constants import EvalRunStatus
from know_engine.models.base import AuditMixin, Base, bigint_pk


class EvalRun(AuditMixin, Base):
    __tablename__ = "eval_run"

    id: Mapped[int] = bigint_pk()
    run_id: Mapped[str] = mapped_column(String(64), unique=True)
    dataset_path: Mapped[str] = mapped_column(String(512))
    config_json: Mapped[dict | None] = mapped_column(JSONB)
    summary_json: Mapped[dict | None] = mapped_column(JSONB)  # aggregate metrics, no per-case detail
    total_cases: Mapped[int] = mapped_column(server_default="0")
    evaluated_cases: Mapped[int] = mapped_column(server_default="0")  # cases with retrieval ground truth
    generation_evaluated_cases: Mapped[int] = mapped_column(server_default="0")  # cases scored by LLM judge
    status: Mapped[str] = mapped_column(String(32), default=EvalRunStatus.COMPLETED)

    __table_args__ = (Index(None, "created_at"),)


class EvalRunCase(AuditMixin, Base):
    __tablename__ = "eval_run_case"

    id: Mapped[int] = bigint_pk()
    run_id: Mapped[str] = mapped_column(String(64), ForeignKey("eval_run.run_id", ondelete="CASCADE"))
    case_id: Mapped[str] = mapped_column(String(64))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str | None] = mapped_column(Text)
    transform_content: Mapped[str | None] = mapped_column(Text)
    retrieved_chunk_ids: Mapped[list | None] = mapped_column(JSONB)
    retrieved_document_ids: Mapped[list | None] = mapped_column(JSONB)
    rag_references: Mapped[list | None] = mapped_column(JSONB)
    retrieval_scores: Mapped[dict | None] = mapped_column(JSONB)
    generation_scores: Mapped[dict | None] = mapped_column(JSONB)
    latency_ms: Mapped[int] = mapped_column(BigInteger, server_default="0")
    success: Mapped[bool] = mapped_column(server_default="true")
    error_message: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (Index(None, "run_id", "case_id"),)
