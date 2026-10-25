"""initial schema: documents, versions, segments, chat, eval

Revision ID: 0001
Revises:
Create Date: 2026-10-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def audit_columns() -> list[sa.Column]:
    """Columns from AuditMixin, shared by every table."""
    return [
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("lock_version", sa.Integer(), server_default="0", nullable=False),
        sa.Column("deleted", sa.Boolean(), server_default="false", nullable=False),
    ]


def upgrade() -> None:
    op.create_table(
        "knowledge_document",
        sa.Column("doc_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("doc_title", sa.String(1024), nullable=False),
        sa.Column("expire_date", sa.Date(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("description", sa.String(512), nullable=True),
        sa.Column("extension", postgresql.JSONB(), nullable=True),
        sa.Column("current_version_id", sa.BigInteger(), nullable=True),
        *audit_columns(),
        sa.PrimaryKeyConstraint("doc_id", name=op.f("pk_knowledge_document")),
    )
    op.create_index(op.f("ix_knowledge_document_status_doc_id"), "knowledge_document", ["status", "doc_id"])
    op.create_index(op.f("ix_knowledge_document_created_at"), "knowledge_document", ["created_at"])

    op.create_table(
        "knowledge_document_version",
        sa.Column("version_id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("doc_id", sa.BigInteger(), nullable=False),
        sa.Column("version", sa.String(32), server_default="1.0.0", nullable=False),
        sa.Column("doc_url", sa.String(2048), nullable=True),
        sa.Column("converted_doc_url", sa.String(2048), nullable=True),
        sa.Column("content_hash", sa.String(64), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("upload_user", sa.String(255), nullable=True),
        sa.Column("changelog", sa.String(1024), nullable=True),
        sa.Column("attempts", sa.Integer(), server_default="0", nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        *audit_columns(),
        sa.ForeignKeyConstraint(
            ["doc_id"],
            ["knowledge_document.doc_id"],
            name=op.f("fk_knowledge_document_version_doc_id_knowledge_document"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("version_id", name=op.f("pk_knowledge_document_version")),
        sa.UniqueConstraint("doc_id", "version", name=op.f("uq_knowledge_document_version_doc_id_version")),
    )
    op.create_index(op.f("ix_knowledge_document_version_doc_id"), "knowledge_document_version", ["doc_id"])
    op.create_index(
        op.f("ix_knowledge_document_version_content_hash"), "knowledge_document_version", ["content_hash"]
    )
    op.create_index(
        op.f("ix_knowledge_document_version_status_updated_at"),
        "knowledge_document_version",
        ["status", "updated_at"],
    )

    op.create_table(
        "knowledge_segment",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("chunk_id", sa.String(64), nullable=False),
        sa.Column("metadata", postgresql.JSONB(), nullable=True),
        sa.Column("document_id", sa.BigInteger(), nullable=False),
        sa.Column("document_version", sa.BigInteger(), nullable=False),
        sa.Column("chunk_order", sa.Integer(), nullable=False),
        sa.Column("embedding_id", sa.String(255), nullable=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("skip_embedding", sa.Boolean(), server_default="false", nullable=False),
        *audit_columns(),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["knowledge_document.doc_id"],
            name=op.f("fk_knowledge_segment_document_id_knowledge_document"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["document_version"],
            ["knowledge_document_version.version_id"],
            name=op.f("fk_knowledge_segment_document_version"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_knowledge_segment")),
        sa.UniqueConstraint("chunk_id", name=op.f("uq_knowledge_segment_chunk_id")),
        sa.UniqueConstraint(
            "document_version", "chunk_order", name=op.f("uq_knowledge_segment_document_version_chunk_order")
        ),
    )
    op.create_index(op.f("ix_knowledge_segment_document_version"), "knowledge_segment", ["document_version"])
    op.create_index(op.f("ix_knowledge_segment_status"), "knowledge_segment", ["status"])
    op.create_index(
        op.f("ix_knowledge_segment_document_id_chunk_order"), "knowledge_segment", ["document_id", "chunk_order"]
    )
    op.create_index(
        op.f("ix_knowledge_segment_document_id_status_skip_embedding"),
        "knowledge_segment",
        ["document_id", "status", "skip_embedding"],
    )

    op.create_table(
        "chat_conversation",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("conversation_id", sa.String(64), nullable=False),
        sa.Column("user_id", sa.String(64), nullable=False),
        sa.Column("title", sa.String(512), nullable=True),
        *audit_columns(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chat_conversation")),
        sa.UniqueConstraint("conversation_id", name=op.f("uq_chat_conversation_conversation_id")),
    )
    op.create_index(op.f("ix_chat_conversation_user_id"), "chat_conversation", ["user_id"])

    op.create_table(
        "chat_message",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("message_id", sa.String(64), nullable=False),
        sa.Column("conversation_id", sa.String(64), nullable=False),
        sa.Column("type", sa.String(32), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("transform_content", sa.Text(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("model_name", sa.String(128), nullable=True),
        sa.Column("rag_references", postgresql.JSONB(), nullable=True),
        *audit_columns(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_chat_message")),
        sa.UniqueConstraint("message_id", name=op.f("uq_chat_message_message_id")),
    )
    op.create_index(
        op.f("ix_chat_message_conversation_id_created_at"), "chat_message", ["conversation_id", "created_at"]
    )

    op.create_table(
        "eval_run",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("dataset_path", sa.String(512), nullable=False),
        sa.Column("config_json", postgresql.JSONB(), nullable=True),
        sa.Column("summary_json", postgresql.JSONB(), nullable=True),
        sa.Column("total_cases", sa.Integer(), server_default="0", nullable=False),
        sa.Column("evaluated_cases", sa.Integer(), server_default="0", nullable=False),
        sa.Column("generation_evaluated_cases", sa.Integer(), server_default="0", nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        *audit_columns(),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_eval_run")),
        sa.UniqueConstraint("run_id", name=op.f("uq_eval_run_run_id")),
    )
    op.create_index(op.f("ix_eval_run_created_at"), "eval_run", ["created_at"])

    op.create_table(
        "eval_run_case",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("case_id", sa.String(64), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("transform_content", sa.Text(), nullable=True),
        sa.Column("retrieved_chunk_ids", postgresql.JSONB(), nullable=True),
        sa.Column("retrieved_document_ids", postgresql.JSONB(), nullable=True),
        sa.Column("rag_references", postgresql.JSONB(), nullable=True),
        sa.Column("retrieval_scores", postgresql.JSONB(), nullable=True),
        sa.Column("generation_scores", postgresql.JSONB(), nullable=True),
        sa.Column("latency_ms", sa.BigInteger(), server_default="0", nullable=False),
        sa.Column("success", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        *audit_columns(),
        sa.ForeignKeyConstraint(
            ["run_id"], ["eval_run.run_id"], name=op.f("fk_eval_run_case_run_id_eval_run"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_eval_run_case")),
    )
    op.create_index(op.f("ix_eval_run_case_run_id_case_id"), "eval_run_case", ["run_id", "case_id"])


def downgrade() -> None:
    op.drop_table("eval_run_case")
    op.drop_table("eval_run")
    op.drop_table("chat_message")
    op.drop_table("chat_conversation")
    op.drop_table("knowledge_segment")
    op.drop_table("knowledge_document_version")
    op.drop_table("knowledge_document")
