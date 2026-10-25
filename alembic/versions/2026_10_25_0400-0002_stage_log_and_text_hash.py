"""ingest stage log and segment text hash

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-25
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("knowledge_segment", sa.Column("text_hash", sa.String(64), nullable=True))
    op.create_index(op.f("ix_knowledge_segment_text_hash"), "knowledge_segment", ["text_hash"])
    op.create_table(
        "ingest_stage_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("doc_id", sa.BigInteger(), nullable=False),
        sa.Column("version_id", sa.BigInteger(), nullable=False),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False),
        sa.Column("duration_ms", sa.BigInteger(), nullable=False),
        sa.Column("items", sa.Integer(), server_default="0", nullable=False),
        sa.Column("detail", postgresql.JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_ingest_stage_log")),
    )
    op.create_index(op.f("ix_ingest_stage_log_version_id"), "ingest_stage_log", ["version_id"])
    op.create_index(op.f("ix_ingest_stage_log_stage_created_at"), "ingest_stage_log", ["stage", "created_at"])


def downgrade() -> None:
    op.drop_table("ingest_stage_log")
    op.drop_index(op.f("ix_knowledge_segment_text_hash"), table_name="knowledge_segment")
    op.drop_column("knowledge_segment", "text_hash")
