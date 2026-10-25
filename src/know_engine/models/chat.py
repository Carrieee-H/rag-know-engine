from sqlalchemy import Index, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from know_engine.models.base import AuditMixin, Base, bigint_pk


class ChatConversation(AuditMixin, Base):
    __tablename__ = "chat_conversation"

    id: Mapped[int] = bigint_pk()
    conversation_id: Mapped[str] = mapped_column(String(64), unique=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str | None] = mapped_column(String(512))


class ChatMessage(AuditMixin, Base):
    __tablename__ = "chat_message"

    id: Mapped[int] = bigint_pk()
    message_id: Mapped[str] = mapped_column(String(64), unique=True)
    conversation_id: Mapped[str] = mapped_column(String(64))
    type: Mapped[str] = mapped_column(String(32))  # USER / ASSISTANT
    content: Mapped[str | None] = mapped_column(Text)
    transform_content: Mapped[str | None] = mapped_column(Text)  # rewritten query
    token_count: Mapped[int | None]
    model_name: Mapped[str | None] = mapped_column(String(128))
    # [{document_id, document_title, chunk_id, chunk_content, similarity_score, retrieval_source}, ...]
    rag_references: Mapped[list | None] = mapped_column(JSONB)

    __table_args__ = (Index(None, "conversation_id", "created_at"),)
