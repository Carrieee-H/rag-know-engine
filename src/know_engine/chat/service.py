"""Streaming chat: conversation bookkeeping, intent routing, RAG and plain chat.

``chat_events`` yields the SSE payloads the Java version produced, in the same order:
progress messages, an optional ``[REFERENCE]:<json>``, answer tokens, then ``[DONE]:<conversationId>``.
"""

import json
import queue
import threading
import uuid
from collections.abc import Iterator

import structlog
from sqlalchemy import func, select, update

from know_engine.config import get_settings
from know_engine.constants import MessageType
from know_engine.db import session_scope
from know_engine.models import ChatConversation, ChatMessage
from know_engine.prompts import load_prompt
from know_engine.rag import llm, pipeline
from know_engine.rag.query import recognize_intent

log = structlog.get_logger()

DEFAULT_USER = "default-user"
HISTORY_MESSAGES = 10  # previous messages (user + assistant) given to rewrite and generation
PROGRESS = "[PROGRESS]:"


def _new_id() -> str:
    return uuid.uuid4().hex


def create_conversation(user_id: str, first_message: str) -> str:
    conversation_id = _new_id()
    with session_scope() as s:
        s.add(ChatConversation(conversation_id=conversation_id, user_id=user_id, title=first_message[:20]))
    threading.Thread(target=_generate_title, args=(conversation_id, first_message), daemon=True).start()
    return conversation_id


def _generate_title(conversation_id: str, first_message: str) -> None:
    """Replace the temporary title (first 20 characters) with a model-written one."""
    try:
        title = llm.complete(get_settings().light_model, [
            {"role": "system", "content": load_prompt("title-summary-system-prompt.txt")},
            {"role": "user", "content": llm.render(load_prompt("title-summary-user-prompt.txt"), it=first_message)},
        ]).strip().strip("\"“”")[:30]
        if title:
            with session_scope() as s:
                s.execute(update(ChatConversation).where(ChatConversation.conversation_id == conversation_id)
                          .values(title=title))
    except Exception as e:
        log.warning("title_generation_failed", conversation_id=conversation_id, error=str(e))


def _save_message(conversation_id: str, type_: str, content: str | None) -> str:
    message_id = _new_id()
    with session_scope() as s:
        s.add(ChatMessage(message_id=message_id, conversation_id=conversation_id, type=type_, content=content))
        s.execute(update(ChatConversation).where(ChatConversation.conversation_id == conversation_id)
                  .values(updated_at=func.now()))
    return message_id


def _update_message(message_id: str, **values) -> None:
    with session_scope() as s:
        s.execute(update(ChatMessage).where(ChatMessage.message_id == message_id).values(**values))


def _history(conversation_id: str, exclude: set[str]) -> list[tuple[str, str]]:
    with session_scope() as s:
        rows = s.scalars(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation_id, ChatMessage.deleted.is_(False))
            .order_by(ChatMessage.created_at.desc(), ChatMessage.id.desc())
            .limit(HISTORY_MESSAGES + len(exclude))
        ).all()
    history = [(m.type, m.content) for m in reversed(rows) if m.message_id not in exclude and m.content]
    return history[-HISTORY_MESSAGES:]


def _retrieve_with_progress(question: str, history: list[tuple[str, str]]) -> Iterator[str | pipeline.Retrieval]:
    """Run retrieval in a worker thread and yield its progress messages as they happen,
    then the Retrieval result itself."""
    events: queue.Queue = queue.Queue()

    def work() -> None:
        try:
            events.put(pipeline.retrieve(question, history, progress=lambda m: events.put(PROGRESS + m)))
        except Exception as e:
            events.put(e)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = events.get()
        if isinstance(item, Exception):
            raise item
        yield item
        if isinstance(item, pipeline.Retrieval):
            return


def chat_events(content: str, conversation_id: str | None, user_id: str | None) -> Iterator[str]:
    user_id = user_id or DEFAULT_USER
    conversation_id = conversation_id or create_conversation(user_id, content)
    history = _history(conversation_id, exclude=set())
    user_message_id = _save_message(conversation_id, MessageType.USER, content)
    assistant_message_id = _save_message(conversation_id, MessageType.ASSISTANT, None)
    settings = get_settings()
    answer: list[str] = []
    model_name = settings.chat_model
    try:
        yield PROGRESS + "正在识别您的意图..."
        intent = recognize_intent(content)
        if not intent.related:
            yield PROGRESS + "正在为您生成回答..."
            messages = [{"role": "system", "content": load_prompt("common-chat-system-prompt.txt")}]
            messages += [{"role": "user" if r == MessageType.USER else "assistant", "content": t} for r, t in history]
            messages.append({"role": "user", "content": content})
            tokens = llm.stream(settings.chat_model, messages)
        else:
            retrieval = None
            for item in _retrieve_with_progress(content, history):
                if isinstance(item, pipeline.Retrieval):
                    retrieval = item
                else:
                    yield item
            references = pipeline.reference_dicts(retrieval)
            _update_message(user_message_id, transform_content=retrieval.rewritten)
            _update_message(assistant_message_id, rag_references=references)
            if references:
                yield "[REFERENCE]:" + json.dumps(references, ensure_ascii=False)
            yield PROGRESS + "正在生成回答..."
            tokens = pipeline.answer_stream(content, retrieval.references, history)
        for token in tokens:
            answer.append(token)
            yield token
    except Exception as e:
        log.error("chat_failed", conversation_id=conversation_id, error=str(e))
        answer.append(f"\n\n（生成回答时出错：{e}）")
        yield answer[-1]
    finally:
        # Also runs when the client disconnects mid-stream: keep what was generated.
        _update_message(assistant_message_id, content="".join(answer), model_name=model_name)
    yield f"[DONE]:{conversation_id}"


# ---------- read side ----------

def _ts(value) -> str | None:
    return value.isoformat() if value else None


def list_conversations(user_id: str | None) -> list[dict]:
    with session_scope() as s:
        rows = s.scalars(
            select(ChatConversation)
            .where(ChatConversation.user_id == (user_id or DEFAULT_USER), ChatConversation.deleted.is_(False))
            .order_by(ChatConversation.updated_at.desc())
        ).all()
        return [{"conversationId": c.conversation_id, "userId": c.user_id, "title": c.title,
                 "createTime": _ts(c.created_at), "updateTime": _ts(c.updated_at)} for c in rows]


def list_messages(conversation_id: str) -> list[dict]:
    with session_scope() as s:
        rows = s.scalars(
            select(ChatMessage)
            .where(ChatMessage.conversation_id == conversation_id, ChatMessage.deleted.is_(False))
            .order_by(ChatMessage.created_at, ChatMessage.id)
        ).all()
        return [{"messageId": m.message_id, "conversationId": m.conversation_id, "type": m.type,
                 "content": m.content, "transformContent": m.transform_content, "modelName": m.model_name,
                 "ragReferences": m.rag_references, "createTime": _ts(m.created_at)} for m in rows]


def delete_conversation(conversation_id: str) -> bool:
    with session_scope() as s:
        s.query(ChatMessage).filter(ChatMessage.conversation_id == conversation_id).delete()
        return s.query(ChatConversation).filter(ChatConversation.conversation_id == conversation_id).delete() > 0
