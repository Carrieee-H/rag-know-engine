"""/chat routes: streaming chat, conversation list / delete, message history."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from know_engine.chat import service

router = APIRouter(prefix="/chat")


def sse(payloads):
    """text/event-stream framing: one event per payload, one ``data:`` line per text line."""
    for payload in payloads:
        yield "".join(f"data:{line}\n" for line in payload.split("\n")) + "\n"


@router.post("/send")
def send(content: str, conversationId: str | None = None, userId: str | None = None):
    return StreamingResponse(
        sse(service.chat_events(content, conversationId, userId)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/conversation/list")
def conversations(userId: str | None = None):
    data = service.list_conversations(userId)
    return {"success": True, "message": None, "data": data, "total": len(data)}


@router.delete("/conversation/delete")
def delete_conversation(conversationId: str):
    ok = service.delete_conversation(conversationId)
    return {"success": ok, "message": "会话已删除" if ok else "删除失败", "data": None, "total": None}


@router.get("/message/list")
def messages(conversationId: str):
    data = service.list_messages(conversationId)
    return {"success": True, "message": None, "data": data, "total": len(data)}
