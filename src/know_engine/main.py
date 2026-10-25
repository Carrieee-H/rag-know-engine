"""FastAPI entry point: ``uvicorn know_engine.main:app --port 8009``."""

import threading
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from know_engine.api import chat, document
from know_engine.api import eval as eval_api
from know_engine.clients import es, storage
from know_engine.config import PROJECT_ROOT
from know_engine.ingest import worker
from know_engine.ingest.errors import ConflictError, NotFoundError
from know_engine.rag import retrieve

log = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    es.ensure_index()
    storage.get_minio()  # creates the bucket if needed
    stop = worker.start_background()
    # Load the reranker now rather than on the first question (~6 s).
    threading.Thread(target=retrieve._reranker, daemon=True).start()
    yield
    stop.set()


app = FastAPI(title="KnowEngine", lifespan=lifespan)
app.include_router(document.router)
app.include_router(chat.router)
app.include_router(eval_api.router)


# Errors come back as plain text: the pages show the response body as the message.
@app.exception_handler(ConflictError)
@app.exception_handler(ValueError)  # invalid input from splitters/converters, like Java's IllegalArgumentException
async def conflict(_: Request, e: ValueError):
    return PlainTextResponse(str(e), status_code=409)


@app.exception_handler(NotFoundError)
async def not_found(_: Request, e: NotFoundError):
    return PlainTextResponse(str(e), status_code=404)


@app.exception_handler(Exception)
async def internal_error(_: Request, e: Exception):
    log.exception("unhandled_error", error=str(e))
    return PlainTextResponse(f"服务器内部错误: {e}", status_code=500)


@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse("/document.html")


# Static pages last, so API routes take precedence.
app.mount("/", StaticFiles(directory=PROJECT_ROOT / "static", html=True), name="static")
