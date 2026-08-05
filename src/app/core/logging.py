"""Structured (JSON-capable) logging, configured once at process start.

Every module gets its logger via `get_logger(__name__)` — never
`print()`/bare `logging.getLogger`. `bind_request_context` attaches a
request/conversation id to every log line emitted while it's active, so a
single conversation's logs can be grepped out of a shared prod log stream.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

import structlog

from app.core.config import AppEnv, Settings

_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
_conversation_id_var: ContextVar[str | None] = ContextVar("conversation_id", default=None)


def _inject_context(
    logger: object, method_name: str, event_dict: structlog.typing.EventDict
) -> structlog.typing.EventDict:
    request_id = _request_id_var.get()
    conversation_id = _conversation_id_var.get()
    if request_id:
        event_dict["request_id"] = request_id
    if conversation_id:
        event_dict["conversation_id"] = conversation_id
    return event_dict


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=settings.log_level.upper(),
    )

    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        _inject_context,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer = (
        structlog.processors.JSONRenderer()
        if settings.app_env != AppEnv.LOCAL
        else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(settings.log_level.upper())),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


@contextmanager
def bind_request_context(request_id: str, conversation_id: str | None = None) -> Iterator[None]:
    request_token = _request_id_var.set(request_id)
    conversation_token = _conversation_id_var.set(conversation_id)
    try:
        yield
    finally:
        _request_id_var.reset(request_token)
        _conversation_id_var.reset(conversation_token)
