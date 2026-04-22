from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)


def _shutdown_executor(persona) -> None:
    executor = getattr(persona, "_bg_executor", None)
    if executor is None:
        return

    try:
        executor.shutdown(wait=False, cancel_futures=False)
    except TypeError:
        try:
            executor.shutdown(wait=False)
        except Exception:
            logger.exception("Failed to shut down persona background executor")
    except Exception:
        logger.exception("Failed to shut down persona background executor")


def close_persona_resources(persona, *, close_persona: bool) -> None:
    if persona is None:
        return

    if close_persona:
        try:
            persona.close()
        except Exception:
            logger.exception("Failed to close persona")

    _shutdown_executor(persona)


def _iter_shared_clients(persona):
    memory = getattr(persona, "memory", None)
    mem0 = getattr(memory, "mem0", None)
    if mem0 is None:
        return

    for attr in ("vector_store", "_telemetry_vector_store"):
        vector_store = getattr(mem0, attr, None)
        client = getattr(vector_store, "client", None)
        if client is not None:
            yield client


def close_shared_memory_clients(personas: Iterable) -> None:
    seen = set()
    for persona in personas:
        if persona is None:
            continue
        for client in _iter_shared_clients(persona):
            client_id = id(client)
            if client_id in seen:
                continue
            seen.add(client_id)
            try:
                client.close()
            except Exception:
                logger.exception("Failed to close shared memory client")


__all__ = ["close_persona_resources", "close_shared_memory_clients"]
