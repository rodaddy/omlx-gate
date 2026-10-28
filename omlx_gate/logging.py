"""Flow-instrumentation decorator for the gate modules (local Rodaddy patch).

One decorator owns the five-point logging story instead of hand-written call
sites: entry (DEBUG), exit with duration (DEBUG), failure (ERROR with
exc_info, then re-raise). DEBUG is silent unless debugging is on; the failure
arm guarantees no decorated path can die without a trace. Explicit log calls
remain for what a wrapper cannot know: guard triggers (WARNING) and
domain-meaningful results (INFO).

Stdlib logging to match the surrounding oMLX codebase; in our own
applications the same decorator wraps loguru (see
_DOCS/STANDARDS-python.md "flow instrumentation is ONE decorator").
"""

from __future__ import annotations

import functools
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any


def logged(func: Callable) -> Callable:
    """Entry/exit/duration/failure logging on the function's own logger."""
    label = func.__qualname__
    logger = logging.getLogger(func.__module__)

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger.debug("GATE: enter %s", label)
            started = time.monotonic()
            try:
                result = await func(*args, **kwargs)
            except Exception:
                logger.error(
                    "GATE: fail %s after %.3fs",
                    label,
                    time.monotonic() - started,
                    exc_info=True,
                )
                raise
            logger.debug("GATE: exit %s in %.3fs", label, time.monotonic() - started)
            return result

        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger.debug("GATE: enter %s", label)
        started = time.monotonic()
        try:
            result = func(*args, **kwargs)
        except Exception:
            logger.error(
                "GATE: fail %s after %.3fs",
                label,
                time.monotonic() - started,
                exc_info=True,
            )
            raise
        logger.debug("GATE: exit %s in %.3fs", label, time.monotonic() - started)
        return result

    return wrapper
