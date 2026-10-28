"""The ONE place omlx-gate touches oMLX internals.

Everything else in this package is stdlib + FastAPI. This module reads the
engine pool's status so /usage can report what is loaded and how much
headroom remains, and so admission can tell "resident but idle" from
"needs a swap". When an oMLX upgrade breaks the gate, it breaks HERE and
only here -- that is the whole point of isolating it.

Coupling surface, verified against oMLX 0.6.0.dev1:
  omlx.server._server_state.engine_pool  -> EnginePool | None
  EnginePool.get_status()                -> {"models": [{"id", "loaded", ...}, ...], ...}

`server` is imported lazily because omlx.server builds the FastAPI app at
import time and importing it from the gate at module load would be circular
when the gate is registered onto that same app.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def pool_status() -> dict[str, Any] | None:
    """Engine pool status, or None with a logged reason.

    Never raises: an unreadable pool degrades /usage to "unavailable" and
    admission to "cannot determine residency", which the callers already
    handle. A dead server is worse than a gate that cannot see memory.
    """
    try:
        from omlx import server  # noqa: PLC0415 - lazy, see module docstring

        pool = server._server_state.engine_pool
        if pool is None:
            logger.debug("GATE: engine pool not initialized yet")
            return None
        return pool.get_status()
    except Exception:
        logger.warning("GATE: could not read engine pool state", exc_info=True)
        return None
