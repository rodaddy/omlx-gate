"""Big-model admission gate with measured, self-adjusting rate limits.

Local Rodaddy patch (not upstream). This box supplies occasional inference,
not a data center. Small models (<= OMLX_GATE_SMALL_MAX_B params by name)
pass untouched; anything larger is admission-controlled:

- At most OMLX_GATE_BIG_CONCURRENCY requests in flight at once, globally.
  Memory cannot hold more than that anyway; the rest get an immediate 429
  instead of piling into the batcher.
- A per-IP, per-model sliding rate limit DERIVED FROM MEASURED SERVICE TIME:
  an EWMA of each model's real end-to-end request seconds, excluding
  cold-load requests (one 66s load-included sample was observed to publish
  a limit ~25x too strict for a model that really takes ~2.7s).

      rpm(model) = clamp(60s * slots * share / ewma_seconds, min, max)

- A memory fit pre-check: a big model that cannot fit beside what is loaded
  is refused with the resident model named, whether it is in use, and the
  exact swap commands -- instead of oMLX's kernel-tuning 507 text.
- Feedback for good citizens on every gated response: x-ratelimit-* window
  headers, measured avg/last request seconds, a suggested pacing interval,
  and computed Retry-After values. GET /usage (or /v1/usage) states the
  whole live contract in one document.

Pure ASGI, modeled on DebugRequestLoggingMiddleware in server.py: no
BaseHTTPMiddleware, so streaming responses are not wrapped in a pipe layer.
Uses stdlib logging to match the surrounding oMLX codebase (house loguru
standard applies to our own applications, not to patches inside upstream
code); the @_logged decorator supplies entry/exit/duration/failure logging
so call sites do not hand-write the five points.

Env knobs (read at import):
    OMLX_GATE_ENABLED=1            master switch
    OMLX_GATE_SMALL_MAX_B=4        largest param count (in B) that passes free
    OMLX_GATE_BIG_CONCURRENCY=2    global in-flight cap for bigger models
    OMLX_GATE_BIG_RPM=10           per-IP/min fallback before samples exist
    OMLX_GATE_SHARE=0.75           fraction of measured capacity one IP may use
    OMLX_GATE_MIN_RPM=1            adaptive floor
    OMLX_GATE_MAX_RPM=30           adaptive ceiling
    OMLX_GATE_MIN_SAMPLES=3        completed requests before adapting
    OMLX_GATE_EXEMPT_LOOPBACK=1    loopback skips the per-IP window
                                   (never the concurrency cap - memory is memory)
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import time
from collections import deque
from collections.abc import Callable
from typing import Any, NamedTuple

from omlx_gate.store import GateStore
from omlx_gate.logging import logged as _logged
from omlx_gate.pool import pool_status as _pool_status_impl

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration (env, read once at import)
# =============================================================================


def _env_flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "")


def _env_num(name: str, default: str) -> float:
    raw = os.environ.get(name, default)
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "GATE: invalid numeric env %s=%r, using default %s", name, raw, default
        )
        return float(default)


GATE_ENABLED = _env_flag("OMLX_GATE_ENABLED", "1")
SMALL_MAX_B = _env_num("OMLX_GATE_SMALL_MAX_B", "4")
BIG_CONCURRENCY = int(_env_num("OMLX_GATE_BIG_CONCURRENCY", "2"))
BIG_RPM_FALLBACK = int(_env_num("OMLX_GATE_BIG_RPM", "10"))
SHARE = _env_num("OMLX_GATE_SHARE", "0.75")
MIN_RPM = int(_env_num("OMLX_GATE_MIN_RPM", "1"))
MAX_RPM = int(_env_num("OMLX_GATE_MAX_RPM", "30"))
MIN_SAMPLES = int(_env_num("OMLX_GATE_MIN_SAMPLES", "3"))
EXEMPT_LOOPBACK = _env_flag("OMLX_GATE_EXEMPT_LOOPBACK", "1")

_EWMA_ALPHA = 0.2
_WINDOW_SECONDS = 60.0
_REJECT_LOG_INTERVAL_S = 30.0

# Generation endpoints only. Embeddings/rerank/audio run tiny models and are
# not what melts the box.
GATED_PATHS = frozenset(
    {
        "/v1/completions",
        "/v1/chat/completions",
        "/v1/messages",
        "/v1/responses",
    }
)

USAGE_PATHS = frozenset({"/usage", "/v1/usage"})

# Model swap endpoints. Unload is always allowed (freeing memory is good
# citizenship, sub-second measured); loads are capped to stop swap roulette
# -- each big-model load costs 6-16s measured, so a caller demanding a 4th
# load inside a minute is thrashing, not working.
_SWAP_RE = re.compile(r"^/v1/models/(?P<model>[^/]+)/(?P<verb>load|unload)$")
SWAP_LOADS_PER_MIN = int(_env_num("OMLX_GATE_SWAPS_PER_MIN", "3"))

# House editorial, per Rico: rate-limit rejections carry the message.
_TREBEK = " Suck it, Trebek."

# Param counts embedded in model names: "27B", "3.8B", the "3B" inside
# "A3B". Must NOT match "4bit"/"8bit", so the B cannot be followed by an
# alphanumeric. The largest match wins: "Qwen3-30B-A3B-4bit" -> 30 (total
# params bound memory, not active params).
_PARAM_RE = re.compile(r"(\d+(?:\.\d+)?)\s*[Bb](?![A-Za-z0-9])")

_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


# =============================================================================
# Model classification
# =============================================================================


@functools.lru_cache(maxsize=256)
def model_size_b(model: str | None) -> float | None:
    """Largest parameter count (in billions) named in the model id, or None.

    LRU-cached: the same few model ids arrive on every request, and this is
    a pure function of its argument, so the regex runs once per distinct id."""
    if not model:
        return None
    sizes = [float(m) for m in _PARAM_RE.findall(model)]
    return max(sizes) if sizes else None


def is_big(model: str | None) -> bool:
    """Unknown or unparseable sizes are treated as big - fail closed."""
    size = model_size_b(model)
    return size is None or size > SMALL_MAX_B


# =============================================================================
# Startup tripwires
# =============================================================================


def _bound_host() -> str | None:
    """The --host this process was started with, if discoverable."""
    import sys

    argv = sys.argv
    for i, arg in enumerate(argv):
        if arg == "--host" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--host="):
            return arg.split("=", 1)[1]
    return None


def _check_bind() -> None:
    """Tripwire: this box serves the LAN. A loopback bind is a regression
    (it has happened before) - external clients and the rate-limit story
    all silently die. Scream at startup so the log names the mistake."""
    host = _bound_host()
    if host in _LOOPBACK:
        logger.error(
            "GATE: BIND REGRESSION - omlx started with --host %s. This server "
            "is meant to serve the LAN on 0.0.0.0 (see the LaunchAgent plist "
            "and the service docs). External clients cannot "
            "reach it.",
            host,
        )


def _check_singleton() -> None:
    """Tripwire: exactly ONE oMLX serves this box, owned by launchd
    (com.example.omlx). A second instance -- typically a hand-run
    `omlx serve` -- steals or shadows the bind (loopback shadowing a
    wildcard bind broke aqmd twice on this fleet). Scream if another
    server process exists; never start one by hand."""
    import subprocess

    try:
        out = subprocess.run(
            ["/usr/bin/pgrep", "-fl", "omlx"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except Exception:
        logger.warning("GATE: singleton check could not run pgrep", exc_info=True)
        return
    me = os.getpid()
    others = [
        line
        for line in out.splitlines()
        if ("omlx-server" in line or "omlx serve" in line)
        and not line.startswith(f"{me} ")
    ]
    if others:
        logger.error(
            "GATE: SINGLETON VIOLATION - another oMLX server process is "
            "running: %s. Exactly one instance serves this box, owned by "
            "launchd (com.example.omlx). Stop the extra one; never start "
            "omlx serve by hand.",
            "; ".join(others),
        )


# =============================================================================
# Admission types
# =============================================================================


class Rejection(NamedTuple):
    """Everything needed to answer a refused request."""

    status: int
    code: str
    retry_after: int
    message: str
    log_reason: str


class RequestContext(NamedTuple):
    """One gated request's admission inputs."""

    client: str
    model: str
    limit: int
    window: deque[float]


# =============================================================================
# The gate
# =============================================================================


class BigModelGateMiddleware:
    def __init__(self, app: Any, store: GateStore | None = None) -> None:
        self.app = app
        self._in_flight = 0
        # Persistence is best-effort: a broken store degrades to in-memory
        # limits (documented fail-open), never a dead server.
        try:
            self._store: GateStore | None = store or GateStore()
        except Exception:
            logger.error("GATE: store unavailable, running in-memory", exc_info=True)
            self._store = None
        # model -> (ewma_seconds, sample_count), seeded from the store so a
        # restart does not reset measured limits to the generous fallback
        self._service_time: dict[str, tuple[float, int]] = (
            self._store.load_service_times() if self._store else {}
        )
        # model -> last <=2 completed-request durations (freshest signal)
        self._recent: dict[str, deque[float]] = {}
        # (client, model) -> deque of request monotonic timestamps, last 60s
        self._windows: dict[tuple[str, str], deque[float]] = {}
        self._last_reject_log: dict[str, float] = {}
        # global timestamps of big-model load requests (swap roulette gate)
        self._swap_loads: deque[float] = deque()
        # Ordered guard checks; adding a rule is one row, not one branch.
        self._checks: tuple[Callable[[RequestContext], Rejection | None], ...] = (
            self._check_rate,
            self._check_concurrency,
            self._check_fit,
        )
        _check_bind()
        _check_singleton()

    def _persist(self, action: Callable[[GateStore], None]) -> None:
        """Best-effort store write; a disk failure degrades persistence,
        never the request path (documented fail-open)."""
        if self._store is None:
            return
        try:
            action(self._store)
        except Exception:
            logger.warning("GATE: store write failed", exc_info=True)

    # -- measured service time ------------------------------------------------

    def _record_duration(self, model: str, seconds: float) -> None:
        prev = self._service_time.get(model)
        if prev is None:
            self._service_time[model] = (seconds, 1)
        else:
            ewma, n = prev
            self._service_time[model] = (ewma + _EWMA_ALPHA * (seconds - ewma), n + 1)
        self._recent.setdefault(model, deque(maxlen=2)).append(seconds)
        ewma, n = self._service_time[model]
        self._persist(lambda s: s.save_service_time(model, ewma, n))
        logger.debug(
            "GATE: timing model=%s took=%.1fs ewma=%.1fs samples=%d limit=%d/min",
            model,
            seconds,
            ewma,
            n,
            self._rpm_limit(model),
        )

    def _avg_seconds(self, model: str) -> float | None:
        entry = self._service_time.get(model)
        if entry is None or entry[1] < MIN_SAMPLES:
            return None
        return entry[0]

    def _last_seconds(self, model: str) -> float | None:
        """Mean of the last one or two completed requests - the freshest
        signal of what the caller should expect right now."""
        recent = self._recent.get(model)
        if not recent:
            return None
        return sum(recent) / len(recent)

    def _suggested_interval(self, model: str) -> float | None:
        """Seconds a single caller should wait between requests: the
        freshest service time divided by its fair share of the slots."""
        basis = self._last_seconds(model) or self._avg_seconds(model)
        if basis is None:
            return None
        return basis / (BIG_CONCURRENCY * SHARE)

    def _rpm_limit(self, model: str) -> int:
        """Per-IP requests/minute: a fair share of measured throughput."""
        avg = self._avg_seconds(model)
        if avg is None or avg <= 0:
            return BIG_RPM_FALLBACK
        capacity = _WINDOW_SECONDS * BIG_CONCURRENCY * SHARE / avg
        return max(MIN_RPM, min(MAX_RPM, int(capacity)))

    # -- per-IP sliding window ------------------------------------------------

    def _window(self, client: str, model: str) -> deque[float]:
        now = time.monotonic()
        if len(self._windows) > 512:
            stale = [
                k for k, w in self._windows.items() if not w or now - w[-1] > 300.0
            ]
            for key in stale:
                del self._windows[key]
            logger.debug("GATE: pruned %d stale rate windows", len(stale))
        window = self._windows.setdefault((client, model), deque())
        while window and now - window[0] > _WINDOW_SECONDS:
            window.popleft()
        return window

    # -- engine pool state ----------------------------------------------------

    @staticmethod
    def _pool_status() -> dict[str, Any] | None:
        """Engine pool status, or None with a logged reason. The oMLX
        coupling itself lives in omlx_gate.pool -- this is the single
        accessor so every consumer degrades the same way."""
        return _pool_status_impl()

    def _model_is_loaded(self, model: str) -> bool | None:
        """True/False from the engine pool, None when undeterminable."""
        status = self._pool_status()
        if status is None:
            return None
        for m in status.get("models", []):
            if m.get("id") == model:
                return bool(m.get("loaded"))
        return None

    def _memory_state(self) -> dict[str, Any]:
        """Live loaded/unloaded state and headroom for /usage, best-effort."""
        status = self._pool_status()
        if status is None:
            return {"unavailable": "engine pool state not readable"}
        loaded: list[str] = []
        loading: list[str] = []
        unloaded: list[str] = []
        buckets = {"loaded": loaded, "loading": loading, "unloaded": unloaded}
        for m in status.get("models", []):
            entry = m.get("id", "?")
            size = m.get("actual_size") or m.get("estimated_size")
            if size:
                entry = f"{entry} (~{size / 1e9:.1f}GB)"
            key = (
                "loaded"
                if m.get("loaded")
                else "loading"
                if m.get("is_loading")
                else "unloaded"
            )
            buckets[key].append(entry)
        state: dict[str, Any] = {"loaded": loaded, "unloaded": unloaded}
        if loading:
            state["loading"] = loading
        ceiling = status.get("final_ceiling")
        used = status.get("current_model_memory")
        if ceiling and used is not None:
            state["metal_ceiling_gb"] = round(ceiling / 1e9, 1)
            state["model_memory_used_gb"] = round(used / 1e9, 1)
            state["headroom_gb"] = round((ceiling - used) / 1e9, 1)
            state["note"] = (
                "A model only fits if its size is under headroom_gb."
                " Unload what you are done with to make room."
            )
        return state

    @_logged
    def _fit_conflict(self, model: str) -> dict[str, Any] | None:
        """None when the model is resident or fits; else the facts a caller
        needs to resolve the conflict: what is running, whether it is in
        use right now, and the swap commands."""
        status = self._pool_status()
        if status is None:
            return None
        me: dict[str, Any] | None = None
        resident_big: list[dict[str, Any]] = []
        for m in status.get("models", []):
            if m.get("id") == model:
                me = m
            elif (m.get("loaded") or m.get("is_loading")) and is_big(m.get("id")):
                resident_big.append(m)
        if me is None or me.get("loaded") or me.get("is_loading"):
            return None
        ceiling = status.get("final_ceiling")
        used = status.get("current_model_memory")
        size = me.get("actual_size") or me.get("estimated_size")
        if not (ceiling and used is not None and size):
            return None
        headroom = ceiling - used
        if size <= headroom or not resident_big:
            return None
        return {
            "needed_gb": round(size / 1e9, 1),
            "headroom_gb": round(headroom / 1e9, 1),
            "resident": [m.get("id") for m in resident_big],
            "in_use": self._in_flight > 0,
        }

    # -- feedback -------------------------------------------------------------

    def _limit_headers(self, model: str, limit: int, used: int) -> list:
        headers = [
            (b"x-ratelimit-limit-requests", str(limit).encode()),
            (b"x-ratelimit-remaining-requests", str(max(0, limit - used)).encode()),
        ]
        optional = (
            (b"x-omlx-avg-request-seconds", self._avg_seconds(model)),
            (b"x-omlx-last-request-seconds", self._last_seconds(model)),
            (b"x-omlx-suggested-interval-seconds", self._suggested_interval(model)),
        )
        headers.extend(
            (name, f"{value:.1f}".encode())
            for name, value in optional
            if value is not None
        )
        return headers

    def _pace_hint(self, model: str) -> str:
        pacing = self._suggested_interval(model)
        if pacing is not None:
            return (
                f" Pace yourself: one request every {pacing:.0f}s is sustainable,"
                " or fall back to a <=4B model (not rate limited)."
            )
        return (
            f" Fall back to a <={SMALL_MAX_B:g}B model (not rate limited)"
            " if you need throughput."
        )

    def _log_reject(self, ctx: RequestContext, reason: str) -> None:
        # One warning per client per interval; the rest at debug so an
        # abuse burst cannot flood the log.
        now = time.monotonic()
        last = self._last_reject_log.get(ctx.client, 0.0)
        level = (
            logging.WARNING if now - last >= _REJECT_LOG_INTERVAL_S else logging.DEBUG
        )
        self._last_reject_log[ctx.client] = now
        logger.log(
            level,
            "GATE: rejected client=%s reason=%s model=%s in_flight=%d/%d",
            ctx.client,
            reason,
            ctx.model,
            self._in_flight,
            BIG_CONCURRENCY,
        )

    # -- guard checks (ordered table consumed by __call__) --------------------

    def _check_rate(self, ctx: RequestContext) -> Rejection | None:
        if EXEMPT_LOOPBACK and ctx.client in _LOOPBACK:
            return None
        if len(ctx.window) < ctx.limit:
            return None
        # The window frees when its oldest entry ages past 60s - tell the
        # caller that real number instead of a flat minute.
        retry = max(1, int(_WINDOW_SECONDS - (time.monotonic() - ctx.window[0])) + 1)
        avg = self._avg_seconds(ctx.model)
        measured = (
            f" (measured: this model averages {avg:.0f}s per request)"
            if avg is not None
            else ""
        )
        return Rejection(
            status=429,
            code="rate_limit_exceeded",
            retry_after=retry,
            message=(
                f"Rate limit exceeded for model '{ctx.model}': this server "
                f"handles about {ctx.limit} requests/minute per client for "
                f"this model{measured}.{_TREBEK}{self._pace_hint(ctx.model)} "
                "GET /usage for the full policy."
            ),
            log_reason=f"adaptive limit {ctx.limit}/min",
        )

    def _check_concurrency(self, ctx: RequestContext) -> Rejection | None:
        if self._in_flight < BIG_CONCURRENCY:
            return None
        # A slot frees roughly half a typical request from now on average;
        # recent measurements make that number honest.
        basis = self._last_seconds(ctx.model) or self._avg_seconds(ctx.model)
        retry = max(5, int(basis / 2) + 1) if basis else 30
        return Rejection(
            status=429,
            code="rate_limit_exceeded",
            retry_after=retry,
            message=(
                f"Server busy: model '{ctx.model}' allows {BIG_CONCURRENCY} "
                "concurrent requests and all slots are in use. Retry when a "
                f"request finishes.{self._pace_hint(ctx.model)} GET /usage "
                "for the full policy."
            ),
            log_reason="concurrency cap",
        )

    def _check_fit(self, ctx: RequestContext) -> Rejection | None:
        conflict = self._fit_conflict(ctx.model)
        if conflict is None:
            return None
        running = ", ".join(conflict["resident"])
        if conflict["in_use"]:
            advice = (
                f"'{running}' is currently IN USE ({self._in_flight} request(s)"
                " in flight). Either use that model for your task, or wait"
                " until /usage shows big_models_in_flight 0/"
                f"{BIG_CONCURRENCY}, then POST /v1/models/{{id}}/unload it"
                f" and POST /v1/models/{ctx.model}/load yours."
            )
        else:
            advice = (
                f"'{running}' is loaded but idle. Either use it, or swap:"
                f" POST /v1/models/{{id}}/unload then"
                f" POST /v1/models/{ctx.model}/load."
            )
        return Rejection(
            status=507,
            code="model_swap_required",
            retry_after=30,
            message=(
                f"Model '{ctx.model}' needs ~{conflict['needed_gb']}GB but "
                f"only {conflict['headroom_gb']}GB headroom remains. {advice}"
                " GET /usage for live state."
            ),
            log_reason="does not fit in memory",
        )

    # -- model swap gate ------------------------------------------------------

    @_logged
    async def _handle_swap(
        self, scope: dict, receive: Callable, send: Callable, swap: re.Match
    ) -> None:
        """Unloads always pass (sub-second, frees memory). Loads are capped
        globally at SWAP_LOADS_PER_MIN: each costs 6-16s measured, so a 4th
        inside a minute is model roulette, not work."""
        client = (scope.get("client") or ("unknown", 0))[0]
        model, verb = swap.group("model"), swap.group("verb")
        if verb == "unload":
            self._persist(lambda s: s.record_event(client, model, "swap:unload", 200))
            await self.app(scope, receive, send)
            return
        now = time.monotonic()
        while self._swap_loads and now - self._swap_loads[0] > _WINDOW_SECONDS:
            self._swap_loads.popleft()
        if len(self._swap_loads) >= SWAP_LOADS_PER_MIN:
            retry = max(1, int(_WINDOW_SECONDS - (now - self._swap_loads[0])) + 1)
            logger.warning(
                "GATE: rejected client=%s reason=swap roulette model=%s (%d loads"
                " in the last minute)",
                client,
                model,
                len(self._swap_loads),
            )
            self._persist(lambda s: s.record_event(client, model, "swap_roulette", 429))
            payload = json.dumps(
                {
                    "error": {
                        "message": (
                            f"Model roulette: that would be load #"
                            f"{len(self._swap_loads) + 1} in the last minute on a"
                            " box where each load costs 6-16 seconds. Pick a"
                            " model and commit, or use the always-on <=4B tier."
                            f"{_TREBEK} Retry in {retry}s; GET /usage for the"
                            " policy."
                        ),
                        "type": "rate_limit_error",
                        "code": "swap_rate_exceeded",
                    }
                }
            ).encode()
            await _send_json(
                send, 429, payload, [(b"retry-after", str(retry).encode())]
            )
            return
        self._swap_loads.append(now)
        self._persist(lambda s: s.record_event(client, model, "swap:load", 200))
        await self.app(scope, receive, send)

    # -- /usage: the rules, stated up front -----------------------------------

    @_logged
    async def _serve_usage(self, send: Callable) -> None:
        models: dict[str, dict[str, Any]] = {}
        for model, (ewma, n) in sorted(self._service_time.items()):
            entry: dict[str, Any] = {
                "measured_avg_request_seconds": round(ewma, 1),
                "samples": n,
                "your_limit_requests_per_minute": self._rpm_limit(model),
            }
            last = self._last_seconds(model)
            if last is not None:
                entry["last_request_seconds"] = round(last, 1)
            interval = self._suggested_interval(model)
            if interval is not None:
                entry["suggested_seconds_between_requests"] = round(interval, 1)
            models[model] = entry
        payload = json.dumps(
            {
                "server": "oMLX on a personal Mac supplying occasional inference"
                " - not a data center",
                "policy": {
                    "small_models": f"Models <= {SMALL_MAX_B:g}B parameters are"
                    " not rate limited.",
                    "big_models": f"Larger models allow {BIG_CONCURRENCY}"
                    " concurrent requests globally; excess requests get an"
                    " immediate 429. Per client IP, requests/minute are derived"
                    " from this server's MEASURED speed for the model you call"
                    f" ({SHARE:.0%} share of real throughput, between {MIN_RPM}"
                    f" and {MAX_RPM}/min; {BIG_RPM_FALLBACK}/min until"
                    " measurements exist).",
                    "model_swaps": "Unload is always allowed and takes <1s."
                    f" Loads are capped at {SWAP_LOADS_PER_MIN}/minute globally"
                    " (each costs 6-16s measured). Swap deliberately:"
                    " unload what you are done with, load what you need,"
                    " commit to it.",
                    "feedback": "Every gated response carries"
                    " x-ratelimit-limit-requests, x-ratelimit-remaining-requests,"
                    " x-omlx-avg/last-request-seconds and"
                    " x-omlx-suggested-interval-seconds. 429s carry Retry-After."
                    " Honor them.",
                },
                "bind": _bound_host() or "unknown",
                "big_models_in_flight": f"{self._in_flight}/{BIG_CONCURRENCY}",
                "memory": self._memory_state(),
                "measured_models": models,
                "actions": {
                    "load": "POST /v1/models/{model_id}/load",
                    "unload": "POST /v1/models/{model_id}/unload",
                    "detail": "GET /v1/models/status",
                },
                "how_to_be_a_good_citizen": [
                    "Prefer the 4B-class models unless you need the big one.",
                    "Back off on 429 for at least Retry-After seconds.",
                    "Only one big model fits in memory at a time - unload the"
                    " one you are done with before asking for another.",
                ],
            },
            indent=2,
        ).encode()
        await _send_json(send, 200, payload)

    # -- ASGI -----------------------------------------------------------------

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] == "http" and scope.get("path") in USAGE_PATHS:
            await self._serve_usage(send)
            return
        if GATE_ENABLED and scope["type"] == "http" and scope.get("method") == "POST":
            swap = _SWAP_RE.match(scope.get("path", ""))
            if swap is not None:
                await self._handle_swap(scope, receive, send, swap)
                return
        gated = (
            GATE_ENABLED
            and scope["type"] == "http"
            and scope.get("method") == "POST"
            and scope.get("path") in GATED_PATHS
        )
        if not gated:
            await self.app(scope, receive, send)
            return

        body = await _read_body(receive)
        replay = _replaying_receive(body, receive)
        model = _extract_model(body)
        if not is_big(model):
            await self.app(scope, replay, send)
            return

        client = (scope.get("client") or ("unknown", 0))[0]
        ctx = RequestContext(
            client=client,
            model=model or "<unspecified>",
            limit=self._rpm_limit(model or "<unspecified>"),
            window=self._window(client, model or "<unspecified>"),
        )
        for check in self._checks:
            rejection = check(ctx)
            if rejection is None:
                continue
            self._log_reject(ctx, rejection.log_reason)
            self._persist(
                lambda s, r=rejection: s.record_event(
                    ctx.client, ctx.model, r.log_reason, r.status
                )
            )
            await self._send_rejection(send, ctx, rejection)
            return
        await self._run_admitted(scope, replay, send, ctx)

    @_logged
    async def _run_admitted(
        self, scope: dict, receive: Callable, send: Callable, ctx: RequestContext
    ) -> None:
        """Occupy a slot, pass the request through with feedback headers,
        and record its duration as a service-time sample (warm-model 2xx
        responses only - cold loads are not service time)."""
        ctx.window.append(time.monotonic())
        status_holder = {"status": 0}
        feedback = self._limit_headers(ctx.model, ctx.limit, len(ctx.window))

        async def send_with_feedback(message: dict) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message.get("status", 0)
                message["headers"] = list(message.get("headers", [])) + feedback
            await send(message)

        was_loaded = self._model_is_loaded(ctx.model)
        self._in_flight += 1
        started = time.monotonic()
        try:
            await self.app(scope, receive, send_with_feedback)
        finally:
            self._in_flight -= 1
            elapsed = time.monotonic() - started
            countable = (
                was_loaded is True
                and 200 <= status_holder["status"] < 300
                and elapsed > 0.05
            )
            if countable:
                self._record_duration(ctx.model, elapsed)
            status = status_holder["status"]
            self._persist(
                lambda s: s.record_event(
                    ctx.client,
                    ctx.model,
                    "ok" if 200 <= status < 300 else "upstream_error",
                    status,
                    elapsed,
                )
            )

    async def _send_rejection(
        self, send: Callable, ctx: RequestContext, rejection: Rejection
    ) -> None:
        error_type = (
            "rate_limit_error"
            if rejection.status == 429
            else "insufficient_memory_error"
        )
        payload = json.dumps(
            {
                "error": {
                    "message": rejection.message,
                    "type": error_type,
                    "code": rejection.code,
                }
            }
        ).encode()
        headers = [
            (b"retry-after", str(rejection.retry_after).encode()),
        ] + self._limit_headers(ctx.model, ctx.limit, len(ctx.window))
        await _send_json(send, rejection.status, payload, extra_headers=headers)


# =============================================================================
# ASGI plumbing helpers
# =============================================================================


async def _read_body(receive: Callable) -> bytes:
    """Buffer the full request body so the model field can be inspected."""
    parts = []
    while True:
        message = await receive()
        parts.append(message)
        if not message.get("more_body", False):
            break
    return b"".join(part.get("body", b"") for part in parts)


def _replaying_receive(body: bytes, receive: Callable) -> Callable:
    """A receive callable that replays the buffered body for the inner app."""
    sent = False

    async def cached_receive() -> dict:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return cached_receive


def _extract_model(body: bytes) -> str | None:
    try:
        parsed = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        logger.debug("GATE: request body not JSON, treating model as unknown")
        return None
    model = parsed.get("model") if isinstance(parsed, dict) else None
    return model if isinstance(model, str) else None


async def _send_json(
    send: Callable,
    status: int,
    payload: bytes,
    extra_headers: list | None = None,
) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(payload)).encode()),
            ]
            + (extra_headers or []),
        }
    )
    await send({"type": "http.response.body", "body": payload})
