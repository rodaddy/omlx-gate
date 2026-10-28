# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the big-model admission gate (local patch).

Behavior under test, not internals: classification, the measured-throughput
limit math, window expiry, admission rejections with their status codes and
feedback headers, cold-load exclusion from service-time samples, and the
/usage document.
"""

import json
import time
from unittest.mock import patch

import pytest

import omlx_gate.middleware as gate
from omlx_gate.middleware import (
    BigModelGateMiddleware,
    Rejection,
    RequestContext,
    is_big,
    model_size_b,
)
from omlx_gate.store import GateStore


def make_gate(app=None, store=None):
    with patch.object(gate, "_check_bind"), patch.object(gate, "_check_singleton"):
        return BigModelGateMiddleware(app, store=store or GateStore(""))


def http_scope(path="/v1/chat/completions", method="POST", client=("10.0.0.9", 1)):
    return {"type": "http", "path": path, "method": method, "client": client}


def body_receive(payload: dict):
    body = json.dumps(payload).encode()
    sent = False

    async def receive():
        nonlocal sent
        if sent:
            raise AssertionError("receive called twice")
        sent = True
        return {"type": "http.request", "body": body, "more_body": False}

    return receive


class SendRecorder:
    def __init__(self):
        self.messages = []

    async def __call__(self, message):
        self.messages.append(message)

    @property
    def status(self):
        return self.messages[0]["status"]

    @property
    def headers(self):
        return {k.decode(): v.decode() for k, v in self.messages[0]["headers"]}

    @property
    def body(self):
        return json.loads(b"".join(m.get("body", b"") for m in self.messages[1:]))


def ok_app(status=200):
    async def app(scope, receive, send):
        await receive()
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    return app


# -- classification -----------------------------------------------------------


@pytest.mark.parametrize(
    "model,size",
    [
        ("Qwen3-4B-Instruct-2507-4bit", 4.0),
        ("Qwen3-30B-A3B-4bit", 30.0),
        ("Qwen3.8-27B-4bit", 27.0),
        ("gemma-3-1b-it", 1.0),
        ("weird-model", None),
        (None, None),
    ],
)
def test_model_size_parses_param_count_not_quant_bits(model, size):
    assert model_size_b(model) == size


def test_unparseable_models_fail_closed_as_big():
    assert is_big(None) and is_big("weird-model")
    assert not is_big("Qwen3-4B-Instruct-2507-4bit")


# -- measured limit math ------------------------------------------------------


def test_limit_is_fallback_until_min_samples_exist():
    g = make_gate()
    g._record_duration("m-27B", 57.0)
    g._record_duration("m-27B", 57.0)
    assert g._rpm_limit("m-27B") == gate.BIG_RPM_FALLBACK


def test_limit_tracks_measured_speed_with_floor_and_ceiling():
    g = make_gate()
    for _ in range(5):
        g._record_duration("slow-27B", 57.0)
        g._record_duration("moe-30B", 7.0)
        g._record_duration("fast-30B", 1.8)
    assert g._rpm_limit("slow-27B") == gate.MIN_RPM
    assert g._rpm_limit("moe-30B") == 12
    assert g._rpm_limit("fast-30B") == gate.MAX_RPM


def test_suggested_interval_prefers_freshest_measurements():
    g = make_gate()
    for _ in range(5):
        g._record_duration("m-27B", 10.0)
    g._record_duration("m-27B", 2.0)
    g._record_duration("m-27B", 2.0)
    # last two samples average 2.0s; interval = 2.0 / (slots * share)
    assert g._suggested_interval("m-27B") == pytest.approx(
        2.0 / (gate.BIG_CONCURRENCY * gate.SHARE)
    )


def test_rate_window_expires_after_sixty_seconds():
    g = make_gate()
    window = g._window("1.2.3.4", "m-27B")
    window.append(time.monotonic() - 61.0)
    assert len(g._window("1.2.3.4", "m-27B")) == 0


# -- admission checks ---------------------------------------------------------


def ctx_for(g, client="10.0.0.9", model="m-27B"):
    return RequestContext(
        client=client,
        model=model,
        limit=g._rpm_limit(model),
        window=g._window(client, model),
    )


def test_rate_check_rejects_past_limit_with_computed_retry_after():
    g = make_gate()
    ctx = ctx_for(g)
    for _ in range(ctx.limit):
        ctx.window.append(time.monotonic())
    rejection = g._check_rate(ctx)
    assert isinstance(rejection, Rejection)
    assert rejection.status == 429
    assert 1 <= rejection.retry_after <= 61


def test_rate_check_exempts_loopback():
    g = make_gate()
    ctx = ctx_for(g, client="127.0.0.1")
    for _ in range(ctx.limit + 5):
        ctx.window.append(time.monotonic())
    assert g._check_rate(ctx) is None


def test_concurrency_check_rejects_when_slots_full():
    g = make_gate()
    g._in_flight = gate.BIG_CONCURRENCY
    rejection = g._check_concurrency(ctx_for(g))
    assert rejection is not None and rejection.status == 429


def test_fit_check_names_resident_model_and_swap_commands():
    g = make_gate()
    pool = {
        "final_ceiling": 26.8e9,
        "current_model_memory": 19.2e9,
        "models": [
            {
                "id": "want-30B",
                "loaded": False,
                "is_loading": False,
                "estimated_size": 18.0e9,
            },
            {
                "id": "resident-27B",
                "loaded": True,
                "is_loading": False,
                "actual_size": 16.5e9,
            },
        ],
    }
    with patch.object(g, "_pool_status", return_value=pool):
        rejection = g._check_fit(ctx_for(g, model="want-30B"))
    assert rejection is not None
    assert rejection.status == 507
    assert rejection.code == "model_swap_required"
    assert "resident-27B" in rejection.message
    assert "/v1/models/want-30B/load" in rejection.message


def test_fit_check_passes_when_model_fits_in_headroom():
    g = make_gate()
    pool = {
        "final_ceiling": 26.8e9,
        "current_model_memory": 2.0e9,
        "models": [
            {
                "id": "want-30B",
                "loaded": False,
                "is_loading": False,
                "estimated_size": 18.0e9,
            },
        ],
    }
    with patch.object(g, "_pool_status", return_value=pool):
        assert g._check_fit(ctx_for(g, model="want-30B")) is None


# -- end-to-end through the ASGI surface --------------------------------------


async def test_small_model_passes_without_feedback_headers():
    g = make_gate(ok_app())
    send = SendRecorder()
    await g(http_scope(), body_receive({"model": "Qwen3-4B-Instruct-2507-4bit"}), send)
    assert send.status == 200
    assert "x-ratelimit-limit-requests" not in send.headers


async def test_big_model_gets_feedback_headers_on_success():
    g = make_gate(ok_app())
    with (
        patch.object(g, "_model_is_loaded", return_value=True),
        patch.object(g, "_pool_status", return_value=None),
    ):
        send = SendRecorder()
        await g(http_scope(), body_receive({"model": "Qwen3.8-27B-4bit"}), send)
    assert send.status == 200
    assert send.headers["x-ratelimit-limit-requests"] == str(gate.BIG_RPM_FALLBACK)


async def test_rejection_body_is_openai_error_shape_with_retry_after():
    g = make_gate(ok_app())
    g._in_flight = gate.BIG_CONCURRENCY
    send = SendRecorder()
    await g(http_scope(), body_receive({"model": "Qwen3.8-27B-4bit"}), send)
    assert send.status == 429
    assert "retry-after" in send.headers
    error = send.body["error"]
    assert error["type"] == "rate_limit_error"
    assert error["code"] == "rate_limit_exceeded"


async def test_cold_load_requests_are_not_service_time_samples():
    g = make_gate(ok_app())
    with (
        patch.object(g, "_model_is_loaded", return_value=False),
        patch.object(g, "_pool_status", return_value=None),
    ):
        await g(
            http_scope(), body_receive({"model": "Qwen3.8-27B-4bit"}), SendRecorder()
        )
    assert "Qwen3.8-27B-4bit" not in g._service_time


async def test_warm_2xx_requests_are_recorded_as_samples():
    g = make_gate(ok_app())
    with (
        patch.object(g, "_model_is_loaded", return_value=True),
        patch.object(g, "_pool_status", return_value=None),
        patch.object(
            gate.time, "monotonic", side_effect=(100.0 + i for i in range(50))
        ),
    ):
        await g(
            http_scope(), body_receive({"model": "Qwen3.8-27B-4bit"}), SendRecorder()
        )
    assert g._service_time["Qwen3.8-27B-4bit"][1] == 1


async def test_failed_requests_are_not_recorded_as_samples():
    g = make_gate(ok_app(status=507))
    with (
        patch.object(g, "_model_is_loaded", return_value=True),
        patch.object(g, "_pool_status", return_value=None),
    ):
        await g(
            http_scope(), body_receive({"model": "Qwen3.8-27B-4bit"}), SendRecorder()
        )
    assert "Qwen3.8-27B-4bit" not in g._service_time


async def test_usage_endpoint_states_the_contract():
    g = make_gate(ok_app())
    with patch.object(g, "_pool_status", return_value=None):
        send = SendRecorder()
        await g(http_scope(path="/usage", method="GET"), body_receive({}), send)
    assert send.status == 200
    doc = send.body
    assert "policy" in doc and "actions" in doc and "memory" in doc
    assert doc["big_models_in_flight"] == f"0/{gate.BIG_CONCURRENCY}"


async def test_slot_is_released_after_inner_app_crash():
    async def crashing_app(scope, receive, send):
        raise RuntimeError("engine died")

    g = make_gate(crashing_app)
    with (
        patch.object(g, "_model_is_loaded", return_value=True),
        patch.object(g, "_pool_status", return_value=None),
        pytest.raises(RuntimeError),
    ):
        await g(
            http_scope(),
            body_receive({"model": "Qwen3.8-27B-4bit"}),
            SendRecorder(),
        )
    assert g._in_flight == 0


# -- persistence --------------------------------------------------------------


def test_measured_limits_survive_a_restart(tmp_path):
    db = str(tmp_path / "gate.db")
    g1 = make_gate(store=GateStore(db))
    for _ in range(5):
        g1._record_duration("slow-27B", 57.0)
    assert g1._rpm_limit("slow-27B") == gate.MIN_RPM
    # a fresh gate on the same db starts with the measured limit, not fallback
    g2 = make_gate(store=GateStore(db))
    assert g2._rpm_limit("slow-27B") == gate.MIN_RPM


def test_rejections_land_in_the_event_record(tmp_path):
    db = str(tmp_path / "gate.db")
    store = GateStore(db)
    g = make_gate(store=store)
    g._in_flight = gate.BIG_CONCURRENCY
    ctx = ctx_for(g)
    rejection = g._check_concurrency(ctx)
    store.record_event(ctx.client, ctx.model, rejection.log_reason, rejection.status)
    rows = store._conn.execute(
        "SELECT client, model, outcome, status FROM events"
    ).fetchall()
    assert rows == [("10.0.0.9", "m-27B", "concurrency cap", 429)]


# -- swap gate ----------------------------------------------------------------


async def swap(g, verb, model="big-27B", client=("10.0.0.9", 1)):
    send = SendRecorder()
    await g(
        http_scope(path=f"/v1/models/{model}/{verb}", client=client),
        body_receive({}),
        send,
    )
    return send


async def test_fourth_load_in_a_minute_is_roulette():
    g = make_gate(ok_app())
    for _ in range(gate.SWAP_LOADS_PER_MIN):
        assert (await swap(g, "load")).status == 200
    send = await swap(g, "load")
    assert send.status == 429
    assert "Please slow down." in send.body["error"]["message"]
    assert send.body["error"]["code"] == "swap_rate_exceeded"


async def test_unload_is_never_rate_limited():
    g = make_gate(ok_app())
    for _ in range(gate.SWAP_LOADS_PER_MIN + 5):
        assert (await swap(g, "unload")).status == 200


async def test_rate_limit_message_asks_caller_to_slow_down():
    g = make_gate(ok_app())
    ctx = ctx_for(g)
    for _ in range(ctx.limit):
        ctx.window.append(time.monotonic())
    assert "Please slow down." in g._check_rate(ctx).message
