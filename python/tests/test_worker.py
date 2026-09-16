"""JSON-RPC worker protocol tests.

Driving the worker over in-memory streams keeps these tests fast and free of
subprocess flakiness, while still exercising the exact framing the Rust client
depends on.
"""

from __future__ import annotations

import io
import json
from typing import Any

from analyzer.contracts.rpc import ErrorCode
from analyzer.worker import Worker


def run_worker(lines: list[str]) -> list[dict[str, Any]]:
    """Feed lines to a worker and return every frame it emitted."""
    stdin = io.StringIO("".join(f"{line}\n" for line in lines))
    out = io.StringIO()
    Worker(stdin=stdin, protocol_out=out).serve_forever()
    return [json.loads(raw) for raw in out.getvalue().splitlines() if raw.strip()]


def request(request_id: int | str, method: str, **params: Any) -> str:
    payload: dict[str, Any] = {"jsonrpc": "2.0", "id": request_id, "method": method}
    if params:
        payload["params"] = params
    return json.dumps(payload)


class TestHandshake:
    def test_emits_ready_notification_before_any_request(self) -> None:
        frames = run_worker([])
        assert len(frames) == 1
        assert frames[0]["method"] == "ready"
        assert "id" not in frames[0], "notifications must not carry an id"

    def test_ready_advertises_schema_version_and_methods(self) -> None:
        from analyzer.contracts.health import HEALTH_SCHEMA_VERSION

        params = run_worker([])[0]["params"]
        assert params["health_schema_version"] == HEALTH_SCHEMA_VERSION
        assert "doctor" in params["methods"]


class TestFraming:
    def test_one_json_object_per_line(self) -> None:
        frames = run_worker([request(1, "doctor"), request(2, "doctor")])
        # ready + two responses
        assert len(frames) == 3
        assert [f.get("id") for f in frames[1:]] == [1, 2]

    def test_blank_lines_are_ignored(self) -> None:
        frames = run_worker(["", "   ", request(1, "doctor")])
        assert len(frames) == 2

    def test_string_ids_are_preserved(self) -> None:
        frames = run_worker([request("req-abc", "doctor")])
        assert frames[1]["id"] == "req-abc"

    def test_responses_are_correlated_in_order(self) -> None:
        frames = run_worker([request(i, "doctor") for i in range(1, 4)])
        assert [f["id"] for f in frames[1:]] == [1, 2, 3]


class TestErrorHandling:
    def test_invalid_json_yields_parse_error_with_null_id(self) -> None:
        frames = run_worker(["{not json"])
        error_frame = frames[1]
        assert error_frame["error"]["code"] == ErrorCode.PARSE_ERROR
        assert error_frame["id"] is None

    def test_non_object_payload_is_invalid_request(self) -> None:
        frames = run_worker(["[1, 2, 3]"])
        assert frames[1]["error"]["code"] == ErrorCode.INVALID_REQUEST

    def test_unknown_method_yields_method_not_found(self) -> None:
        frames = run_worker([request(7, "nope")])
        assert frames[1]["error"]["code"] == ErrorCode.METHOD_NOT_FOUND
        assert frames[1]["id"] == 7

    def test_malformed_request_still_reports_its_id(self) -> None:
        """Losing the id would orphan the caller's pending promise and hang the UI."""
        frames = run_worker([json.dumps({"jsonrpc": "2.0", "id": 42})])  # no method
        assert frames[1]["id"] == 42
        assert frames[1]["error"]["code"] == ErrorCode.INVALID_REQUEST

    def test_error_does_not_stop_the_loop(self) -> None:
        frames = run_worker(["{bad", request(2, "doctor")])
        assert frames[1]["error"]["code"] == ErrorCode.PARSE_ERROR
        assert frames[2]["id"] == 2
        assert "result" in frames[2]


class TestDoctorOverRpc:
    def test_returns_a_serializable_report(self) -> None:
        frames = run_worker([request(1, "doctor")])
        result = frames[1]["result"]
        assert result["schema_version"] >= 1
        assert isinstance(result["components"], list)
        assert result["components"], "doctor must probe at least one component"
        assert "compute" in result
        assert isinstance(result["generated_at"], str)

    def test_every_component_has_a_status_string(self) -> None:
        result = run_worker([request(1, "doctor")])[1]["result"]
        valid = {"ok", "degraded", "missing", "error"}
        for component in result["components"]:
            assert component["status"] in valid, component
