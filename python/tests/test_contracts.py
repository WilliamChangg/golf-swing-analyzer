"""Contract model tests.

These guard the desktop <-> engine boundary. A change that breaks serialization
here breaks the UI, and the generated TypeScript along with it.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from analyzer.contracts.health import (
    HEALTH_SCHEMA_VERSION,
    ComponentStatus,
    ComputeInfo,
    EnvironmentReport,
    HealthStatus,
    PlatformInfo,
    aggregate_status,
)
from analyzer.contracts.rpc import EngineError, ErrorCode, RpcRequest


def _component(name: str, status: HealthStatus) -> ComponentStatus:
    return ComponentStatus(name=name, status=status, detail="probe result")


class TestAggregateStatus:
    def test_all_ok_is_ok(self) -> None:
        components = [_component("a", HealthStatus.OK), _component("b", HealthStatus.OK)]
        assert aggregate_status(components) is HealthStatus.OK

    @pytest.mark.parametrize(
        ("statuses", "expected"),
        [
            ([HealthStatus.OK, HealthStatus.DEGRADED], HealthStatus.DEGRADED),
            ([HealthStatus.OK, HealthStatus.MISSING], HealthStatus.MISSING),
            ([HealthStatus.DEGRADED, HealthStatus.MISSING], HealthStatus.MISSING),
            ([HealthStatus.MISSING, HealthStatus.ERROR], HealthStatus.ERROR),
            ([HealthStatus.ERROR, HealthStatus.OK], HealthStatus.ERROR),
        ],
    )
    def test_worst_status_wins(self, statuses: list[HealthStatus], expected: HealthStatus) -> None:
        components = [_component(f"c{i}", s) for i, s in enumerate(statuses)]
        assert aggregate_status(components) is expected

    def test_empty_is_error_not_ok(self) -> None:
        """Probing nothing is a failure, not a pass.

        Reporting OK for an empty component list would be exactly the kind of
        false green this project exists to avoid.
        """
        assert aggregate_status([]) is HealthStatus.ERROR


class TestEnvironmentReport:
    def _report(self) -> EnvironmentReport:
        return EnvironmentReport(
            generated_at=datetime.now(UTC),
            overall_status=HealthStatus.OK,
            platform=PlatformInfo(
                system="Darwin",
                release="25.4.0",
                machine="arm64",
                cpu_count=10,
                python_version="3.12.14",
                python_executable="/usr/bin/python3",
            ),
            components=[_component("ffmpeg", HealthStatus.OK)],
            compute=ComputeInfo(
                torch_version="2.14.0",
                mps_available=True,
                mps_built=True,
                cuda_available=False,
                selected_device="mps",
                cpu_fallback_available=True,
                mediapipe_delegate="cpu",
                ffmpeg_hwaccels=["videotoolbox"],
            ),
        )

    def test_round_trips_through_json(self) -> None:
        original = self._report()
        restored = EnvironmentReport.model_validate_json(original.model_dump_json())
        assert restored.platform == original.platform
        assert restored.compute == original.compute
        assert restored.components == original.components

    def test_schema_version_defaults_to_current(self) -> None:
        assert self._report().schema_version == HEALTH_SCHEMA_VERSION

    def test_json_mode_dump_is_serializable(self) -> None:
        """The worker dumps with mode='json'; datetimes must survive that."""
        import json

        payload = self._report().model_dump(mode="json")
        assert isinstance(json.dumps(payload), str)
        assert isinstance(payload["generated_at"], str)

    def test_status_serializes_as_plain_string(self) -> None:
        """The TypeScript side expects string unions, not nested enum objects."""
        payload = self._report().model_dump(mode="json")
        assert payload["overall_status"] == "ok"
        assert payload["components"][0]["status"] == "ok"


class TestRpcContracts:
    def test_request_defaults_params_to_empty_dict(self) -> None:
        request = RpcRequest(id=1, method="doctor")
        assert request.params == {}

    def test_engine_error_carries_code_into_rpc_error(self) -> None:
        error = EngineError("nope", code=ErrorCode.METHOD_NOT_FOUND, data={"method": "x"})
        rpc_error = error.to_rpc_error()
        assert rpc_error.code == ErrorCode.METHOD_NOT_FOUND
        assert rpc_error.message == "nope"
        assert rpc_error.data == {"method": "x"}

    def test_engine_error_defaults_to_internal(self) -> None:
        assert EngineError("boom").to_rpc_error().code == ErrorCode.INTERNAL_ERROR
