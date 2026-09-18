from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from cmdrisk.teacher.client import (
    ConfigurationError,
    CostLedger,
    CostLimitExceeded,
    DryRunError,
    ModelConfig,
    OpenAICompatibleClient,
    SchemaValidationError,
)
from cmdrisk.teacher.responses import TeacherLabelResponse


def _response(
    content: dict[str, object],
    *,
    status: int = 200,
    usage: bool = True,
) -> httpx.Response:
    payload: dict[str, object] = {
        "choices": [{"message": {"content": json.dumps(content)}}],
    }
    if usage:
        payload["usage"] = {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        }
    return httpx.Response(status, json=payload)


def _config(**updates: object) -> ModelConfig:
    values: dict[str, object] = {
        "role": "A",
        "model": "test-model",
        "base_url": "https://teacher.invalid/v1",
        "api_key": "secret",
        "max_retries": 1,
        "input_usd_per_million": 1.0,
        "output_usd_per_million": 2.0,
        "max_output_tokens": 100,
    }
    values.update(updates)
    return ModelConfig(**values)


def _valid_label() -> dict[str, object]:
    return {
        "malicious": False,
        "first_evidence_index": None,
        "technique_evidence": {},
        "tactics": [],
        "confidence": 0.99,
    }


def test_env_config_is_resolved_at_runtime() -> None:
    config = ModelConfig.from_env(
        "A",
        environ={
            "CMDRISK_TEACHER_A_MODEL": "runtime-model",
            "CMDRISK_TEACHER_A_BASE_URL": "https://runtime.invalid/v1",
            "CMDRISK_TEACHER_A_API_KEY": "runtime-key",
            "CMDRISK_TEACHER_A_INPUT_USD_PER_MTOK": "0.5",
        },
    )
    assert config.model == "runtime-model"
    assert config.base_url == "https://runtime.invalid/v1"
    assert config.api_key == "runtime-key"
    assert config.input_usd_per_million == 0.5


def test_retry_cache_and_token_usd_ledger(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["authorization"] == "Bearer secret"
        if calls == 1:
            return httpx.Response(500, text="retry")
        return _response(_valid_label())

    ledger = CostLedger(max_cost_usd=1.0)
    client = OpenAICompatibleClient(
        _config(),
        ledger=ledger,
        cache_dir=tmp_path,
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    first = client.chat_json(
        system_prompt="system",
        user_prompt="user",
        response_model=TeacherLabelResponse,
        cache_scope="test",
    )
    second = client.chat_json(
        system_prompt="system",
        user_prompt="user",
        response_model=TeacherLabelResponse,
        cache_scope="test",
    )
    assert first == second
    assert calls == 2
    assert ledger.summary()["calls"] == 1
    assert ledger.summary()["total_tokens"] == 120
    assert ledger.spent_usd == pytest.approx(0.00014)
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_schema_validation_retries_then_fails() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response({"malicious": "yes"})

    client = OpenAICompatibleClient(
        _config(max_retries=1),
        ledger=CostLedger(1.0),
        transport=httpx.MockTransport(handler),
        sleep=lambda _: None,
    )
    with pytest.raises(SchemaValidationError):
        client.chat_json(
            system_prompt="system",
            user_prompt="user",
            response_model=TeacherLabelResponse,
        )
    assert calls == 2


def test_no_max_cost_is_dry_run_and_does_not_call_network() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return _response(_valid_label())

    client = OpenAICompatibleClient(
        _config(),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(DryRunError) as exc_info:
        client.chat_json(
            system_prompt="system",
            user_prompt="user",
            response_model=TeacherLabelResponse,
        )
    assert not called
    assert exc_info.value.plan["dry_run"] is True
    assert exc_info.value.plan["estimated_max_cost_usd"] >= 0


def test_cost_cap_is_enforced_before_network() -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return _response(_valid_label())

    client = OpenAICompatibleClient(
        _config(input_usd_per_million=1000, output_usd_per_million=1000),
        ledger=CostLedger(0.000001),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(CostLimitExceeded):
        client.chat_json(
            system_prompt="system",
            user_prompt="a" * 100,
            response_model=TeacherLabelResponse,
        )
    assert not called


def test_runtime_config_rejects_missing_key() -> None:
    config = _config(api_key="")
    with pytest.raises(ConfigurationError):
        config.validate_runtime()


def test_cache_hit_still_requires_explicit_cost_cap(tmp_path: Path) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(_valid_label())

    cache_dir = tmp_path / "cache"
    with OpenAICompatibleClient(
        _config(),
        ledger=CostLedger(1.0),
        cache_dir=cache_dir,
        transport=httpx.MockTransport(handler),
    ) as client:
        client.chat_json(
            system_prompt="system",
            user_prompt="user",
            response_model=TeacherLabelResponse,
            cache_scope="test",
        )

    dry_run_client = OpenAICompatibleClient(
        _config(),
        cache_dir=cache_dir,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(DryRunError):
        dry_run_client.chat_json(
            system_prompt="system",
            user_prompt="user",
            response_model=TeacherLabelResponse,
            cache_scope="test",
        )
    assert calls == 1
