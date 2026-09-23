"""OpenAI-compatible JSON chat client with cache, retry, and cost accounting."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, Field, ValidationError

ResponseT = TypeVar("ResponseT", bound=BaseModel)


class TeacherError(RuntimeError):
    """Base error for Teacher-side failures."""


class ConfigurationError(TeacherError):
    """Raised when a model configuration is incomplete."""


class DryRunError(TeacherError):
    """Raised when a network-backed request is attempted without a cost cap."""

    def __init__(self, plan: dict[str, Any]) -> None:
        super().__init__("max_cost_usd is required for network requests; request was dry-run")
        self.plan = plan


class CostLimitExceeded(TeacherError):
    """Raised when a request would exceed its explicit cost budget."""

    def __init__(self, *, limit_usd: float, spent_usd: float, estimated_usd: float) -> None:
        super().__init__(
            f"cost limit exceeded: spent=${spent_usd:.6f}, "
            f"estimated=${estimated_usd:.6f}, limit=${limit_usd:.6f}"
        )
        self.limit_usd = limit_usd
        self.spent_usd = spent_usd
        self.estimated_usd = estimated_usd


class ChatAPIError(TeacherError):
    """Raised for a non-retryable or exhausted API error."""


class SchemaValidationError(TeacherError):
    """Raised when an API response does not satisfy its response model."""


@dataclass(frozen=True)
class LedgerEntry:
    role: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    usage_estimated: bool
    cache_scope: str
    revision: str = ""


class CostLedger:
    """Thread-safe cumulative token and USD accounting."""

    def __init__(self, max_cost_usd: float) -> None:
        if max_cost_usd < 0:
            raise ValueError("max_cost_usd must be non-negative")
        self.max_cost_usd = float(max_cost_usd)
        self._spent_usd = 0.0
        self._reserved_usd = 0.0
        self._entries: list[LedgerEntry] = []
        self._lock = threading.Lock()

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent_usd

    @property
    def reserved_usd(self) -> float:
        with self._lock:
            return self._reserved_usd

    @property
    def remaining_usd(self) -> float:
        return self.max_cost_usd - self.spent_usd - self.reserved_usd

    @property
    def entries(self) -> tuple[LedgerEntry, ...]:
        with self._lock:
            return tuple(self._entries)

    def reserve(self, estimated_usd: float) -> str:
        if estimated_usd < 0:
            raise ValueError("estimated_usd must be non-negative")
        with self._lock:
            if self._spent_usd + self._reserved_usd + estimated_usd > self.max_cost_usd + 1e-12:
                raise CostLimitExceeded(
                    limit_usd=self.max_cost_usd,
                    spent_usd=self._spent_usd + self._reserved_usd,
                    estimated_usd=estimated_usd,
                )
            self._reserved_usd += estimated_usd
        return hashlib.sha256(
            f"{time.time_ns()}:{estimated_usd}:{len(self._entries)}".encode()
        ).hexdigest()

    def release(self, reservation_id: str, estimated_usd: float) -> None:
        del reservation_id
        with self._lock:
            self._reserved_usd = max(0.0, self._reserved_usd - estimated_usd)

    def settle(
        self,
        *,
        reservation_id: str,
        estimated_usd: float,
        actual_usd: float,
        entry: LedgerEntry,
    ) -> None:
        del reservation_id
        with self._lock:
            self._reserved_usd = max(0.0, self._reserved_usd - estimated_usd)
            if self._spent_usd + actual_usd > self.max_cost_usd + 1e-12:
                self._reserved_usd += estimated_usd
                raise CostLimitExceeded(
                    limit_usd=self.max_cost_usd,
                    spent_usd=self._spent_usd,
                    estimated_usd=actual_usd,
                )
            self._spent_usd += actual_usd
            self._entries.append(entry)

    def summary(self) -> dict[str, Any]:
        entries = self.entries
        return {
            "max_cost_usd": self.max_cost_usd,
            "spent_usd": self.spent_usd,
            "remaining_usd": self.max_cost_usd - self.spent_usd,
            "calls": len(entries),
            "prompt_tokens": sum(entry.prompt_tokens for entry in entries),
            "completion_tokens": sum(entry.completion_tokens for entry in entries),
            "total_tokens": sum(entry.total_tokens for entry in entries),
            "entries": [asdict(entry) for entry in entries],
        }


class ModelConfig(BaseModel):
    """A/B/C model settings resolved from the process environment at runtime."""

    role: str = Field(min_length=1)
    model: str = Field(min_length=1)
    revision: str = ""
    base_url: str = Field(min_length=1)
    api_key: str = Field(default="", exclude=True, repr=False)
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_output_tokens: int = Field(default=4096, gt=0)
    input_usd_per_million: float = Field(default=0.0, ge=0)
    output_usd_per_million: float = Field(default=0.0, ge=0)
    temperature: float = Field(default=0.0, ge=0, le=2)

    @classmethod
    def from_env(
        cls,
        role: str,
        *,
        environ: Mapping[str, str] | None = None,
        namespace: str = "CMDRISK_TEACHER",
    ) -> ModelConfig:
        env = os.environ if environ is None else environ
        role_key = role.upper().replace("-", "_")
        prefix = f"{namespace.upper()}_{role_key}"
        api_key = env.get(f"{prefix}_API_KEY", env.get("OPENAI_API_KEY", ""))
        return cls(
            role=role,
            base_url=env.get(
                f"{prefix}_BASE_URL",
                env.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            ),
            api_key=api_key,
            model=env.get(f"{prefix}_MODEL", role_key.lower()),
            revision=env.get(f"{prefix}_REVISION", "").strip(),
            timeout_s=float(env.get(f"{prefix}_TIMEOUT_S", "60")),
            max_retries=int(env.get(f"{prefix}_MAX_RETRIES", "3")),
            max_output_tokens=int(env.get(f"{prefix}_MAX_OUTPUT_TOKENS", "4096")),
            input_usd_per_million=float(env.get(f"{prefix}_INPUT_USD_PER_MTOK", "0")),
            output_usd_per_million=float(env.get(f"{prefix}_OUTPUT_USD_PER_MTOK", "0")),
            temperature=float(env.get(f"{prefix}_TEMPERATURE", "0")),
        )

    @property
    def identity(self) -> tuple[str, str, str]:
        """Return the endpoint/model/revision identity used by consensus gates."""

        return (self.base_url.rstrip("/"), self.model.strip(), self.revision.strip())

    def validate_runtime(self) -> None:
        if not self.api_key:
            raise ConfigurationError(f"missing runtime API key for role {self.role!r}")
        if self.model == self.role.lower():
            raise ConfigurationError(f"missing runtime model name for role {self.role!r}")


def validate_distinct_reviewer_configs(
    configs: Mapping[str, ModelConfig] | list[ModelConfig] | tuple[ModelConfig, ...],
    *,
    roles: tuple[str, ...] = ("A", "B", "C"),
    require_revision: bool = True,
) -> None:
    """Reject duplicate endpoint/model/revision triples before network calls.

    The default ``require_revision`` is intentionally strict: a proxy audit is
    not reproducible when a provider silently moves a model alias.  Tests and
    dry-run planning can pass ``False`` when no model call will be made.
    """

    if isinstance(configs, Mapping):
        missing = set(roles) - set(configs)
        if missing:
            raise ConfigurationError(f"missing reviewer configurations: {sorted(missing)}")
        ordered = [configs[role] for role in roles]
    else:
        ordered = list(configs)
        if len(ordered) != len(roles):
            raise ConfigurationError(
                f"expected {len(roles)} reviewer configurations, got {len(ordered)}"
            )
    for role, config in zip(roles, ordered, strict=True):
        if not isinstance(config, ModelConfig):
            raise ConfigurationError(f"reviewer {role!r} is not a ModelConfig")
        if not config.base_url.strip():
            raise ConfigurationError(f"reviewer {role!r} has an empty endpoint")
        if not config.model.strip():
            raise ConfigurationError(f"reviewer {role!r} has an empty model")
        if require_revision and not config.revision.strip():
            raise ConfigurationError(f"reviewer {role!r} requires a non-empty revision")
    seen: dict[tuple[str, str, str], str] = {}
    for role, config in zip(roles, ordered, strict=True):
        identity = config.identity
        previous = seen.get(identity)
        if previous is not None:
            raise ConfigurationError(
                "reviewer configurations must be distinct by endpoint/model/revision: "
                f"{previous!r} and {role!r} both use {identity!r}"
            )
        seen[identity] = role


# Backwards/readable aliases used by orchestration scripts and downstream code.
def validate_distinct_model_configs(
    configs: Mapping[str, ModelConfig] | list[ModelConfig] | tuple[ModelConfig, ...],
    *,
    roles: tuple[str, ...] = ("A", "B", "C"),
    require_revision: bool = True,
) -> None:
    validate_distinct_reviewer_configs(
        configs,
        roles=roles,
        require_revision=require_revision,
    )


def model_config_identity(config: ModelConfig) -> tuple[str, str, str]:
    """Stable identity value shared by cache and reviewer provenance code."""

    return config.identity


class JsonChatClient(Protocol):
    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseT],
        cache_scope: str = "",
    ) -> ResponseT: ...


class OpenAICompatibleClient:
    """Call an OpenAI-compatible chat-completions endpoint and validate JSON."""

    def __init__(
        self,
        config: ModelConfig,
        *,
        ledger: CostLedger | None = None,
        cache_dir: Path | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.cache_dir = cache_dir
        self.dry_run = ledger is None
        self._sleep = sleep
        self._client = httpx.Client(timeout=config.timeout_s, transport=transport)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAICompatibleClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseT],
        cache_scope: str = "",
    ) -> ResponseT:
        cache_key = self._cache_key(system_prompt, user_prompt, response_model, cache_scope)
        prompt_tokens = _estimate_tokens(system_prompt) + _estimate_tokens(user_prompt)
        estimated_cost = self._cost(prompt_tokens, self.config.max_output_tokens)
        plan = self._request_plan(
            cache_key=cache_key,
            prompt_tokens=prompt_tokens,
            estimated_cost=estimated_cost,
            cache_scope=cache_scope,
        )
        if self.dry_run:
            raise DryRunError(plan)

        cached = self._read_cache(cache_key, response_model)
        if cached is not None:
            return cached

        self.config.validate_runtime()
        assert self.ledger is not None
        reservation = self.ledger.reserve(estimated_cost)
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                result, prompt_usage, completion_usage, usage_estimated = self._post_json(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_model=response_model,
                )
                actual_cost = self._cost(prompt_usage, completion_usage)
                self.ledger.settle(
                    reservation_id=reservation,
                    estimated_usd=estimated_cost,
                    actual_usd=actual_cost,
                    entry=LedgerEntry(
                        role=self.config.role,
                        model=self.config.model,
                        prompt_tokens=prompt_usage,
                        completion_tokens=completion_usage,
                        total_tokens=prompt_usage + completion_usage,
                        cost_usd=actual_cost,
                        usage_estimated=usage_estimated,
                        cache_scope=cache_scope,
                        revision=self.config.revision,
                    ),
                )
                self._write_cache(cache_key, result)
                return result
            except (httpx.HTTPError, ChatAPIError, SchemaValidationError, ValueError) as exc:
                last_error = exc
                if not self._is_retryable(exc) or attempt >= self.config.max_retries:
                    break
                self._sleep(min(2**attempt, 8))
        self.ledger.release(reservation, estimated_cost)
        if isinstance(last_error, SchemaValidationError):
            raise last_error
        if isinstance(last_error, ChatAPIError):
            raise last_error
        raise ChatAPIError(f"chat request failed after retries: {last_error}") from last_error

    def _post_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: type[ResponseT],
    ) -> tuple[ResponseT, int, int, bool]:
        response = self._client.post(
            self._endpoint,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
            json={
                "model": self.config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_output_tokens,
                "response_format": {"type": "json_object"},
            },
        )
        if response.status_code == 429 or response.status_code >= 500:
            raise ChatAPIError(
                f"retryable HTTP {response.status_code}: {response.text[:500]}"
            )
        if response.status_code >= 400:
            raise ChatAPIError(f"HTTP {response.status_code}: {response.text[:500]}")
        try:
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise SchemaValidationError("invalid chat-completions response envelope") from exc
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str):
            raise SchemaValidationError("chat response content must be a string")
        try:
            result = response_model.model_validate_json(_strip_json_fence(content))
        except ValidationError as exc:
            raise SchemaValidationError(
                f"response failed {response_model.__name__} validation"
            ) from exc
        usage = payload.get("usage") or {}
        usage_estimated = not bool(usage)
        prompt_tokens = int(
            usage.get(
                "prompt_tokens",
                usage.get("input_tokens", _estimate_tokens(system_prompt + user_prompt)),
            )
        )
        completion_tokens = int(
            usage.get("completion_tokens", usage.get("output_tokens", _estimate_tokens(content)))
        )
        return result, max(prompt_tokens, 0), max(completion_tokens, 0), usage_estimated

    def _request_plan(
        self,
        *,
        cache_key: str,
        prompt_tokens: int,
        estimated_cost: float,
        cache_scope: str,
    ) -> dict[str, Any]:
        return {
            "role": self.config.role,
            "model": self.config.model,
            "revision": self.config.revision,
            "endpoint": self._endpoint,
            "cache_key": cache_key,
            "cache_scope": cache_scope,
            "estimated_prompt_tokens": prompt_tokens,
            "max_output_tokens": self.config.max_output_tokens,
            "estimated_max_cost_usd": estimated_cost,
            "dry_run": True,
        }

    @property
    def _endpoint(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def _cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.config.input_usd_per_million
            + completion_tokens * self.config.output_usd_per_million
        ) / 1_000_000

    def _cache_key(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: type[BaseModel],
        cache_scope: str,
    ) -> str:
        body = {
            "role": self.config.role,
            "model": self.config.model,
            "revision": self.config.revision,
            "base_url": self.config.base_url,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "schema": response_model.model_json_schema(),
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_output_tokens,
            "cache_scope": cache_scope,
        }
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _read_cache(self, key: str, response_model: type[ResponseT]) -> ResponseT | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return response_model.model_validate(payload["response"])
        except (FileNotFoundError, KeyError, ValueError, ValidationError):
            return None

    def _write_cache(self, key: str, result: BaseModel) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self.cache_dir / f"{key}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "role": self.config.role,
                    "model": self.config.model,
                    "revision": self.config.revision,
                    "response": result.model_dump(mode="json"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, SchemaValidationError):
            return True
        if isinstance(exc, ChatAPIError):
            text = str(exc)
            return text.startswith("retryable HTTP") or text.startswith("chat request failed")
        return isinstance(exc, httpx.HTTPError)


def _estimate_tokens(text: str) -> int:
    return max(1, (len(text) + 3) // 4)


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped
