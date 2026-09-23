"""Versioned prompt loading and typed wrappers around the JSON chat client."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from string import Template
from typing import Any

from cmdrisk.schema import SessionRecord
from cmdrisk.teacher.client import JsonChatClient
from cmdrisk.teacher.responses import (
    HardNegativeBatchResponse,
    HardNegativeValidationResponse,
    MaliciousValidationResponse,
    ProxyReviewResponse,
    RewriteBatchResponse,
    TeacherLabelResponse,
)

PROMPT_ROOT = Path(__file__).resolve().parents[3] / "configs" / "prompts"


class PromptError(ValueError):
    """Raised when a prompt file is missing or malformed."""


class PromptStore:
    def __init__(self, root: Path = PROMPT_ROOT) -> None:
        self.root = root

    def load(self, name: str) -> dict[str, str]:
        path = self.root / f"{name}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PromptError(f"prompt not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise PromptError(f"invalid prompt JSON: {path}") from exc
        if not isinstance(payload, dict):
            raise PromptError(f"prompt must be a JSON object: {path}")
        for key in ("name", "version", "system", "user_template"):
            if not isinstance(payload.get(key), str) or not payload[key].strip():
                raise PromptError(f"prompt {name!r} requires non-empty {key}")
        if payload["name"] != name:
            raise PromptError(f"prompt name mismatch: expected {name!r}, got {payload['name']!r}")
        return payload

    def fingerprint(self, name: str) -> str:
        """Return a stable SHA-256 fingerprint of the complete prompt config."""

        payload = self.load(name)
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def prompt_hash(self, name: str) -> str:
        """Readable alias for callers constructing audit provenance."""

        return self.fingerprint(name)

    def render(self, name: str, **values: Any) -> tuple[str, str]:
        prompt = self.load(name)
        serializable = {
            key: value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            for key, value in values.items()
        }
        try:
            user_prompt = Template(prompt["user_template"]).substitute(serializable)
        except KeyError as exc:
            raise PromptError(f"prompt {name!r} is missing value {exc.args[0]!r}") from exc
        return prompt["system"], user_prompt


class PromptedModel:
    """Bind semantic operations to versioned prompts."""

    def __init__(
        self,
        client: JsonChatClient,
        *,
        label_prompt: str = "teacher_label_v1",
        rewrite_prompt: str = "teacher_augment_v1",
        malicious_validator_prompt: str = "teacher_validator_malicious_v1",
        hard_negative_prompt: str = "teacher_hard_negative_v1",
        hard_negative_validator_prompt: str = "teacher_validator_hard_negative_v1",
        review_prompt: str = "proxy_reviewer_v1",
        prompts: PromptStore | None = None,
    ) -> None:
        self.client = client
        self.prompts = prompts or PromptStore()
        self.label_prompt = label_prompt
        self.rewrite_prompt = rewrite_prompt
        self.malicious_validator_prompt = malicious_validator_prompt
        self.hard_negative_prompt = hard_negative_prompt
        self.hard_negative_validator_prompt = hard_negative_validator_prompt
        self.review_prompt = review_prompt

    def label(self, session: SessionRecord) -> TeacherLabelResponse:
        system, user = self.prompts.render(
            self.label_prompt,
            session=_session_payload(session),
        )
        return self.client.chat_json(
            system_prompt=system,
            user_prompt=user,
            response_model=TeacherLabelResponse,
            cache_scope=f"prompt:{self.label_prompt}:teacher-label",
        )

    def rewrite(self, session: SessionRecord, count: int) -> RewriteBatchResponse:
        system, user = self.prompts.render(
            self.rewrite_prompt,
            session=_session_payload(session),
            count=count,
        )
        return self.client.chat_json(
            system_prompt=system,
            user_prompt=user,
            response_model=RewriteBatchResponse,
            cache_scope=f"prompt:{self.rewrite_prompt}:augment:{count}",
        )

    def validate_malicious_variant(
        self,
        parent: SessionRecord,
        commands: list[str],
        first_evidence_index: int,
    ) -> MaliciousValidationResponse:
        system, user = self.prompts.render(
            self.malicious_validator_prompt,
            parent=_session_payload(parent),
            variant={"commands": commands, "first_evidence_index": first_evidence_index},
        )
        return self.client.chat_json(
            system_prompt=system,
            user_prompt=user,
            response_model=MaliciousValidationResponse,
            cache_scope=f"prompt:{self.malicious_validator_prompt}:validate-malicious",
        )

    def hard_negative(self, seed: dict[str, Any], count: int) -> HardNegativeBatchResponse:
        system, user = self.prompts.render(
            self.hard_negative_prompt,
            seed=seed,
            count=count,
        )
        return self.client.chat_json(
            system_prompt=system,
            user_prompt=user,
            response_model=HardNegativeBatchResponse,
            cache_scope=f"prompt:{self.hard_negative_prompt}:hard-negative:{count}",
        )

    def validate_hard_negative(self, commands: list[str]) -> HardNegativeValidationResponse:
        system, user = self.prompts.render(
            self.hard_negative_validator_prompt,
            candidate={"commands": commands},
        )
        return self.client.chat_json(
            system_prompt=system,
            user_prompt=user,
            response_model=HardNegativeValidationResponse,
            cache_scope=f"prompt:{self.hard_negative_validator_prompt}:validate-hard-negative",
        )

    def review(self, review_context: dict[str, Any]) -> ProxyReviewResponse:
        safe_context = reviewer_payload(review_context)
        system, user = self.prompts.render(
            self.review_prompt,
            case=safe_context,
        )
        return self.client.chat_json(
            system_prompt=system,
            user_prompt=user,
            response_model=ProxyReviewResponse,
            cache_scope=f"prompt:{self.review_prompt}:proxy-review",
        )


def _session_payload(session: SessionRecord) -> dict[str, Any]:
    return {
        "session_id": session.session_id,
        "commands": [
            {"index": index, "cmd": event.cmd, "dt": event.dt}
            for index, event in enumerate(session.commands)
        ],
    }


def reviewer_payload(case: dict[str, Any]) -> dict[str, Any]:
    """Return only fields a blind reviewer is allowed to see."""

    return {
        "session_id": case["session_id"],
        "commands": case["commands"],
    }
