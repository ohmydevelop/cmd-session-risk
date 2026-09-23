"""Tokenization for command-session prefixes.

The runtime model consumes a marker-delimited command transcript rather than a
plain concatenation of command strings.  This module deliberately keeps the
tokenizer dependency behind :meth:`CommandTokenizer.from_pretrained` so the
marker construction and truncation rules can be unit tested without Torch.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SESSION_TOKEN = "<SESSION>"
SESSION_END_TOKEN = "</SESSION>"
CMD_START_TOKEN = "<CMD_START>"
CMD_END_TOKEN = "<CMD_END>"
TRUNCATED_TOKEN = "<TRUNCATED>"

DT_0S = "<DT_0S>"
DT_LE_1S = "<DT_LE_1S>"
DT_LE_10S = "<DT_LE_10S>"
DT_LE_1M = "<DT_LE_1M>"
DT_LE_10M = "<DT_LE_10M>"
DT_GT_10M = "<DT_GT_10M>"
DT_UNKNOWN = "<DT_UNKNOWN>"

DT_BUCKET_TOKENS: tuple[str, ...] = (
    DT_0S,
    DT_LE_1S,
    DT_LE_10S,
    DT_LE_1M,
    DT_LE_10M,
    DT_GT_10M,
    DT_UNKNOWN,
)

SPECIAL_TOKENS: tuple[str, ...] = (
    SESSION_TOKEN,
    SESSION_END_TOKEN,
    CMD_START_TOKEN,
    CMD_END_TOKEN,
    TRUNCATED_TOKEN,
    *DT_BUCKET_TOKENS,
)

DEFAULT_MAX_LENGTH = 8192
DEFAULT_COMMAND_MAX_TOKENS = 512
DEFAULT_SESSION_HEAD_TOKENS = 1024
DEFAULT_SESSION_TAIL_TOKENS = 7168


def dt_bucket(dt: float | None) -> str:
    """Map a delta time to one of the fixed marker buckets."""

    if dt is None:
        return DT_UNKNOWN
    if dt < 0:
        raise ValueError("dt must be non-negative or None")
    if dt == 0:
        return DT_0S
    if dt <= 1:
        return DT_LE_1S
    if dt <= 10:
        return DT_LE_10S
    if dt <= 60:
        return DT_LE_1M
    if dt <= 600:
        return DT_LE_10M
    return DT_GT_10M


@dataclass(frozen=True)
class TokenizedCommandSequence:
    """Token IDs produced from one command prefix."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    truncated: bool

    def as_dict(self) -> dict[str, list[int] | bool]:
        return {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "truncated": self.truncated,
        }


class CommandTokenizer:
    """Wrap a Hugging Face tokenizer with the command-session marker schema.

    A command's content is capped at ``command_max_tokens`` by keeping its first
    and last halves.  Session sequences are capped at ``max_length`` by keeping
    the first ``session_head_tokens`` tokens and, after inserting a single
    ``<TRUNCATED>`` marker, the final tokens available in the budget.  Keeping
    the tail is intentional: later commands carry the attack progression.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int = DEFAULT_MAX_LENGTH,
        command_max_tokens: int = DEFAULT_COMMAND_MAX_TOKENS,
        session_head_tokens: int | None = None,
        session_tail_tokens: int | None = None,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if command_max_tokens <= 0:
            raise ValueError("command_max_tokens must be positive")
        if session_head_tokens is None:
            session_head_tokens = min(DEFAULT_SESSION_HEAD_TOKENS, max_length - 1)
        if session_tail_tokens is None:
            session_tail_tokens = min(
                DEFAULT_SESSION_TAIL_TOKENS,
                max_length - int(session_head_tokens) - 1,
            )
        if session_head_tokens < 0 or session_tail_tokens < 0:
            raise ValueError("session truncation budgets must be non-negative")
        if session_head_tokens + 1 > max_length:
            raise ValueError("session_head_tokens leaves no room for <TRUNCATED>")
        if session_tail_tokens + 1 > max_length:
            raise ValueError("session_tail_tokens leaves no room for <TRUNCATED>")

        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.command_max_tokens = int(command_max_tokens)
        self.session_head_tokens = int(session_head_tokens)
        self.session_tail_tokens = int(session_tail_tokens)
        self._special_ids: dict[str, int] = {}
        self._prepare_special_tokens()

    @property
    def vocab_size(self) -> int:
        value = getattr(self.tokenizer, "vocab_size", None)
        if value is not None:
            return int(value)
        try:
            return len(self.tokenizer)
        except TypeError:
            return max(self._special_ids.values(), default=0) + 1

    @property
    def pad_token_id(self) -> int:
        value = getattr(self.tokenizer, "pad_token_id", None)
        if value is not None:
            return int(value)
        return self._special_ids[SESSION_END_TOKEN]

    def _prepare_special_tokens(self) -> None:
        add = getattr(self.tokenizer, "add_special_tokens", None)
        if callable(add):
            add({"additional_special_tokens": list(SPECIAL_TOKENS)})

        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        known_ids: set[int] = set()
        for token in SPECIAL_TOKENS:
            token_id = None
            if callable(convert):
                token_id = convert(token)
            if token_id is None or token_id == unk_id:
                token_id = None
            if token_id is None:
                token_id = self._next_special_id(known_ids)
            token_id = int(token_id)
            self._special_ids[token] = token_id
            known_ids.add(token_id)

    def _next_special_id(self, known_ids: set[int]) -> int:
        candidate = max(self.vocab_size, max(known_ids, default=-1) + 1)
        while candidate in known_ids:
            candidate += 1
        return candidate

    def special_token_id(self, token: str) -> int:
        return self._special_ids[token]

    def _encode_text(self, text: str) -> list[int]:
        encode = getattr(self.tokenizer, "encode", None)
        if callable(encode):
            values = encode(text, add_special_tokens=False)
            return [int(value) for value in values]

        called = self.tokenizer(text, add_special_tokens=False)
        if isinstance(called, dict):
            values = called["input_ids"]
        else:
            values = called.input_ids
        if values and isinstance(values[0], list):
            values = values[0]
        return [int(value) for value in values]

    def _truncate_command(self, token_ids: list[int]) -> list[int]:
        if len(token_ids) <= self.command_max_tokens:
            return token_ids
        first = self.command_max_tokens // 2
        last = self.command_max_tokens - first
        return token_ids[:first] + token_ids[-last:]

    @staticmethod
    def _command_and_dt(item: Any) -> tuple[str, float | None]:
        if isinstance(item, str):
            return item, None
        if isinstance(item, dict):
            cmd = item.get("cmd")
            dt = item.get("dt")
        else:
            cmd = getattr(item, "cmd", None)
            dt = getattr(item, "dt", None)
        if not isinstance(cmd, str) or not cmd.strip():
            raise ValueError("each command must be a non-empty string")
        return cmd, dt

    def _build_stream(self, commands: Sequence[Any]) -> list[int]:
        if not commands:
            raise ValueError("at least one command is required")
        session_id = self.special_token_id(SESSION_TOKEN)
        session_end_id = self.special_token_id(SESSION_END_TOKEN)
        cmd_start_id = self.special_token_id(CMD_START_TOKEN)
        cmd_end_id = self.special_token_id(CMD_END_TOKEN)

        stream = [session_id]
        for item in commands:
            command, dt = self._command_and_dt(item)
            command_ids = self._truncate_command(self._encode_text(command))
            stream.append(cmd_start_id)
            stream.extend(command_ids)
            stream.extend((cmd_end_id, self.special_token_id(dt_bucket(dt))))
        stream.append(session_end_id)
        return stream

    def encode(self, commands: Sequence[Any]) -> TokenizedCommandSequence:
        stream = self._build_stream(commands)
        if len(stream) <= self.max_length:
            return TokenizedCommandSequence(tuple(stream), (1,) * len(stream), False)

        # The marker itself consumes one token.  Preserve the requested head
        # first and use the remaining budget for the most recent tail.
        head = stream[: self.session_head_tokens]
        tail_budget = self.max_length - len(head) - 1
        tail = stream[-tail_budget:] if tail_budget else []
        truncated = [*head, self.special_token_id(TRUNCATED_TOKEN), *tail]
        return TokenizedCommandSequence(tuple(truncated), (1,) * len(truncated), True)

    def save_pretrained(self, save_directory: str | Path) -> Path:
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(directory)
        metadata = {
            "version": 1,
            "max_length": self.max_length,
            "command_max_tokens": self.command_max_tokens,
            "session_head_tokens": self.session_head_tokens,
            "session_tail_tokens": self.session_tail_tokens,
            "special_tokens": list(SPECIAL_TOKENS),
        }
        (directory / "command_tokenization.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return directory

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs: Any,
    ) -> CommandTokenizer:
        from transformers import AutoTokenizer

        source = Path(pretrained_model_name_or_path)
        metadata_path = source / "command_tokenization.json"
        metadata: dict[str, Any] = {}
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        requested_max_length = kwargs.pop("max_length", None)
        stored_max_length = metadata.get("max_length")
        max_length = int(
            requested_max_length
            if requested_max_length is not None
            else stored_max_length
            if stored_max_length is not None
            else DEFAULT_MAX_LENGTH
        )
        command_max_tokens = int(
            kwargs.pop(
                "command_max_tokens",
                metadata.get("command_max_tokens", DEFAULT_COMMAND_MAX_TOKENS),
            )
        )
        # The stored truncation budgets are derived from the stored max_length.
        # When a caller asks for a different max_length, recompute them unless
        # the budgets were supplied explicitly, otherwise the head budget can
        # exceed the new sequence budget.
        stored_budgets_are_stale = (
            requested_max_length is not None
            and stored_max_length is not None
            and int(requested_max_length) != int(stored_max_length)
        )
        if stored_budgets_are_stale:
            session_head_tokens = kwargs.pop("session_head_tokens", None)
            session_tail_tokens = kwargs.pop("session_tail_tokens", None)
        else:
            session_head_tokens = kwargs.pop(
                "session_head_tokens", metadata.get("session_head_tokens")
            )
            session_tail_tokens = kwargs.pop(
                "session_tail_tokens", metadata.get("session_tail_tokens")
            )
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(
            tokenizer,
            max_length=max_length,
            command_max_tokens=command_max_tokens,
            session_head_tokens=(
                int(session_head_tokens) if session_head_tokens is not None else None
            ),
            session_tail_tokens=(
                int(session_tail_tokens) if session_tail_tokens is not None else None
            ),
        )


def tokenize_command_prefix(
    tokenizer: CommandTokenizer,
    commands: Sequence[Any],
) -> TokenizedCommandSequence:
    """Convenience function used by datasets and export smoke checks."""

    return tokenizer.encode(commands)
