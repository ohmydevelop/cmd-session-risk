from __future__ import annotations

from pathlib import Path

import pytest

from cmdrisk.tokenization import (
    CMD_END_TOKEN,
    CMD_START_TOKEN,
    DEFAULT_MAX_LENGTH,
    DT_0S,
    DT_GT_10M,
    DT_LE_1M,
    DT_LE_1S,
    DT_LE_10M,
    DT_LE_10S,
    DT_UNKNOWN,
    SESSION_END_TOKEN,
    SESSION_TOKEN,
    TRUNCATED_TOKEN,
    CommandTokenizer,
    dt_bucket,
)


class FakeTokenizer:
    """Whitespace tokenizer with Hugging Face-like special-token methods."""

    unk_token_id = 0

    def __init__(self) -> None:
        self.tokens = {"[UNK]": 0, "whoami": 1, "uname": 2}
        self.specials: list[str] = []

    def __len__(self) -> int:
        return max(self.tokens.values(), default=0) + 1

    def add_special_tokens(self, values: dict[str, list[str]]) -> None:
        for token in values["additional_special_tokens"]:
            if token not in self.tokens:
                self.tokens[token] = len(self.tokens)
                self.specials.append(token)

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.tokens.get(token, self.unk_token_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = []
        for token in text.split():
            if token not in self.tokens:
                self.tokens[token] = len(self.tokens)
            ids.append(self.tokens[token])
        return ids

    def save_pretrained(self, directory: str | Path) -> None:
        Path(directory, "fake_tokenizer.txt").write_text("saved", encoding="utf-8")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, DT_UNKNOWN),
        (0, DT_0S),
        (1, DT_LE_1S),
        (1.1, DT_LE_10S),
        (10, DT_LE_10S),
        (60, DT_LE_1M),
        (600, DT_LE_10M),
        (600.1, DT_GT_10M),
    ],
)
def test_dt_buckets(value: float | None, expected: str) -> None:
    assert dt_bucket(value) == expected


def test_dt_rejects_negative_values() -> None:
    with pytest.raises(ValueError):
        dt_bucket(-0.1)


def test_command_stream_uses_markers_and_dt_bucket() -> None:
    tokenizer = CommandTokenizer(FakeTokenizer())
    encoded = tokenizer.encode(
        [{"cmd": "whoami", "dt": 0}, {"cmd": "uname", "dt": None}]
    )
    ids = list(encoded.input_ids)
    assert ids[0] == tokenizer.special_token_id(SESSION_TOKEN)
    assert ids[-1] == tokenizer.special_token_id(SESSION_END_TOKEN)
    assert tokenizer.special_token_id(CMD_START_TOKEN) in ids
    assert tokenizer.special_token_id(CMD_END_TOKEN) in ids
    assert tokenizer.special_token_id(DT_0S) in ids
    assert tokenizer.special_token_id(DT_UNKNOWN) in ids


def test_long_command_keeps_512_tokens_and_both_ends() -> None:
    tokenizer = CommandTokenizer(FakeTokenizer(), command_max_tokens=512)
    raw_tokens = [f"token{i}" for i in range(600)]
    encoded = tokenizer.encode([" ".join(raw_tokens)])
    ids = list(encoded.input_ids)
    start = ids.index(tokenizer.special_token_id(CMD_START_TOKEN))
    end = ids.index(tokenizer.special_token_id(CMD_END_TOKEN))
    command_ids = ids[start + 1 : end]
    assert len(command_ids) == 512
    assert command_ids[0] == tokenizer._encode_text("token0")[0]
    assert command_ids[-1] == tokenizer._encode_text("token599")[0]


def test_long_session_keeps_head_tail_and_truncated_marker() -> None:
    tokenizer = CommandTokenizer(FakeTokenizer())
    command = " ".join(f"x{i}" for i in range(120))
    commands = [command for _ in range(80)]
    stream = tokenizer._build_stream(commands)
    assert len(stream) > DEFAULT_MAX_LENGTH
    encoded = tokenizer.encode(commands)
    ids = list(encoded.input_ids)
    assert encoded.truncated is True
    assert len(ids) == DEFAULT_MAX_LENGTH
    assert ids[:1024] == stream[:1024]
    assert ids[1024] == tokenizer.special_token_id(TRUNCATED_TOKEN)
    assert ids[1025:] == stream[-(DEFAULT_MAX_LENGTH - 1025) :]
    assert ids[-1] == tokenizer.special_token_id(SESSION_END_TOKEN)


def test_short_session_is_not_truncated() -> None:
    tokenizer = CommandTokenizer(FakeTokenizer())
    encoded = tokenizer.encode(["whoami", "uname"])
    assert encoded.truncated is False
    assert all(bit == 1 for bit in encoded.attention_mask)


def test_save_pretrained_writes_tokenization_metadata(tmp_path: Path) -> None:
    tokenizer = CommandTokenizer(FakeTokenizer(), max_length=1024, command_max_tokens=100)
    directory = tokenizer.save_pretrained(tmp_path / "tokenizer")
    assert (directory / "command_tokenization.json").is_file()
    assert (directory / "fake_tokenizer.txt").read_text(encoding="utf-8") == "saved"
