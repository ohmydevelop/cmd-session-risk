"""Torch datasets and dynamic-padding collators for prefix examples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

import torch
from torch.utils.data import Dataset

from cmdrisk.tokenization import CommandTokenizer
from cmdrisk.training.data import PrefixExample


class CommandRiskDataset(Dataset[dict[str, Any]]):
    def __init__(self, examples: Sequence[PrefixExample], tokenizer: CommandTokenizer) -> None:
        self.examples = list(examples)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        commands = [
            {"cmd": command, "dt": dt}
            for command, dt in zip(example.commands, example.dts, strict=True)
        ]
        encoded = self.tokenizer.encode(commands)
        return {
            "input_ids": list(encoded.input_ids),
            "attention_mask": list(encoded.attention_mask),
            "labels": {
                "malicious": float(example.malicious),
                "tactics": list(example.tactics),
                "techniques": list(example.techniques),
            },
            "metadata": asdict(example),
        }


class CommandRiskCollator:
    """Pad token IDs and stack scalar/multi-label targets."""

    def __init__(self, tokenizer: CommandTokenizer) -> None:
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(len(item["input_ids"]) for item in features)
        input_ids = []
        attention_mask = []
        malicious = []
        tactics = []
        techniques = []
        for item in features:
            padding = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(item["attention_mask"] + [0] * padding)
            malicious.append(item["labels"]["malicious"])
            tactics.append(item["labels"]["tactics"])
            techniques.append(item["labels"]["techniques"])

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": {
                "malicious": torch.tensor(malicious, dtype=torch.float32),
                "tactics": torch.tensor(tactics, dtype=torch.float32),
                "techniques": torch.tensor(techniques, dtype=torch.float32),
            },
        }
