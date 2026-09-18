#!/usr/bin/env python3
"""CLI for consensus labeling, malicious augmentation, and hard negatives."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmdrisk.schema import SessionRecord, SourceInfo  # noqa: E402
from cmdrisk.teacher import (  # noqa: E402
    CostLedger,
    DryRunError,
    HardNegativeGenerator,
    InsufficientVariantsError,
    MaliciousAugmenter,
    ModelConfig,
    OpenAICompatibleClient,
    PromptedModel,
    TeacherError,
)
from cmdrisk.teacher.labeling import ConsensusLabeler  # noqa: E402


def _read_jsonl(path: Path, model: type[SessionRecord]) -> list[SessionRecord]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return records


def _write_jsonl(path: Path, records: Iterable[SessionRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json() + "\n")


def _model_clients(
    *,
    max_cost_usd: float | None,
    cache_dir: Path,
) -> tuple[dict[str, PromptedModel], CostLedger | None]:
    ledger = CostLedger(max_cost_usd) if max_cost_usd is not None else None
    models: dict[str, PromptedModel] = {}
    for role in ("A", "B", "C"):
        client = OpenAICompatibleClient(
            ModelConfig.from_env(role),
            ledger=ledger,
            cache_dir=cache_dir,
        )
        models[role] = PromptedModel(client)
    return models, ledger


def _print_dry_run(error: DryRunError) -> None:
    print(json.dumps(error.plan, sort_keys=True), file=sys.stderr)


def _label(args: argparse.Namespace) -> int:
    records = _read_jsonl(args.input, SessionRecord)
    models, ledger = _model_clients(
        max_cost_usd=args.max_cost_usd,
        cache_dir=args.cache_dir,
    )
    labeler = ConsensusLabeler(models)
    accepted: list[SessionRecord] = []
    isolated: list[SessionRecord] = []
    for record in records:
        try:
            decision = labeler.label(record)
        except DryRunError as exc:
            _print_dry_run(exc)
            return 0
        updated = record.model_copy(update={"label": decision.label})
        (accepted if decision.accepted_for_training else isolated).append(updated)
    _write_jsonl(args.accepted_out, accepted)
    _write_jsonl(args.isolated_out, isolated)
    if ledger is not None:
        args.ledger_out.write_text(
            json.dumps(ledger.summary(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return 0


def _augment(args: argparse.Namespace) -> int:
    parents = _read_jsonl(args.input, SessionRecord)
    models, ledger = _model_clients(
        max_cost_usd=args.max_cost_usd,
        cache_dir=args.cache_dir,
    )
    augmenter = MaliciousAugmenter(
        models["A"],
        (models["B"], models["C"]),
        max_rounds=args.max_rounds,
    )
    variants: list[SessionRecord] = []
    for parent in parents:
        try:
            result = augmenter.augment(parent, args.count)
        except DryRunError as exc:
            _print_dry_run(exc)
            return 0
        except InsufficientVariantsError as exc:
            print(f"{parent.session_id}: {exc}", file=sys.stderr)
            continue
        variants.extend(result.records)
    _write_jsonl(args.output, variants)
    if ledger is not None:
        args.ledger_out.write_text(
            json.dumps(ledger.summary(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return 0


def _hard_negative(args: argparse.Namespace) -> int:
    models, ledger = _model_clients(
        max_cost_usd=args.max_cost_usd,
        cache_dir=args.cache_dir,
    )
    seed = json.loads(args.seed)
    source = SourceInfo(
        kind="script_benign",
        name=args.source_name,
        license=args.license,
    )
    generator = HardNegativeGenerator(
        models["A"],
        (models["B"], models["C"]),
        max_rounds=args.max_rounds,
    )
    try:
        result = generator.generate(
            seed,
            source=source,
            family_id=args.family_id,
            count=args.count,
        )
    except DryRunError as exc:
        _print_dry_run(exc)
        return 0
    _write_jsonl(args.output, result.records)
    if ledger is not None:
        args.ledger_out.write_text(
            json.dumps(ledger.summary(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return 0


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/teacher"))
    parser.add_argument("--ledger-out", type=Path, default=Path("teacher-ledger.json"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    label = subparsers.add_parser("label", help="A/B/C consensus-label session JSONL")
    label.add_argument("--input", type=Path, required=True)
    label.add_argument("--accepted-out", type=Path, required=True)
    label.add_argument("--isolated-out", type=Path, required=True)
    _common_arguments(label)
    label.set_defaults(func=_label)

    augment = subparsers.add_parser("augment", help="generate validated malicious variants")
    augment.add_argument("--input", type=Path, required=True)
    augment.add_argument("--output", type=Path, required=True)
    augment.add_argument("--count", type=int, default=10, choices=range(8, 13))
    augment.add_argument("--max-rounds", type=int, default=3)
    _common_arguments(augment)
    augment.set_defaults(func=_augment)

    hard_negative = subparsers.add_parser(
        "hard-negative",
        help="generate twice-validated benign attack lookalikes",
    )
    hard_negative.add_argument("--seed", required=True, help="JSON seed/context object")
    hard_negative.add_argument("--output", type=Path, required=True)
    hard_negative.add_argument("--source-name", required=True)
    hard_negative.add_argument("--license", required=True)
    hard_negative.add_argument("--family-id", required=True)
    hard_negative.add_argument("--count", type=int, default=8)
    hard_negative.add_argument("--max-rounds", type=int, default=3)
    _common_arguments(hard_negative)
    hard_negative.set_defaults(func=_hard_negative)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (TeacherError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
