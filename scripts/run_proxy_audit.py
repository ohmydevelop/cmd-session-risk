#!/usr/bin/env python3
"""Run a blind multi-reviewer proxy audit over JSONL cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cmdrisk.teacher import (  # noqa: E402
    AuditCase,
    ConfigurationError,
    CostLedger,
    DryRunError,
    ModelConfig,
    OpenAICompatibleClient,
    PromptedModel,
    ProxyAuditor,
    TeacherError,
)


def _read_cases(path: Path) -> list[AuditCase]:
    cases: list[AuditCase] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                cases.append(AuditCase.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        help="required for network calls; omitted means a no-network dry run",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/proxy-audit"))
    parser.add_argument("--unresolved-out", type=Path)
    parser.add_argument(
        "--tiebreak-role",
        action="append",
        default=[],
        metavar="ROLE",
        help=(
            "optional reviewer role consulted only after the primary R1-R3 panel "
            "cannot resolve a case; reads CMDRISK_REVIEWER_<ROLE>_* and is repeatable"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cases = _read_cases(args.input)
        ledger = CostLedger(args.max_cost_usd) if args.max_cost_usd is not None else None
        primary_roles = ("R1", "R2", "R3")
        tiebreak_roles = tuple(args.tiebreak_role)
        roles = primary_roles + tiebreak_roles
        configs = [
            ModelConfig.from_env(role, namespace="CMDRISK_REVIEWER")
            for role in roles
        ]
        if len({config.model for config in configs}) != len(configs):
            if tiebreak_roles:
                raise ConfigurationError("proxy audit requires distinct reviewer models")
            raise ConfigurationError("proxy audit requires three distinct reviewer models")
        reviewers = [
            PromptedModel(
                OpenAICompatibleClient(
                    config,
                    ledger=ledger,
                    cache_dir=args.cache_dir,
                )
            )
            for config in configs
        ]
        auditor = ProxyAuditor(
            tuple(reviewers[:3]),
            tuple(reviewers[3:]),
        )
        try:
            report = auditor.audit(cases)
        except DryRunError as exc:
            print(json.dumps(exc.plan, sort_keys=True), file=sys.stderr)
            return 0
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        if args.unresolved_out is not None:
            unresolved = [
                result for result in report.results if result.consensus == "unresolved"
            ]
            args.unresolved_out.parent.mkdir(parents=True, exist_ok=True)
            args.unresolved_out.write_text(
                "".join(item.model_dump_json() + "\n" for item in unresolved),
                encoding="utf-8",
            )
        print(json.dumps(report.model_dump(exclude={"results"}), sort_keys=True))
        return 0
    except (TeacherError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
