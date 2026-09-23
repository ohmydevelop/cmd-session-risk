"""Source manifests and reproducible file hashes."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .yaml_utils import load_yaml_file

REQUIRED_SOURCE_FIELDS = {
    "id",
    "adapter",
    "url",
    "license",
    "sha256",
    "password",
    "redistribution",
}

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MD5_RE = re.compile(r"^[0-9a-fA-F]{32}$")
_REVISION_RE = re.compile(r"^(?:[A-Za-z0-9_.-]+@)?[0-9a-fA-F]{7,64}$")
_SCP_GIT_RE = re.compile(r"^[^@\s]+@[^:\s]+:[^\s]+$")


def md5_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - required for upstream source verification
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksum(value: str) -> tuple[str, str]:
    """Parse ``sha256:<hex>``, ``md5:<hex>``, or a plain SHA256 digest."""

    if not isinstance(value, str):
        raise ValueError("checksum must be a string")
    candidate = value.strip()
    if not candidate:
        raise ValueError("checksum must not be empty")
    if ":" in candidate:
        algorithm, digest = candidate.split(":", 1)
        algorithm = algorithm.strip().lower()
        digest = digest.strip()
        if algorithm not in {"sha256", "md5"}:
            raise ValueError(f"unsupported checksum algorithm: {algorithm or '<empty>'}")
    else:
        algorithm, digest = "sha256", candidate
    pattern = _SHA256_RE if algorithm == "sha256" else _MD5_RE
    if not pattern.fullmatch(digest):
        raise ValueError(f"invalid {algorithm} digest: {digest!r}")
    return algorithm, digest.lower()


def verify_file_checksum(path: str | Path, checksum: str) -> tuple[str, str]:
    """Verify one checksum declaration and return ``(algorithm, digest)``."""

    algorithm, expected = parse_checksum(checksum)
    actual = sha256_file(path) if algorithm == "sha256" else md5_file(path)
    if actual.casefold() != expected.casefold():
        raise ValueError(
            f"{algorithm.upper()} mismatch for {Path(path).name}: expected {expected}, got {actual}"
        )
    return algorithm, expected


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_url(url: str) -> None:
    if not url:
        raise ValueError("url must not be empty")
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"}:
        if not parsed.netloc or not parsed.path:
            raise ValueError(f"invalid HTTP(S) source URL: {url!r}")
        return
    if parsed.scheme in {"ssh", "git"}:
        if not parsed.netloc or not parsed.path:
            raise ValueError(f"invalid Git source URL: {url!r}")
        return
    if _SCP_GIT_RE.fullmatch(url):
        return
    raise ValueError(f"unsupported source URL scheme: {url!r}")


def _validate_revision(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("revision must be a non-empty string when present")
    revision = value.strip()
    if not _REVISION_RE.fullmatch(revision):
        raise ValueError(f"revision must be a pinned 7-64 character Git commit: {revision!r}")
    return revision


def source_checksums(source: dict[str, Any]) -> list[str]:
    """Return validated checksum declarations, rejecting conflicts."""

    declarations: list[str] = []
    raw_sha256 = source.get("sha256")
    if raw_sha256 is not None:
        if not isinstance(raw_sha256, str) or not raw_sha256.strip():
            raise ValueError("sha256 must be null or a non-empty checksum string")
        algorithm, _ = parse_checksum(raw_sha256)
        if algorithm != "sha256":
            raise ValueError("sha256 field must contain a SHA256 digest")
        declarations.append(raw_sha256.strip())

    raw_checksum = source.get("checksum")
    if raw_checksum is not None:
        if not isinstance(raw_checksum, str) or not raw_checksum.strip():
            raise ValueError("checksum must be null or a non-empty checksum string")
        declarations.append(raw_checksum.strip())

    parsed = [parse_checksum(value) for value in declarations]
    by_algorithm: dict[str, str] = {}
    for algorithm, digest in parsed:
        previous = by_algorithm.get(algorithm)
        if previous is not None and previous != digest:
            raise ValueError(f"conflicting {algorithm} checksum declarations")
        by_algorithm[algorithm] = digest
    return declarations


def _is_git_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path if parsed.scheme else url.split(":", 1)[-1]
    return path.rstrip("/").endswith(".git")


def _http_url_is_immutable(url: str) -> bool:
    return bool(re.search(r"(?<![0-9a-fA-F])[0-9a-fA-F]{40,64}(?![0-9a-fA-F])", url))


def load_data_sources(path: str | Path) -> list[dict[str, Any]]:
    document = load_yaml_file(path)
    if not isinstance(document, dict) or not isinstance(document.get("sources"), list):
        raise ValueError("data source manifest must contain a 'sources' list")
    sources: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, source in enumerate(document["sources"]):
        if not isinstance(source, dict):
            raise ValueError(f"source #{index} must be a mapping")
        missing = REQUIRED_SOURCE_FIELDS.difference(source)
        if missing:
            raise ValueError(f"source #{index} missing fields: {sorted(missing)}")
        source_id = str(source["id"]).strip()
        if not source_id or source_id in seen:
            raise ValueError(f"source #{index} has an empty or duplicate id")
        if Path(source_id).name != source_id or source_id in {".", ".."}:
            raise ValueError(f"source #{index} id is not path-safe: {source_id!r}")
        adapter = str(source["adapter"]).strip()
        if not adapter:
            raise ValueError(f"source #{index} has an empty adapter")
        license_name = str(source["license"]).strip()
        if not license_name:
            raise ValueError(f"source #{index} has an empty license")
        url = str(source["url"]).strip()
        try:
            _validate_url(url)
        except ValueError as exc:
            raise ValueError(f"source #{index} ({source_id}): {exc}") from exc
        try:
            checksums = source_checksums(source)
            revision = _validate_revision(source.get("revision"))
        except ValueError as exc:
            raise ValueError(f"source #{index} ({source_id}): {exc}") from exc

        if _is_git_url(url):
            if revision is None:
                raise ValueError(
                    f"source #{index} ({source_id}): Git sources require a pinned revision"
                )
        elif not checksums and not _http_url_is_immutable(url):
            raise ValueError(
                f"source #{index} ({source_id}): mutable HTTP sources require a checksum "
                "or an immutable commit-pinned URL"
            )
        seen.add(source_id)
        sources.append(source)
    return sources


def write_sha256_manifest(
    paths: Iterable[str | Path], output: str | Path, *, root: str | Path | None = None
) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = Path(root) if root is not None else output_path.parent
    lines: list[str] = []
    for value in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        try:
            display = value.resolve().relative_to(base.resolve())
        except ValueError:
            display = value
        lines.append(f"{sha256_file(value)}  {display.as_posix()}")
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    temporary.replace(output_path)
    return output_path


def load_sha256_manifest(path: str | Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for line_number, raw_line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            digest, name = raw_line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid SHA256 manifest line {line_number}") from exc
        manifest[name] = digest
    return manifest


def write_json_atomic(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
