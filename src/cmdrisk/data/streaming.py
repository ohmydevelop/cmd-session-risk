"""Memory-bounded readers and scalable near-duplicate clustering.

The v3 corpus was built by materialising every source in RAM.  That works for
a few thousand sessions, but the million-scale v4 corpus streams a 878 MB
encrypted JSON array (125,280 Microsoft IoT sessions / 8.2 M commands) and
must therefore never call :func:`json.loads` on a whole document.

Two building blocks live here:

``iter_json_array_objects``
    Incremental top-level JSON-array reader.  Objects are decoded with
    :meth:`json.JSONDecoder.raw_decode`, so scanning cost is proportional to the
    document size and peak memory is one decode window (~4 MB) plus the largest
    single object.

``MinHashLSH``
    One-permutation-free banded MinHash LSH used to cluster near-duplicate
    sessions before any train/dev/test split.  It is the standard web-scale
    approach (Broder 1997; Leskovec et al. 2020) and replaces the O(n^2)
    all-pairs Jaccard comparison used for the small v3 corpus.
"""

from __future__ import annotations

import io
import json
import zipfile
import zlib
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CHUNK_CHARS = 4 * 1024 * 1024
_UINT64_MAX = (1 << 64) - 1
_FNV_PRIME = np.uint64(0x100000001B3)


class StreamingFormatError(RuntimeError):
    """Raised when a streamed document does not match the expected shape."""


@contextmanager
def open_json_text(
    path: str | Path, *, password: str | None = None
) -> Iterator[io.TextIOBase]:
    """Open a JSON document from a plain path or an (optionally encrypted) ZIP."""

    source = Path(path)
    if source.suffix.casefold() == ".zip" or zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            names = [name for name in archive.namelist() if name.casefold().endswith(".json")]
            if not names:
                raise StreamingFormatError(f"ZIP archive has no JSON document: {source}")
            name = names[0]
            password_bytes = password.encode("utf-8") if password else None
            try:
                raw = archive.open(name, pwd=password_bytes)
            except RuntimeError as exc:  # pragma: no cover - depends on archive
                raise StreamingFormatError(
                    f"could not decrypt {name} in {source}: {exc}"
                ) from exc
            with raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                yield text
        return
    with source.open("r", encoding="utf-8-sig", newline="") as text:
        yield text


def iter_json_array_objects(
    text: io.TextIOBase, *, chunk_chars: int = DEFAULT_CHUNK_CHARS
) -> Iterator[Any]:
    """Yield the top-level objects of a JSON array without buffering all of it."""

    if chunk_chars < 1:
        raise ValueError("chunk_chars must be positive")
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    eof = False
    while True:
        while position < len(buffer) and buffer[position] in " \t\r\n,":
            position += 1
        if position >= len(buffer):
            if eof:
                if not started:
                    raise StreamingFormatError("document is not a JSON array")
                return
            chunk = text.read(chunk_chars)
            if not chunk:
                eof = True
                continue
            buffer = buffer[position:] + chunk
            position = 0
            continue
        head = buffer[position]
        if not started:
            if head != "[":
                raise StreamingFormatError(f"expected '[' to open the JSON array, found {head!r}")
            started = True
            position += 1
            continue
        if head == "]":
            return
        try:
            value, end = decoder.raw_decode(buffer, position)
        except json.JSONDecodeError:
            if eof:
                raise
            chunk = text.read(chunk_chars)
            if not chunk:
                eof = True
                continue
            buffer = buffer[position:] + chunk
            position = 0
            continue
        position = end
        yield value


def iter_stream_values(
    text: io.TextIOBase, *, chunk_chars: int = DEFAULT_CHUNK_CHARS
) -> Iterator[Any]:
    """Yield every top-level JSON value from a text stream.

    The reader auto-detects the three shapes that appear in the source corpus:

    * a top-level JSON array (``[{...}, {...}]``) streamed element by element;
    * JSONL / newline-delimited objects, one value per line;
    * a single (possibly pretty-printed) JSON object.

    Peak memory stays at one decode window plus the largest single value, so a
    878 MB document is never materialised.
    """

    if chunk_chars < 1:
        raise ValueError("chunk_chars must be positive")
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False
    in_array: bool | None = None
    while True:
        while position < len(buffer) and buffer[position] in " \t\r\n,":
            position += 1
        if position >= len(buffer):
            if eof:
                return
            chunk = text.read(chunk_chars)
            if not chunk:
                eof = True
                continue
            buffer = buffer[position:] + chunk
            position = 0
            continue
        head = buffer[position]
        if in_array is None:
            in_array = head == "["
            if in_array:
                position += 1
                continue
        if in_array and head == "]":
            return
        try:
            value, end = decoder.raw_decode(buffer, position)
        except json.JSONDecodeError:
            if eof:
                raise
            chunk = text.read(chunk_chars)
            if not chunk:
                eof = True
                continue
            buffer = buffer[position:] + chunk
            position = 0
            continue
        position = end
        yield value


@contextmanager
def open_zip_member_text(
    path: str | Path, name: str, *, password: str | None = None
) -> Iterator[io.TextIOBase]:
    """Open one member of a ZIP archive as a streaming UTF-8 text handle."""

    source = Path(path)
    password_bytes = password.encode("utf-8") if password else None
    with zipfile.ZipFile(source) as archive:
        try:
            raw = archive.open(name, pwd=password_bytes)
        except RuntimeError as exc:  # pragma: no cover - depends on archive
            raise StreamingFormatError(f"could not decrypt {name} in {source}: {exc}") from exc
        with raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
            yield text


def iter_zip_member_values(
    path: str | Path,
    *,
    suffixes: tuple[str, ...] = (".json", ".jsonl"),
    password: str | None = None,
) -> Iterator[tuple[str, Any]]:
    """Stream ``(member_name, value)`` pairs from matching ZIP members."""

    source = Path(path)
    with zipfile.ZipFile(source) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.casefold().endswith(suffixes) and not name.endswith("/")
        ]
    for name in names:
        with open_zip_member_text(source, name, password=password) as text:
            for value in iter_stream_values(text):
                yield name, value


def iter_zip_member_texts(
    path: str | Path,
    *,
    suffixes: tuple[str, ...] = (".json", ".jsonl"),
    password: str | None = None,
) -> Iterator[tuple[str, io.TextIOBase]]:
    """Yield ``(member_name, text_handle)`` pairs from matching ZIP members.

    The archive and the current member remain open while the consumer advances
    the returned iterator.  This keeps large members streaming instead of
    materialising their decompressed contents in RAM.
    """

    source = Path(path)
    with zipfile.ZipFile(source) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.casefold().endswith(suffixes) and not name.endswith("/")
        ]
        password_bytes = password.encode("utf-8") if password else None
        for name in names:
            try:
                raw = archive.open(name, pwd=password_bytes)
            except RuntimeError as exc:  # pragma: no cover - depends on archive
                raise StreamingFormatError(
                    f"could not decrypt {name} in {source}: {exc}"
                ) from exc
            with raw, io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                yield name, text


def iter_jsonl_objects(path: str | Path, *, encoding: str = "utf-8-sig") -> Iterator[Any]:
    """Stream one JSON value per line without materialising the file."""

    source = Path(path)
    with source.open("r", encoding=encoding) as handle:
        for number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise StreamingFormatError(f"invalid JSON at {source}:{number}") from exc


def iter_microsoft_iot_items(
    path: str | Path, *, password: str | None = None
) -> Iterator[dict[str, Any]]:
    """Stream Microsoft IoT honeypot session objects from the encrypted ZIP."""

    with open_json_text(path, password=password) as text:
        for item in iter_json_array_objects(text):
            if isinstance(item, dict):
                yield item


# --------------------------------------------------------------------------
# Scalable near-duplicate clustering
# --------------------------------------------------------------------------


def hash64(text: str) -> int:
    """Deterministic 64-bit hash of a short token (two CRC32 lanes)."""

    payload = text.encode("utf-8", errors="replace")
    high = zlib.crc32(payload)
    low = zlib.crc32(payload, 0x9E3779B9)
    return ((high << 32) | low) & _UINT64_MAX


def shingle_hashes(
    skeleton: Sequence[str], *, ngram: int = 3, max_commands: int = 160
) -> np.ndarray:
    """Return the sorted unique hashes of command-``ngram`` shingles."""

    if ngram < 1:
        raise ValueError("ngram must be positive")
    tokens = list(skeleton[:max_commands])
    if not tokens:
        return np.empty(0, dtype=np.uint64)
    if len(tokens) < ngram:
        grams = ["\u001f".join(tokens)]
    else:
        grams = [
            "\u001f".join(tokens[index : index + ngram])
            for index in range(len(tokens) - ngram + 1)
        ]
    values = np.fromiter((hash64(gram) for gram in grams), dtype=np.uint64, count=len(grams))
    return np.unique(values)


def _permutation_parameters(num_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    mask = 1 << 63
    multiply = (rng.integers(0, mask, size=num_perm, dtype=np.uint64) | np.uint64(1))
    add = rng.integers(0, mask, size=num_perm, dtype=np.uint64)
    return multiply, add


def minhash_signatures(
    shingle_sets: Sequence[np.ndarray],
    *,
    num_perm: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Compute a ``(len(shingle_sets), num_perm)`` MinHash signature matrix."""

    if num_perm < 1:
        raise ValueError("num_perm must be positive")
    multiply, add = _permutation_parameters(num_perm, seed)
    signature = np.full((len(shingle_sets), num_perm), _UINT64_MAX, dtype=np.uint64)
    for index, values in enumerate(shingle_sets):
        if values.size == 0:
            continue
        permuted = values[:, None] * multiply[None, :] + add[None, :]
        signature[index] = permuted.min(axis=0)
    return signature


def _band_keys(signature: np.ndarray, *, bands: int, rows: int, seed: int) -> np.ndarray:
    if bands * rows != signature.shape[1]:
        raise ValueError("bands * rows must equal the signature width")
    salt = np.uint64(seed & _UINT64_MAX)
    view = signature.reshape(signature.shape[0], bands, rows)
    keys = view[:, :, 0].copy()
    keys += salt
    for row in range(1, rows):
        keys = keys * _FNV_PRIME + view[:, :, row]
    return keys


def candidate_pairs(
    signature: np.ndarray,
    *,
    bands: int = 8,
    rows: int = 8,
    max_bucket_size: int = 512,
    max_pairs: int = 20_000_000,
) -> tuple[np.ndarray, int]:
    """Return LSH candidate index pairs as an ``(M, 2)`` int64 array.

    Pairs are returned in a compact NumPy buffer instead of a Python list of
    tuples: a million-scale corpus can propose tens of millions of candidate
    pairs and the object overhead of ``list[tuple[int, int]]`` would otherwise
    dominate memory.  Buckets larger than ``max_bucket_size`` are skipped
    entirely (boilerplate buckets carry no near-duplicate signal) and reported
    so the caller can surface the loss instead of hiding it.
    """

    if signature.ndim != 2:
        raise ValueError("signature must be a 2-D matrix")
    keys = _band_keys(signature, bands=bands, rows=rows, seed=0x5EED)
    flat = keys.reshape(-1)
    owner = np.repeat(np.arange(signature.shape[0], dtype=np.int64), bands)
    order = np.argsort(flat, kind="stable")
    sorted_keys = flat[order]
    sorted_owner = owner[order]

    total = int(sorted_keys.size)
    if total == 0:
        return np.empty((0, 2), dtype=np.int64), 0
    boundaries = np.flatnonzero(np.diff(sorted_keys) != 0) + 1
    starts = np.concatenate((np.zeros(1, dtype=np.int64), boundaries)).astype(np.int64)
    ends = np.concatenate((boundaries, np.array([total], dtype=np.int64))).astype(np.int64)

    left_parts: list[np.ndarray] = []
    right_parts: list[np.ndarray] = []
    skipped = 0
    produced = 0
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        if end - start < 2:
            continue
        members = np.unique(sorted_owner[start:end])
        if members.size > max_bucket_size:
            skipped += 1
            continue
        if members.size == 2:
            left_parts.append(members[0:1])
            right_parts.append(members[1:2])
            produced += 1
        else:
            local_left, local_right = np.triu_indices(int(members.size), k=1)
            left_parts.append(members[local_left])
            right_parts.append(members[local_right])
            produced += int(local_left.size)
        if produced > max_pairs:
            raise StreamingFormatError(
                f"LSH produced more than {max_pairs} candidate pairs; "
                "lower max_bucket_size or raise the threshold"
            )
    if not left_parts:
        return np.empty((0, 2), dtype=np.int64), skipped
    pairs = np.stack(
        [np.concatenate(left_parts), np.concatenate(right_parts)], axis=1
    ).astype(np.int64, copy=False)
    return pairs, skipped


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    candidate_pairs: int
    verified_pairs: int
    skipped_buckets: int
    forced_pairs: int = 0


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    intersection = np.intersect1d(left, right, assume_unique=True).size
    union = left.size + right.size - intersection
    return float(intersection) / float(union) if union else 0.0


def cluster_records(
    shingle_sets: Sequence[np.ndarray],
    *,
    threshold: float = 0.9,
    num_perm: int = 64,
    bands: int = 8,
    rows: int = 8,
    seed: int = 42,
    max_bucket_size: int = 512,
    max_pairs: int = 20_000_000,
    extra_pairs: Sequence[tuple[int, int]] = (),
) -> ClusterResult:
    """Cluster near-duplicate sessions with banded MinHash LSH.

    Candidate pairs proposed by shared LSH bands are verified with an exact
    Jaccard computation over the shingle hashes, so the resulting clusters are
    never looser than ``threshold``.  ``extra_pairs`` are unioned before any
    similarity test; callers use them for provenance that must never be broken
    by a similarity threshold (explicit ``family_id`` groups, parent/child
    lineage, and exact duplicates of very short sessions).
    """

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    count = len(shingle_sets)
    if count == 0:
        return ClusterResult(np.empty(0, dtype=np.int64), 0, 0, 0, 0)

    signature = minhash_signatures(shingle_sets, num_perm=num_perm, seed=seed)

    # Short, information-poor sessions (e.g. a single command) are placed in
    # their own family unless an explicit lineage already groups them; without
    # enough shingles the MinHash estimate is not meaningful.
    usable = np.fromiter(
        (values.size >= max(bands, 4) for values in shingle_sets),
        dtype=bool,
        count=count,
    )
    lsh_index = np.flatnonzero(usable)
    if lsh_index.size:
        pairs, skipped = candidate_pairs(
            signature[lsh_index],
            bands=bands,
            rows=rows,
            max_bucket_size=max_bucket_size,
            max_pairs=max_pairs,
        )
    else:
        pairs, skipped = np.empty((0, 2), dtype=np.int64), 0

    parent = np.arange(count, dtype=np.int64)

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = int(parent[root])
        while parent[node] != root:
            parent[node], node = root, int(parent[node])
        return root

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    forced = 0
    for left, right in extra_pairs:
        left_index = int(left)
        right_index = int(right)
        if left_index == right_index:
            continue
        if find(left_index) != find(right_index):
            union(left_index, right_index)
            forced += 1

    verified = 0
    for left_local, right_local in pairs:
        left = int(lsh_index[int(left_local)])
        right = int(lsh_index[int(right_local)])
        if find(left) == find(right):
            continue
        if jaccard(shingle_sets[left], shingle_sets[right]) >= threshold:
            union(left, right)
            verified += 1

    labels = np.fromiter((find(index) for index in range(count)), dtype=np.int64, count=count)
    return ClusterResult(labels, int(pairs.shape[0]), verified, skipped, forced)


def chunked(iterable: Iterable[Any], size: int) -> Iterator[list[Any]]:
    """Yield fixed-size lists from an iterable (last chunk may be short)."""

    if size < 1:
        raise ValueError("size must be positive")
    chunk: list[Any] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
