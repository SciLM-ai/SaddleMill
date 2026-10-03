"""Buffered diagnostic appenders and lightweight I/O accounting.

The helpers in this module are diagnostics-only. They do not evaluate forces,
change optimizer state, or alter convergence decisions. High-frequency rows are
serialized in memory and written in bounded batches while preserving the same
CSV/JSONL schemas and append semantics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Iterable, Mapping, Sequence


@dataclass
class DiagnosticIOStats:
    open_calls: int = 0
    write_calls: int = 0
    flush_calls: int = 0
    rows_serialized: int = 0
    bytes_written: int = 0
    files_created: int = 0
    serialization_ns: int = 0
    filesystem_ns: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


class _BufferedAppender:
    def __init__(self, path, *, max_records: int = 16, max_bytes: int = 262144):
        self.path = Path(path)
        self.max_records = max(1, int(max_records))
        self.max_bytes = max(1, int(max_bytes))
        self.stats = DiagnosticIOStats()
        self._pending: list[str] = []
        self._pending_bytes = 0
        self._handle = None
        self._closed = False

    def _open(self):
        if self._handle is not None:
            return self._handle
        start = time.perf_counter_ns()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists()
        self._handle = self.path.open("a", encoding="utf-8", newline="")
        self.stats.open_calls += 1
        if not existed:
            self.stats.files_created += 1
        self.stats.filesystem_ns += time.perf_counter_ns() - start
        return self._handle

    def _queue_text(self, text: str, *, rows: int = 1):
        if self._closed:
            raise RuntimeError(f"diagnostic writer already closed: {self.path}")
        self._pending.append(text)
        self._pending_bytes += len(text.encode("utf-8"))
        self.stats.rows_serialized += int(rows)
        if len(self._pending) >= self.max_records or self._pending_bytes >= self.max_bytes:
            self.flush()

    def flush(self, *, durable: bool = False):
        if not self._pending:
            if durable and self._handle is not None:
                start = time.perf_counter_ns()
                self._handle.flush()
                os.fsync(self._handle.fileno())
                self.stats.flush_calls += 1
                self.stats.filesystem_ns += time.perf_counter_ns() - start
            return
        payload = "".join(self._pending)
        self._pending.clear()
        self._pending_bytes = 0
        handle = self._open()
        start = time.perf_counter_ns()
        handle.write(payload)
        self.stats.write_calls += 1
        self.stats.bytes_written += len(payload.encode("utf-8"))
        if durable:
            handle.flush()
            os.fsync(handle.fileno())
            self.stats.flush_calls += 1
        self.stats.filesystem_ns += time.perf_counter_ns() - start

    def close(self, *, durable: bool = False):
        if self._closed:
            return
        try:
            self.flush(durable=durable)
        finally:
            if self._handle is not None:
                start = time.perf_counter_ns()
                self._handle.close()
                self.stats.filesystem_ns += time.perf_counter_ns() - start
            self._closed = True


class BufferedJSONLAppender(_BufferedAppender):
    def append(self, row: Mapping[str, object], *, default=None):
        start = time.perf_counter_ns()
        kwargs = {"sort_keys": True, "separators": (",", ":")}
        if default is not None:
            kwargs["default"] = default
        text = json.dumps(dict(row), **kwargs) + "\n"
        self.stats.serialization_ns += time.perf_counter_ns() - start
        self._queue_text(text)


class BufferedCSVAppender(_BufferedAppender):
    """CSV append with one schema check/open per recorder lifetime.

    Existing additive headers are migrated once before opening the long-lived
    append handle. Incompatible schemas fail closed exactly as the predecessor
    helper did.
    """

    def __init__(self, path, fieldnames: Sequence[str], **kwargs):
        self.fieldnames = list(fieldnames)
        self._prepared = False
        self._needs_header = False
        super().__init__(path, **kwargs)

    def _prepare(self):
        if self._prepared:
            return
        start = time.perf_counter_ns()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists()
        size = self.path.stat().st_size if exists else 0
        if size > 0:
            with self.path.open("r", encoding="utf-8", newline="") as handle:
                self.stats.open_calls += 1
                reader = csv.DictReader(handle)
                old_fields = list(reader.fieldnames or [])
                if old_fields != self.fieldnames:
                    if not old_fields or not set(old_fields).issubset(set(self.fieldnames)):
                        raise ValueError(
                            f"Refusing incompatible diagnostic CSV schema migration for {self.path}: "
                            f"old={old_fields}, new={self.fieldnames}"
                        )
                    rows = list(reader)
                    with tempfile.NamedTemporaryFile(
                        mode="w", encoding="utf-8", newline="", dir=self.path.parent,
                        delete=False,
                    ) as temp_handle:
                        temp_path = temp_handle.name
                        writer = csv.DictWriter(temp_handle, fieldnames=self.fieldnames)
                        writer.writeheader()
                        for old_row in rows:
                            writer.writerow({key: old_row.get(key, "") for key in self.fieldnames})
                    os.replace(temp_path, self.path)
        else:
            self._needs_header = True
        self.stats.filesystem_ns += time.perf_counter_ns() - start
        self._prepared = True

    def append(self, row: Mapping[str, object]):
        self._prepare()
        start = time.perf_counter_ns()
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=self.fieldnames)
        if self._needs_header:
            writer.writeheader()
            self._needs_header = False
        writer.writerow({key: row.get(key, "") for key in self.fieldnames})
        text = stream.getvalue()
        self.stats.serialization_ns += time.perf_counter_ns() - start
        self._queue_text(text)


def append_jsonl_durable(path, row: Mapping[str, object], *, default=str):
    writer = BufferedJSONLAppender(path, max_records=1)
    try:
        writer.append(row, default=default)
        writer.close(durable=True)
    except Exception:
        writer.close()
        raise
    return writer.stats.to_dict()
