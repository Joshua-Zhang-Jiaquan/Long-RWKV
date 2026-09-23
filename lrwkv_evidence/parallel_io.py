"""Bounded, ordered parallel reads of independent JSON evidence files."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path


def _read_json(path):
    path = Path(path)
    raw = path.read_bytes()
    return path, raw, json.loads(raw)


def read_json_parallel(paths, *, max_workers=16, max_inflight=64):
    """Yield (Path, exact bytes, parsed JSON) in input order.

    At most ``max_inflight`` reads are submitted at once. The input iterable is
    consumed lazily; callers choose its deterministic order. File and decoding
    errors propagate without skipping records. Closing the iterator cancels
    queued work and waits for already-running reads to finish.
    """
    if max_workers < 1 or max_inflight < 1:
        raise ValueError('worker and in-flight limits must be positive')
    source = iter(paths)
    pending = deque()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        try:
            for _ in range(max_inflight):
                try:
                    path = next(source)
                except StopIteration:
                    break
                pending.append(executor.submit(_read_json, path))
            while pending:
                yield pending.popleft().result()
                try:
                    path = next(source)
                except StopIteration:
                    continue
                pending.append(executor.submit(_read_json, path))
        finally:
            for future in pending:
                future.cancel()
