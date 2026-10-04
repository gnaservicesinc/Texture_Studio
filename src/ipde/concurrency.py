"""Bounded file/CPU work with deterministic, caller-owned publication.

File copies, hashing and scientific array decoding do not have an MPS kernel.
Threads overlap their native I/O/computation without moving or converting data.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
from typing import Callable, Iterable, Iterator, TypeVar

_Input = TypeVar("_Input")
_Output = TypeVar("_Output")


def available_workers() -> int:
    count = getattr(os, "process_cpu_count", os.cpu_count)()
    return max(1, count or 1)


def resolve_workers(workers: int | None = None) -> int:
    """None/zero selects available process cores; explicit counts must be positive."""
    if workers is not None and (not isinstance(workers, int) or isinstance(workers, bool) or workers < 0):
        raise ValueError("workers must be zero (automatic) or a positive integer")
    return available_workers() if workers in {None, 0} else workers


def memory_limited_workers(workers: int | None, bytes_per_task: int, *, budget_bytes: int = 512 * 1024**2) -> int:
    """Keep decoded full-resolution work bounded even on many-core systems."""
    return min(resolve_workers(workers), max(1, budget_bytes // max(1, bytes_per_task)))


def ordered_map(function: Callable[[_Input], _Output], items: Iterable[_Input], *,
                workers: int | None = None) -> Iterator[_Output]:
    """Yield in input order with at most one submitted task per worker.

    Only the caller consumes results and changes manifests/caches. On failure,
    finish active workers before returning so temporary output can be removed
    safely. Unlike Executor.map on older Python, submission is never unbounded.
    """
    count = resolve_workers(workers)
    source = iter(items)
    if count == 1:
        for item in source:
            yield function(item)
        return
    with ThreadPoolExecutor(max_workers=count, thread_name_prefix="ipde-files") as executor:
        pending = deque()
        for _ in range(count):
            try:
                pending.append(executor.submit(function, next(source)))
            except StopIteration:
                break
        try:
            while pending:
                result = pending.popleft().result()
                yield result
                del result
                try:
                    pending.append(executor.submit(function, next(source)))
                except StopIteration:
                    pass
        finally:
            for future in pending:
                future.cancel()
