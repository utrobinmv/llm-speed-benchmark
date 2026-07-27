"""Common worker helpers extracted from bench_multi, bench_vision, bench_audio.

Shared functionality:
- Periodic wall-time sender thread
- on_chunk live-updates callback factory
- Stats message senders (success / error)
"""

from __future__ import annotations

import time
from threading import Lock, Thread
from typing import Any, Callable, Optional

from .streaming import ChunkCallback, StreamMetrics
from .utils import format_time


# ---------------------------------------------------------------------------
# Time-sender thread
# ---------------------------------------------------------------------------

def make_time_sender(
    q: Any,
    worker_id: int,
    start_time: float,
    duration: Optional[int],
    state: Optional[dict[str, Any]] = None,
    state_lock: Optional[Lock] = None,
) -> Thread:
    """Return a daemon Thread that periodically sends wall-time updates to *q*.

    Args:
        q: Queue-like object with ``.put()``.
        worker_id: Numeric ID of the worker.
        start_time: ``time.time()`` value at worker start.
        duration: Max duration in seconds (``None`` = unlimited).
        state: Shared state dict with ``total_gen`` and ``chunk_count`` keys
            (used by vision/audio workers for live avg).
        state_lock: Lock protecting *state* (must be provided with *state*).

    Returns:
        A started daemon ``Thread``.
    """

    def _send() -> None:
        while True:
            if duration is not None:
                elapsed = time.time() - start_time
                if elapsed >= duration:
                    break
            time.sleep(1)
            wall = time.time() - start_time
            msg: dict[str, Any] = {
                "type": "time",
                "id": worker_id,
                "wall": format_time(wall),
            }
            if state is not None and state_lock is not None and wall > 0:
                with state_lock:
                    tg = state["total_gen"]
                    cc = state["chunk_count"]
                est_gen = tg + cc
                avg_sp = round(est_gen / wall, 1) if wall > 0 else 0
                msg["avg"] = avg_sp
            q.put(msg)

    t = Thread(target=_send, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# on_chunk callback factory
# ---------------------------------------------------------------------------

def make_on_chunk_callback(
    q: Any,
    worker_id: int,
    total_gen_ref: list[int],
    total_chunks: list[int],
    total_ttft_ref: list[float],
    get_media_name: Optional[Callable[[], str]] = None,
    media_count_ref: Optional[list[int]] = None,
) -> ChunkCallback:
    """Factory returning a ``ChunkCallback`` that pushes live updates to *q*.

    Args:
        q: Queue-like object with ``.put()``.
        worker_id: Numeric ID of the worker.
        total_gen_ref: Single-element ``list[int]`` holding cumulative generated
            tokens (mutable so the callback can read the outer variable).
        total_chunks: Single-element ``list[int]`` holding cumulative chunks.
        total_ttft_ref: Single-element ``list[float]`` holding cumulative TTFT.
        get_media_name: Optional callable returning the current media name
            (used by vision/audio workers).
        media_count_ref: Optional single-element ``list[int]`` holding the current
            media count (updated before each call by the worker loop).
    """

    def _on_chunk(
        chunk_count: int,
        inst_sp: float,
        ttft: Optional[float],
        tail: str,
        wall_elapsed: float,
    ) -> None:
        current_ttft = ttft if ttft is not None else 0
        ttft_sum_live = total_ttft_ref[0] + current_ttft
        gen_est = total_gen_ref[0] + chunk_count
        avg_sp = round(gen_est / wall_elapsed, 1) if wall_elapsed > 0 else 0

        msg: dict[str, Any] = {
            "type": "live",
            "id": worker_id,
            "tail": tail,
            "tok": chunk_count,
            "est_tok": gen_est,
            "chunks": total_chunks[0] + chunk_count,
            "inst": inst_sp,
            "avg": avg_sp,
            "ttft": ttft if ttft is not None else 0,
            "ttft_sum": round(ttft_sum_live, 2),
            "wall": format_time(wall_elapsed),
        }

        if get_media_name is not None:
            msg["media"] = get_media_name()
        if media_count_ref is not None and media_count_ref[0] > 0:
            msg["media_count"] = media_count_ref[0]

        q.put(msg)

    return _on_chunk


# ---------------------------------------------------------------------------
# Stats senders
# ---------------------------------------------------------------------------

def send_stats(
    q: Any,
    worker_id: int,
    call_count: int,
    metrics: StreamMetrics,
    total_gen: int,
    total_chunks: int,
    total_ttft: float,
    wall_total: float,
    media_name: Optional[str] = None,
    media_count: int = 0,
    tail: str = "",
    round_num: Optional[int] = None,
    prompt_tokens: Optional[int] = None,
    total_tokens: Optional[int] = None,
) -> None:
    """Send a ``stats`` message to *q* after a successful call.

    Args:
        q: Queue-like object with ``.put()``.
        worker_id: Numeric ID of the worker.
        call_count: Number of calls made so far (1-indexed).
        metrics: ``StreamMetrics`` from the last call.
        total_gen: Cumulative generated tokens.
        total_chunks: Cumulative chunks.
        total_ttft: Cumulative TTFT.
        wall_total: Wall-clock elapsed since worker start.
        media_name: Optional media file stem (vision/audio).
        media_count: Number of media items in the request.
        tail: Trailing text snippet of the response.
        round_num: Current round number (bench_multi).
        prompt_tokens: Prompt tokens from API usage (bench_multi).
        total_tokens: prompt + completion tokens (bench_multi).
    """
    avg_speed = total_gen / wall_total if wall_total > 0 else 0

    msg: dict[str, Any] = {
        "type": "stats",
        "id": worker_id,
        "calls": call_count,
        "g": total_gen,
        "cg": metrics.completion_tokens,
        "chunks": total_chunks,
        "est_gen": total_gen,
        "speed": round(metrics.call_speed, 1),
        "avg_speed": round(avg_speed, 1),
        "inst_speed": round(metrics.instant_speed, 1),
        "ttft": round(metrics.ttft, 2) if metrics.ttft is not None else 0,
        "ttft_sum": round(total_ttft, 2),
        "tail": tail,
        "wall": format_time(wall_total),
    }

    if media_name is not None:
        msg["media"] = media_name
        msg["media_count"] = media_count
    if round_num is not None:
        msg["round"] = round_num
    if prompt_tokens is not None:
        msg["p"] = prompt_tokens
    if total_tokens is not None:
        msg["total"] = total_tokens

    q.put(msg)


def send_error_stats(
    q: Any,
    worker_id: int,
    call_count: int,
    total_gen: int,
    wall_total: float,
    media_name: Optional[str] = None,
    media_count: int = 0,
    tail: str = "",
) -> None:
    """Send a ``stats`` message to *q* after an error or empty response.

    Args:
        q: Queue-like object with ``.put()``.
        worker_id: Numeric ID of the worker.
        call_count: Number of calls made so far (1-indexed).
        total_gen: Cumulative generated tokens (unchanged by this call).
        wall_total: Wall-clock elapsed since worker start.
        media_name: Optional media file stem (vision/audio).
        media_count: Number of media items in the request.
        tail: Trailing text snippet or ``"error"`` / ``"empty"``.
    """
    msg: dict[str, Any] = {
        "type": "stats",
        "id": worker_id,
        "calls": call_count,
        "g": total_gen,
        "cg": 0,
        "speed": 0,
        "avg_speed": 0,
        "inst_speed": 0,
        "ttft": 0,
        "tail": tail,
        "wall": format_time(wall_total),
    }

    if media_name is not None:
        msg["media"] = media_name
        msg["media_count"] = media_count

    q.put(msg)