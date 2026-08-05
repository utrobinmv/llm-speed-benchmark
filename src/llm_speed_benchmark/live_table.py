"""
llm_speed_benchmark/live_table.py

Shared LiveTable base class extracted from bench_multi, bench_vision, and bench_audio.
Provides common worker-data management, message merging, and Rich table rendering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rich.console import Console
from rich.table import Table, box as _rich_box


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseLiveTable(ABC):
    """Abstract base for Rich Live tables used across benchmark modes.

    Subclasses implement ``render()`` (or ``__rich__()``) to produce the final
    Table with mode-specific columns and title.
    """

    def __init__(
        self,
        duration: int | None,
        total_workers: int,
        response_width: int = 60,
    ) -> None:
        self.duration = duration
        self.total_workers = total_workers
        self.response_width = response_width
        self.workers: dict[int, dict[str, Any]] = {}
        self._errors: dict[int, str] = {}
        self.console = Console()

    # ---- data helpers -----------------------------------------------------

    @staticmethod
    def _merge(w: dict[str, Any], data: dict[str, Any]) -> None:
        """Merge non-None values from *data* into worker dict *w*."""
        for k, v in data.items():
            if v is not None:
                w[k] = v

    def update_stats(self, msg: dict[str, Any]) -> None:
        """Merge fields from a ``stats`` message into the worker dict."""
        w = self.workers.setdefault(msg["id"], {})
        self._merge(w, {
            "calls": msg.get("calls"),
            "gen": msg.get("g"),
            "gen_est": msg.get("est_gen"),
            "chunks": msg.get("chunks"),
            "call_gen": msg.get("cg"),
            "speed": msg.get("inst_speed"),
            "avg": msg.get("avg_speed"),
            "ttft": msg.get("ttft"),
            "ttft_sum": msg.get("ttft_sum"),
            "wall": msg.get("wall"),
            "tail": msg.get("tail"),
        })

    def update_live(self, msg: dict[str, Any]) -> None:
        """Merge fields from a ``live`` message into the worker dict."""
        w = self.workers.setdefault(msg["id"], {})
        self._merge(w, {
            "speed": msg.get("inst"),
            "avg": msg.get("avg"),
            "ttft": msg.get("ttft"),
            "ttft_sum": msg.get("ttft_sum"),
            "gen_est": msg.get("est_tok"),
            "chunks": msg.get("chunks"),
            "wall": msg.get("wall"),
            "tail": msg.get("tail"),
        })

    def update_time(self, msg: dict[str, Any]) -> None:
        """Merge fields from a ``time`` message into the worker dict."""
        w = self.workers.setdefault(msg["id"], {})
        self._merge(w, {
            "wall": msg.get("wall"),
            "avg": msg.get("avg"),
        })

    def mark_error(self, worker_id: int, traceback_str: str) -> None:
        """Record a traceback for *worker_id*."""
        self._errors[worker_id] = traceback_str

    def mark_stopped(self, worker_id: int, error_msg: str) -> None:
        """Mark a worker as stopped due to an error."""
        w = self.workers.setdefault(worker_id, {})
        w["stopped"] = True
        w["error"] = error_msg

    # ---- text cleaning ----------------------------------------------------

    def _clean_tail(self, tail: str, max_len: int | None = None) -> str:
        """Clean and truncate response text for the Response column.

        Replaces wide characters (CJK, emoji, pictograms -- 2+ terminal cells)
        with dots so they do not break column width.  Cyrillic, Latin, and
        other single-cell characters are preserved.

        Uses the most comprehensive wide-char detection from bench_multi.
        """
        if max_len is None:
            max_len = self.response_width
        if not tail:
            return tail

        # Wide-character detection (2+ terminal cells)
        def is_wide(c: str) -> bool:
            cp = ord(c)
            # CJK Unified Ideographs and extensions
            if 0x4E00 <= cp <= 0x9FFF:
                return True
            if 0x3400 <= cp <= 0x4DBF:
                return True
            if 0x20000 <= cp <= 0x2A6DF:
                return True
            if 0x2A700 <= cp <= 0x2B73F:
                return True
            if 0x2B740 <= cp <= 0x2B81F:
                return True
            if 0x2B820 <= cp <= 0x2CEAF:
                return True
            if 0xF900 <= cp <= 0xFAFF:
                return True
            if 0x2F800 <= cp <= 0x2FA1F:
                return True
            # CJK symbols and punctuation
            if 0x3000 <= cp <= 0x303F:
                return True
            if 0xFF01 <= cp <= 0xFF60:
                return True
            # Hangul
            if 0xAC00 <= cp <= 0xD7AF:
                return True
            if 0xD7B0 <= cp <= 0xD7FF:
                return True
            # Katakana/Hiragana (wide)
            if 0x30A0 <= cp <= 0x30FF:
                return True
            if 0x3040 <= cp <= 0x309F:
                return True
            # Thai and other SE-Asian
            if 0x0E01 <= cp <= 0x0E3E:
                return True
            if 0x0E40 <= cp <= 0x0E4E:
                return True
            # Emoji and supplementary symbols
            if cp >= 0x1F000:
                return True
            if 0x1F300 <= cp <= 0x1F9FF:
                return True
            if 0x1FA00 <= cp <= 0x1FA6F:
                return True
            if 0x1FA70 <= cp <= 0x1FAFF:
                return True
            if 0x2600 <= cp <= 0x26FF:  # Misc Symbols
                return True
            if 0x2700 <= cp <= 0x27BF:  # Dingbats
                return True
            if 0x2300 <= cp <= 0x23FF:  # Misc Technical
                return True
            if 0xFE00 <= cp <= 0xFE0F:  # Variation selectors -- skip
                return False
            if 0xFE30 <= cp <= 0xFE4F:
                return True
            if 0x2000 <= cp <= 0x206F:  # General punctuation -- narrow
                return False
            return False

        # Pass 1: replace wide / non-printable characters, preserve tags
        clean: str = ""
        in_tag = False
        for c in tail:
            if c == "[":
                in_tag = True
                clean += c
                continue
            if c == "]" and in_tag:
                in_tag = False
                clean += c
                continue
            if in_tag:
                clean += c
                continue
            # Outside of tags: replace problematic characters
            if c in "\n\r\t":
                clean += " "
            elif c.isprintable() and not is_wide(c):
                clean += c
            else:
                clean += "."

        # Pass 2: collapse multiple spaces (only outside tags)
        prev: str | None = None
        while prev != clean:
            prev = clean
            result: str = ""
            in_tag = False
            prev_space = False
            for c in prev:
                if c == "[":
                    in_tag = True
                    result += c
                    prev_space = False
                    continue
                if c == "]" and in_tag:
                    in_tag = False
                    result += c
                    prev_space = False
                    continue
                if in_tag:
                    result += c
                    prev_space = False
                    continue
                if c == " " and prev_space:
                    continue
                prev_space = c == " "
                result += c
            clean = result

        clean = clean.strip()
        if not clean:
            return tail

        # Pass 3: truncate to visual width (counting only visible characters)
        def visual_length(s: str) -> int:
            length = 0
            in_t = False
            for ch in s:
                if ch == "[":
                    in_t = True
                    continue
                if ch == "]" and in_t:
                    in_t = False
                    continue
                if not in_t:
                    length += 1
            return length

        vlen = visual_length(clean)
        max_vlen = max_len - 3  # room for "..."
        if vlen > max_vlen:
            truncated: str = ""
            count = 0
            in_t = False
            for ch in clean:
                if ch == "[":
                    in_t = True
                    truncated += ch
                    continue
                if ch == "]" and in_t:
                    in_t = False
                    truncated += ch
                    continue
                if in_t:
                    truncated += ch
                    continue
                if count >= max_vlen:
                    break
                count += 1
                truncated += ch
            clean = "..." + truncated

        return clean

    # ---- table helpers ----------------------------------------------------

    def _make_table(self) -> Table:
        """Create a Rich Table with the common columns.

        Columns: W#, Calls, Gen, Call, Speed, Avg, TTFT, TTFT sum, Wall, Response.
        """
        table = Table(box=_rich_box.DOUBLE, show_header=True)

        table.add_column("W#", style="cyan", width=4, justify="right")
        table.add_column("Calls", style="magenta", width=6, justify="right")
        table.add_column("Gen", style="yellow", width=8, justify="right")
        table.add_column("Call", style="white", width=7, justify="right")
        table.add_column("Speed", style="bold green", width=9, justify="right")
        table.add_column("Avg", style="bold blue", width=8, justify="right")
        table.add_column("TTFT", style="red", width=8, justify="right")
        table.add_column("TTFT sum", style="red", width=10, justify="right")
        table.add_column("Wall", style="dim", width=7, justify="right")
        table.add_column("Response", style="dim white", width=self.response_width)

        return table

    def _make_media_table(
        self,
        count_label: str = "Imgs",
        name_label: str = "Image",
    ) -> Table:
        """Create a Rich Table for media benchmarks (vision / audio).

        Adds two extra columns between Calls and Gen:
        a count column (e.g. Imgs / Audios) and a name column (e.g. Image / Audio).
        """
        table = Table(box=_rich_box.DOUBLE, show_header=True)

        table.add_column("W#", style="cyan", width=4, justify="right")
        table.add_column("Calls", style="magenta", width=6, justify="right")
        table.add_column(count_label, style="bold yellow", width=6, justify="right")
        table.add_column(name_label, style="green", width=14)
        table.add_column("Gen", style="yellow", width=8, justify="right")
        table.add_column("Call", style="white", width=7, justify="right")
        table.add_column("Speed", style="bold green", width=9, justify="right")
        table.add_column("Avg", style="bold blue", width=8, justify="right")
        table.add_column("TTFT", style="red", width=8, justify="right")
        table.add_column("TTFT sum", style="red", width=10, justify="right")
        table.add_column("Wall", style="dim", width=7, justify="right")
        table.add_column("Response", style="dim white", width=self.response_width)

        return table

    def _make_dual_media_table(self) -> Table:
        """Create a Rich Table for mixed image+video benchmarks.

        Shows both Imgs/Vid count columns and Image/Video name columns
        when the request may contain both types of media simultaneously.
        """
        table = Table(box=_rich_box.DOUBLE, show_header=True)

        table.add_column("W#", style="cyan", width=4, justify="right")
        table.add_column("Calls", style="magenta", width=6, justify="right")
        table.add_column("Imgs", style="bold yellow", width=6, justify="right")
        table.add_column("Image", style="green", width=14)
        table.add_column("Vid", style="bold yellow", width=5, justify="right")
        table.add_column("Video", style="green", width=14)
        table.add_column("Gen", style="yellow", width=8, justify="right")
        table.add_column("Call", style="white", width=7, justify="right")
        table.add_column("Speed", style="bold green", width=9, justify="right")
        table.add_column("Avg", style="bold blue", width=8, justify="right")
        table.add_column("TTFT", style="red", width=8, justify="right")
        table.add_column("TTFT sum", style="red", width=10, justify="right")
        table.add_column("Wall", style="dim", width=7, justify="right")
        table.add_column("Response", style="dim white", width=self.response_width)

        return table

    def _render_worker_row(
        self,
        table: Table,
        wid: int,
        w: dict[str, Any],
        *,
        media: bool = False,
        dual_media: bool = False,
    ) -> None:
        """Add a single worker row to *table*.

        Handles three states:
        - Error  (wid in self._errors)
        - Stopped (w["stopped"] is True)
        - Normal  (all metrics displayed)

        Set *media* = True when the table has the extra media columns
        (count + name) so the row cell count matches.
        Set *dual_media* = True for mixed image+video table (14 columns).
        """
        err = self._errors.get(wid)

        if err:
            if dual_media:
                table.add_row(
                    str(wid), str(w.get("calls", 0)),
                    "", "", "", "",
                    "ERROR", "", "", "", "", "", "",
                    "[red]SEE LOG[/]",
                )
            elif media:
                table.add_row(
                    str(wid),
                    str(w.get("calls", 0)),
                    "",
                    "",
                    "ERROR",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "[red]SEE LOG[/]",
                )
            else:
                table.add_row(
                    str(wid),
                    str(w.get("calls", 0)),
                    "ERROR",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "[red]SEE LOG[/]",
                )
            return

        if w.get("stopped"):
            error_text = self._clean_tail(w.get("error", "unknown"), self.response_width)
            if dual_media:
                table.add_row(
                    str(wid), str(w.get("calls", 0)),
                    "", "", "", "",
                    "STOPPED", "", "", "", "", "", "",
                    f"[red]{error_text}[/]",
                )
            elif media:
                table.add_row(
                    str(wid),
                    str(w.get("calls", 0)),
                    "",
                    "",
                    "STOPPED",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    f"[red]{error_text}[/]",
                )
            else:
                table.add_row(
                    str(wid),
                    str(w.get("calls", 0)),
                    "STOPPED",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    f"[red]{error_text}[/]",
                )
            return

        # Normal row
        gen_val = w.get("gen_est", w.get("gen", 0))
        tail = self._clean_tail(w.get("tail", ""), self.response_width)

        if dual_media:
            img_name = w.get("media_img", "")
            vid_name = w.get("media_vid", "")
            img_count = w.get("media_img_count", 0)
            vid_count = w.get("media_vid_count", 0)
            table.add_row(
                str(wid),
                str(w.get("calls", 0)),
                str(img_count),
                img_name,
                str(vid_count),
                vid_name,
                f"{gen_val:,}",
                f"{w.get('call_gen', 0):,}",
                f"{w.get('speed', 0):.1f} t/s",
                f"{w.get('avg', 0):.1f} t/s",
                f"{w.get('ttft', 0):.2f}s",
                f"{w.get('ttft_sum', 0):.1f}s",
                w.get("wall", ""),
                tail,
            )
        elif media:
            media_name = w.get("media", "")
            media_count = w.get("media_count", 0)
            table.add_row(
                str(wid),
                str(w.get("calls", 0)),
                str(media_count),
                media_name,
                f"{gen_val:,}",
                f"{w.get('call_gen', 0):,}",
                f"{w.get('speed', 0):.1f} t/s",
                f"{w.get('avg', 0):.1f} t/s",
                f"{w.get('ttft', 0):.2f}s",
                f"{w.get('ttft_sum', 0):.1f}s",
                w.get("wall", ""),
                tail,
            )
        else:
            table.add_row(
                str(wid),
                str(w.get("calls", 0)),
                f"{gen_val:,}",
                f"{w.get('call_gen', 0):,}",
                f"{w.get('speed', 0):.1f} t/s",
                f"{w.get('avg', 0):.1f} t/s",
                f"{w.get('ttft', 0):.2f}s",
                f"{w.get('ttft_sum', 0):.1f}s",
                w.get("wall", ""),
                tail,
            )

    # ---- rendering (abstract) ---------------------------------------------

    @abstractmethod
    def render(self) -> Table:
        """Render the full Rich Table for the current state.

        Subclasses must implement this and return a ``rich.table.Table``.
        """
        ...