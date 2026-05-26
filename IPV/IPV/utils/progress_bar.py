"""
Reusable terminal progress bar.
"""

import sys
import time
from dataclasses import dataclass
from typing import TextIO

DEMO_PROGRESS_BAR = False
DEMO_TOTAL_STEPS = 50
DEMO_SLEEP_SECONDS = 0.05


@dataclass
class ProgressBar:
    """Render a single-line terminal progress bar with optional status text."""

    total: int
    label: str = "Progress"
    width: int = 40
    output_stream: TextIO = sys.stdout
    enabled: bool = True
    status_max_length: int = 45

    def __post_init__(self) -> None:
        self.total = int(self.total)
        self.safe_total = max(self.total, 1)
        self.width = max(int(self.width), 1)
        self.current = 0
        self.status = ""
        self.start_time: float | None = None

    def __enter__(self) -> "ProgressBar":
        self.start_time = time.time()
        self.render()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.enabled:
            self.output_stream.write("\n")
            self.output_stream.flush()

    def update(self, increment: int = 1, status: str | None = None) -> None:
        """Advance the bar and optionally update the status text."""
        if status is not None:
            self.status = str(status)

        self.current = min(self.current + int(increment), self.safe_total)
        self.render()

    def set_status(self, status: str) -> None:
        """Update the status text without advancing the bar."""
        self.status = str(status)
        self.render()

    def render(self) -> None:
        """Write the current progress state to the output stream."""
        if not self.enabled:
            return

        fraction = min(self.current / self.safe_total, 1.0)
        filled = int(self.width * fraction)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        rate = self.current / elapsed if elapsed > 0 else 0.0
        remaining = (self.safe_total - self.current) / rate if rate > 0 else 0.0
        status_text = f" current={self.truncate_text(self.status, self.status_max_length)}" if self.status else ""

        message = (
            f"\r\t{self.label}: "
            f"[{bar}] "
            f"{self.current}/{self.total} "
            f"({fraction * 100:6.2f}%) "
            f"elapsed={self.format_seconds(elapsed)} "
            f"eta={self.format_seconds(remaining)}"
            f"{status_text}\033[K"
        )

        self.output_stream.write(message)
        self.output_stream.flush()

    @staticmethod
    def format_seconds(seconds: float) -> str:
        """Format seconds as HH:MM:SS."""
        seconds = int(seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def truncate_text(value: str, max_length: int) -> str:
        """Shorten text for display in the progress line."""
        value = str(value)

        if len(value) <= max_length:
            return value

        return f"{value[:max_length - 3]}..."


def run_demo() -> None:
    """Run a small local demo of the progress bar."""
    with ProgressBar(total=DEMO_TOTAL_STEPS, label="Demo") as progress_bar:
        for index in range(DEMO_TOTAL_STEPS):
            time.sleep(DEMO_SLEEP_SECONDS)
            progress_bar.update(status=f"step_{index + 1}")


def main() -> None:
    """Run the demo when enabled."""
    if DEMO_PROGRESS_BAR:
        run_demo()


if __name__ == "__main__":
    main()
