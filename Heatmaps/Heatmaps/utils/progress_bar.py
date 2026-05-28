"""
Reusable terminal progress bar.
"""

import sys
import time
from dataclasses import dataclass
from typing import TextIO


@dataclass
class ProgressBar:
    """Render a single-line terminal progress bar."""

    total: int
    label: str = 'Progress'
    width: int = 40
    output_stream: TextIO = sys.stdout
    enabled: bool = True
    status_max_length: int = 45

    def __post_init__(self):
        self.total = int(self.total)
        self.safe_total = max(self.total, 1)
        self.width = max(int(self.width), 1)
        self.current = 0
        self.status = ''
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        self.render()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            self.output_stream.write('\n')
            self.output_stream.flush()

    def update(self, increment=1, status=None):
        """Advance the bar and optionally update the status."""
        if status is not None:
            self.status = str(status)

        self.current = min(self.current + int(increment), self.safe_total)
        self.render()

    def set_status(self, status):
        """Update the status without advancing the bar."""
        self.status = str(status)
        self.render()

    def render(self):
        """Write the current progress state."""
        if not self.enabled:
            return

        fraction = min(self.current / self.safe_total, 1.0)
        filled = int(self.width * fraction)
        bar = '#' * filled + '-' * (self.width - filled)
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        rate = self.current / elapsed if elapsed > 0 else 0.0
        remaining = (self.safe_total - self.current) / rate if rate > 0 else 0.0
        status_text = f' current={self.truncate_text(self.status, self.status_max_length)}' if self.status else ''
        message = f"\r\t{self.label}: [{bar}] {self.current}/{self.total} ({fraction * 100:6.2f}%) elapsed={self.format_seconds(elapsed)} eta={self.format_seconds(remaining)}{status_text}\033[K"
        self.output_stream.write(message)
        self.output_stream.flush()

    @staticmethod
    def format_seconds(seconds):
        """Format seconds as HH:MM:SS."""
        seconds = int(seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    @staticmethod
    def truncate_text(value, max_length):
        """Shorten text for display."""
        value = str(value)
        return value if len(value) <= max_length else f'{value[:max_length - 3]}...'
