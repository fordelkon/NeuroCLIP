from typing import TYPE_CHECKING

from lightning.pytorch.callbacks.progress.rich_progress import (
    BatchesProcessedColumn,
    CustomBarColumn,
    ProcessingSpeedColumn,
    RichProgressBar,
)
from rich.progress import TextColumn, TimeElapsedColumn, TimeRemainingColumn

if TYPE_CHECKING:
    from lightning.pytorch import Trainer


class AsciiRichProgressBar(RichProgressBar):
    """Rich progress bar variant that avoids Unicode-only time separators."""

    def configure_columns(self, trainer: "Trainer") -> list:
        return [
            TextColumn("[progress.description]{task.description}"),
            CustomBarColumn(
                complete_style=self.theme.progress_bar,
                finished_style=self.theme.progress_bar_finished,
                pulse_style=self.theme.progress_bar_pulse,
            ),
            BatchesProcessedColumn(style=self.theme.batch_progress),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            ProcessingSpeedColumn(style=self.theme.processing_speed),
        ]
