"""Contains a class for a movable line that allows selecting time points in raw data."""

import logging

import pyqtgraph as pg
from qtpy.QtCore import QObject, Signal, Slot

logger = logging.getLogger(__name__)


class RawTimeSelector(QObject):
    """Represents a vertical line slider that allows the user to select a time point.

    Provides getter and setter for the currently selected time in seconds and emits a
    signal carrying the new selected time whenever the user changes the selection
    by dragging the line.
    """

    sigSelectedTimeChanged = Signal(float)

    def __init__(
        self,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)

        self._selector = pg.InfiniteLine(
            pos=0, angle=90, movable=True, pen=pg.mkPen("r", width=3)
        )

        self._selector.sigPositionChanged.connect(
            lambda: self._signal_user_selected_time_change()
        )
        self.suppress_change_signal = False

    @Slot()
    def _signal_user_selected_time_change(self) -> None:
        """Emit the signal if the change was made by user interaction."""
        if not self.suppress_change_signal:
            self.sigSelectedTimeChanged.emit(self.get_selected_time())

    def get_selected_time(self) -> float:
        """Get the currently selected time in seconds."""
        time = self._selector.value()
        # Get rid of IDE type warnings by explicitly checking the type
        if not isinstance(time, float):
            raise TypeError(f"Expected selected time to be float, got {type(time)}")
        return time

    def set_selected_time_no_signal(self, time_seconds: float) -> None:
        """Set the position of the selector in seconds WITHOUT emitting a signal."""
        self.suppress_change_signal = True
        self._selector.setValue(time_seconds)
        self.suppress_change_signal = False

    def get_selector(self) -> pg.InfiniteLine:
        """Get the InfiniteLine selector."""
        return self._selector
