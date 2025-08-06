"""Contains a class for a movable line that allows selecting time points in raw data."""

import logging

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QObject, Signal, Slot  # type: ignore

logger = logging.getLogger(__name__)


class TimeSelector(QObject):
    """Vertical line slider that allows the user to select a time point from a plot.

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
        # Flag to suppress the signal when setting the position programmatically.
        self.suppress_change_signal = False

    @property
    def selected_time(self) -> float:
        """Get the currently selected time in seconds."""
        time = self._selector.value()
        if not isinstance(time, int | float):
            raise TypeError(
                "The value of vertical line selector is not a single number. "
                f"Got {type(time)} instead."
            )

        return float(time)

    @property
    def selector(self) -> pg.InfiniteLine:
        """Get the InfiniteLine selector."""
        return self._selector

    def set_selected_time_no_signal(self, time_seconds: float) -> None:
        """Set the position of the selector in seconds WITHOUT emitting a signal."""
        self.suppress_change_signal = True
        self._selector.setValue(time_seconds)
        self.suppress_change_signal = False

    def clamp_selected_time_to_range(
        self, time_range: tuple[float, float], padding: float = 0.0
    ) -> None:
        """Clamp the selected time to a given range, optionally applying padding.

        Does not emit a signal even if the value of the selector changes.

        Parameters
        ----------
        time_range : tuple[float, float]
            The range to clamp the selected time to, as (min_time, max_time).
        padding : float, optional
            Padding to apply to the range, default is 0.0 seconds.
        """
        current_time = self.selected_time
        clamped_value = np.clip(
            current_time, time_range[0] + padding, time_range[1] - padding
        )

        if clamped_value != current_time:
            self.set_selected_time_no_signal(clamped_value)

    def add_to_plot(self, plot_widget: pg.PlotWidget) -> None:
        """Add the selector to a given plot widget."""
        plot_widget.addItem(self._selector)

    @Slot()
    def _signal_user_selected_time_change(self) -> None:
        """Emit the signal if the change was made by user interaction."""
        if not self.suppress_change_signal:
            self.sigSelectedTimeChanged.emit(self.selected_time)
