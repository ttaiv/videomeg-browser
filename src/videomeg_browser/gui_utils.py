"""Contains helpers and GUI components for video and audio browser."""

import logging
from importlib.resources import files

from qtpy.QtCore import QObject, Qt, Signal  # type: ignore
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QLabel, QLayout, QSlider, QWidget

logger = logging.getLogger(__name__)


def load_icon_pixmap(file_name: str) -> QPixmap | None:
    """Load an icon pixmap from icons directory.

    Parameters
    ----------
    file_name : str
        The name of the icon file to load.
        This should be just the file name with file extension, e.g., "icon.png".

    Returns
    -------
    QPixmap | None
        The loaded QPixmap if successful, or None if the file was not found.
    """
    icons_module = "videomeg_browser.icons"
    pixmap = QPixmap()
    resource = files(icons_module).joinpath(file_name)
    try:
        with resource.open("rb") as file:
            pixmap.loadFromData(file.read())
        return pixmap
    except (FileNotFoundError, ModuleNotFoundError) as e:
        logger.warning(f"Icon '{file_name}' not found from '{icons_module}': {e}")
        return None


class ElapsedTimeLabel:
    """A label for displaying time in a format [current_time] / [max_time].

    The format is either mm:ss or hh:mm:ss, depending on whether the
    maximum time is less than or greater than one hour.

    Parameters
    ----------
    current_time_seconds : float
        The current time in seconds to display.
    max_time_seconds : float
        The maximum time in seconds to display.
    parent : QWidget, optional
        The parent widget for this label, by default None
    """

    def __init__(
        self,
        current_time_seconds: float,
        max_time_seconds: float,
        parent: QWidget | None = None,
    ) -> None:
        if current_time_seconds > max_time_seconds:
            logger.warning("Current time exceeds maximum time.")
        self._current_time_seconds = current_time_seconds
        self._max_time_seconds = max_time_seconds
        self._label = QLabel(parent=parent)

        # Determine whether to include hours in the time display.
        if max_time_seconds < 3600:
            self._include_hours = False
        else:
            self._include_hours = True

        self._current_time_text = self._format_time(current_time_seconds)
        self._max_time_text = self._format_time(max_time_seconds)
        self._label.setText(f"{self._current_time_text} / {self._max_time_text}")

    def set_current_time(self, current_time_seconds: float) -> None:
        """Update the current time displayed in the label."""
        if current_time_seconds > self._max_time_seconds:
            logger.warning("Current time exceeds maximum time.")
        self._current_time_text = self._format_time(current_time_seconds)
        self._label.setText(f"{self._current_time_text} / {self._max_time_text}")

    def set_max_time(self, max_time_seconds: float) -> None:
        """Update the maximum time displayed in the label.

        Also updates the display format to include or exclude hours based on
        `max_time_seconds` being more or less than an hour.
        """
        if max_time_seconds < self._current_time_seconds:
            logger.warning("Maximum time is less than current time.")
        if max_time_seconds < 3600:
            self._include_hours = False
        else:
            self._include_hours = True

        self._max_time_text = self._format_time(max_time_seconds)
        # Also update the current time text in case display format changed.
        self._current_time_text = self._format_time(self._current_time_seconds)
        self._label.setText(f"{self._current_time_text} / {self._max_time_text}")

    def set_current_and_max_time(
        self, current_time_seconds: float, max_time_seconds: float
    ) -> None:
        """Update both current and maximum time displayed in the label.

        Also updates the display format to include or exclude hours based on
        `max_time_seconds` being more or less than an hour.
        """
        if current_time_seconds > max_time_seconds:
            logger.warning("Current time exceeds maximum time.")
        self._current_time_seconds = current_time_seconds
        # This handles also updating the current time text.
        self.set_max_time(max_time_seconds)

    def add_to_layout(self, layout: QLayout) -> None:
        """Add the label to the given layout.

        Parameters
        ----------
        layout : QLayout
            The layout to which the label will be added.
        """
        layout.addWidget(self._label)

    def _format_time(self, time_seconds: float) -> str:
        """Format seconds as mm:ss or hh:mm:ss, depending on the include_hours flag."""
        minutes, seconds = divmod(int(time_seconds), 60)
        if self._include_hours:
            hours, minutes = divmod(minutes, 60)
            return f"{hours}:{minutes:02d}:{seconds:02d}"

        return f"{minutes}:{seconds:02d}"


class IndexSlider(QObject):
    """A slider for navigating indices, such as video frames.

    Emits a signal when the index changes and provides methods to manipulate the slider,
    optionally without emitting the signal.

    Parameters
    ----------
    min_value : int
        The minimum value of the slider.
    max_value : int
        The maximum value of the slider.
    value : int
        The initial value of the slider.
    parent : QWidget, optional
        The parent widget for this slider, by default None
    """

    sigIndexChanged = Signal(int)

    def __init__(
        self, min_value: int, max_value: int, value: int, parent: QWidget | None = None
    ) -> None:
        if max_value < min_value:
            raise ValueError("Maximum value must be greater than or equal to minimum.")
        if value < min_value or value > max_value:
            raise ValueError(
                f"Value must be between {min_value} and {max_value}, inclusive. "
                f"Got {value}."
            )
        self._min_value = min_value
        self._max_value = max_value

        super().__init__(parent=parent)
        self._slider = QSlider(Qt.Horizontal, parent=parent)

        self._slider.setMinimum(min_value)
        self._slider.setMaximum(max_value)
        self._slider.setValue(value)

        self._slider.valueChanged.connect(
            lambda value: self.sigIndexChanged.emit(value)
        )

    def set_max_value(self, max_value: int, signal: bool) -> None:
        """Set the maximum value of the slider.

        Parameters
        ----------
        max_value : int
            The maximum value to set for the slider.
        signal : bool
            Whether to emit the `sigIndexChanged` if the value of the slider changes.
        """
        if max_value < self._min_value:
            raise ValueError(
                f"Maximum value must be greater than or equal to minimum value "
                f"{self._min_value}. Got {max_value}."
            )
        self._max_value = max_value
        if signal:
            self._slider.setMaximum(max_value)
        else:
            self._slider.blockSignals(True)
            self._slider.setMaximum(max_value)
            self._slider.blockSignals(False)

    def set_value(self, value: int, signal: bool) -> None:
        """Set the slider value and optionally emit the valueChanged signal.

        Parameters
        ----------
        value : int
            The value to set for the slider.
        signal : bool, optional
            Whether to emit the `sigIndexChanged` signal if the value of the slider
            changes.
        """
        if value < self._min_value or value > self._max_value:
            raise ValueError(
                f"Value must be between {self._min_value} and {self._max_value}, "
                f"inclusive. Got {value}."
            )
        if signal:
            self._slider.setValue(value)
        else:
            self._slider.blockSignals(True)
            self._slider.setValue(value)
            self._slider.blockSignals(False)

    def add_to_layout(self, layout: QLayout) -> None:
        """Add the slider to the given layout.

        Parameters
        ----------
        layout : QLayout
            The layout to which the slider will be added.
        """
        layout.addWidget(self._slider)
