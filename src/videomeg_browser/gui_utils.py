"""Contains helpers and GUI components for video and audio browser."""

import logging
import math
from importlib.resources import files

from qtpy.QtCore import Qt, Signal  # type: ignore
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QWidget

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


class ElapsedTimeLabel(QLabel):
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
        super().__init__(parent=parent)
        if current_time_seconds < 0:
            raise ValueError("Current time cannot be negative.")
        if max_time_seconds < 0:
            raise ValueError("Maximum time cannot be negative.")
        if current_time_seconds > max_time_seconds:
            logger.warning("Current time exceeds maximum time.")
        self._current_time_seconds = current_time_seconds
        self._max_time_seconds = max_time_seconds

        # Determine whether to include hours in the time display.
        if max_time_seconds < 3600:
            self._include_hours = False
        else:
            self._include_hours = True

        self._current_time_text = self._format_time(current_time_seconds)
        self._max_time_text = self._format_time(max_time_seconds)
        self.setText(f"{self._current_time_text} / {self._max_time_text}")

    def set_current_time(self, current_time_seconds: float) -> None:
        """Update the current time displayed in the label."""
        if current_time_seconds < 0:
            raise ValueError("Current time cannot be negative.")
        if current_time_seconds > self._max_time_seconds and not math.isclose(
            current_time_seconds,
            self._max_time_seconds,
            abs_tol=1e-3,  # allow current time to be less than 1 ms greater than max
        ):
            logger.warning(
                f"Current time {current_time_seconds} exceeds maximum time "
                f"{self._max_time_seconds}."
            )

        self._current_time_seconds = current_time_seconds
        self._current_time_text = self._format_time(current_time_seconds)
        self.setText(f"{self._current_time_text} / {self._max_time_text}")

    def set_max_time(self, max_time_seconds: float) -> None:
        """Update the maximum time displayed in the label.

        Also updates the display format to include or exclude hours based on
        `max_time_seconds` being more or less than an hour.
        """
        if max_time_seconds < 0:
            raise ValueError("Maximum time cannot be negative.")
        if max_time_seconds < self._current_time_seconds:
            logger.warning("Maximum time is less than current time.")

        if max_time_seconds < 3600:
            self._include_hours = False
        else:
            self._include_hours = True

        self._max_time_seconds = max_time_seconds
        self._max_time_text = self._format_time(max_time_seconds)
        # Also update the current time text in case display format changed.
        self._current_time_text = self._format_time(self._current_time_seconds)
        self.setText(f"{self._current_time_text} / {self._max_time_text}")

    def _format_time(self, time_seconds: float) -> str:
        """Format seconds as mm:ss.mmm or hh:mm:ss.mmm, based on include_hours flag."""
        total_seconds = int(time_seconds)
        milliseconds = int((time_seconds - total_seconds) * 1000)

        minutes, seconds = divmod(total_seconds, 60)
        if self._include_hours:
            hours, minutes = divmod(minutes, 60)
            return f"{hours}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

        return f"{minutes}:{seconds:02d}.{milliseconds:03d}"


class IndexSlider(QWidget):
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
        self._layout = QHBoxLayout()
        # Remove margins so that the slider does not have extra space around it.
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        self._slider = QSlider(Qt.Horizontal, parent=self)
        self._layout.addWidget(self._slider)
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


class NavigationBar(QWidget):
    """A navigation bar with Previous, Play/Pause, and Next buttons.

    Emits signals when the buttons are clicked and provides methods to enable/disable
    the buttons. Handles toggling the Play/Pause button text when the button is clicked.

    Parameters
    ----------
    prev_button_text : str
        The text for the Previous button.
    next_button_text : str
        The text for the Next button.
    play_text : str, optional
        The text for the play/pause button when in play state, by default "Play".
    pause_text : str, optional
        The text for the play/pause button when in pause state, by default "Pause".
    button_min_width : int, optional
        The minimum width for the buttons, by default 100.
    parent : QWidget, optional
        The parent widget for this navigation bar, by default None
    """

    sigNextClicked = Signal()
    sigPreviousClicked = Signal()
    sigPlayPauseClicked = Signal()

    def __init__(
        self,
        prev_button_text: str,
        next_button_text: str,
        play_text: str = "Play",
        pause_text: str = "Pause",
        button_min_width: int = 100,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        # Save the play and pause texts for toggling later.
        self._play_text = play_text
        self._pause_text = pause_text

        self._layout = QHBoxLayout()
        # Remove margins so that the buttons do not have extra space around them.
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self._layout)

        self._prev_button = QPushButton(prev_button_text)
        self._prev_button.clicked.connect(lambda: self.sigPreviousClicked.emit())
        self._prev_button.setMinimumWidth(button_min_width)
        self._layout.addWidget(self._prev_button)

        self._play_pause_button = QPushButton(play_text)  # Assuming we start paused
        self._play_pause_button.clicked.connect(lambda: self.sigPlayPauseClicked.emit())
        self._play_pause_button.setMinimumWidth(button_min_width)
        self._layout.addWidget(self._play_pause_button)

        self._next_button = QPushButton(next_button_text)
        self._next_button.clicked.connect(lambda: self.sigNextClicked.emit())
        self._next_button.setMinimumWidth(button_min_width)
        self._layout.addWidget(self._next_button)

    def set_prev_enabled(self, enabled: bool) -> None:
        """Enable/disable the Previous button."""
        self._prev_button.setEnabled(enabled)

    def set_play_pause_enabled(self, enabled: bool) -> None:
        """Enable/disable the Play/Pause button."""
        self._play_pause_button.setEnabled(enabled)

    def set_next_enabled(self, enabled: bool) -> None:
        """Enable/disable the Next button."""
        self._next_button.setEnabled(enabled)

    def set_paused(self) -> None:
        """Set the button text to play text to indicate paused state."""
        self._play_pause_button.setText(self._play_text)

    def set_playing(self) -> None:
        """Set the button text to pause text to indicate playing state."""
        self._play_pause_button.setText(self._pause_text)
