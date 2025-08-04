"""Contains helper for video and audio browser GUI."""

import logging
from importlib.resources import files

from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QLabel, QLayout, QWidget

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
