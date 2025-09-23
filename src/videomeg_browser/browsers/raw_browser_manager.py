"""Contains a class for managing the MNE Qt raw data browser in sync with video data."""

import logging

import mne
import numpy as np
from mne_qt_browser._pg_figure import MNEQtBrowser
from qtpy.QtCore import QObject, Signal, Slot  # type: ignore

from .syncable_browser import SyncableBrowserObject
from .time_selector import TimeSelector

logger = logging.getLogger(__name__)


class RawBrowserInterface(QObject):
    """Interface for the MNE Qt raw browser that the manager depends on.

    Provides methods for getting and setting the time range (x-axis) of the raw data
    browser and for adding an item to the plot. Emits a signal when visible time
    range of the browser changes.

    NOTE: Some attribute accesses are marked with `# type: ignore`, because my IDE
    does not understand that the attributes exist.
    """

    sigTimeRangeChanged = Signal(tuple)

    def __init__(
        self, raw_browser: MNEQtBrowser, parent: QObject | None = None
    ) -> None:
        super().__init__(parent=parent)
        self.browser = raw_browser
        self.plt = self.browser.mne.plt  # type: ignore

        # Connect the signal for time range changes to emit a custom signal
        self.plt.sigXRangeChanged.connect(
            lambda _, xrange: self.sigTimeRangeChanged.emit(xrange)
        )

    def get_max_time(self) -> float:
        """Return the maximum time in the raw data browser."""
        return self.browser.mne.xmax  # type: ignore

    def get_view_time_range(self) -> tuple[float, float]:
        """Return the bounds of currently visible time axis."""
        return self.plt.getViewBox().viewRange()[0]

    def set_view_time_range(
        self, min_time_seconds: float, max_time_seconds: float, padding: float = 0
    ) -> None:
        """Set the bounds for currently visible time axis."""
        self.plt.setXRange(min_time_seconds, max_time_seconds, padding=padding)

    def get_visible_duration(self) -> float:
        """Return the duration of the currently visible time range in seconds."""
        return self.browser.mne.duration  # type: ignore

    def add_item_to_plot(self, item) -> None:
        """Add an item to the plot. Calls the addItem method for browser.mne.plt."""
        self.plt.addItem(item)

    def show(self) -> None:
        """Show the raw data browser."""
        self.browser.show()

    def hide(self) -> None:
        """Hide the raw data browser."""
        self.browser.hide()


class RawBrowserManager(SyncableBrowserObject):
    """Manager for raw browser instance tailored for time syncing with video.

    Provides methods for manipulating the view and adds a 'time selector'
    (vertical line) that marks the time point used for syncing with video.
    Emits signal when the selected time is changed.

    Parameters
    ----------
    raw_browser : RawBrowserInterface
        The raw browser to manage wrapped in RawBrowserInterface.
    raw : mne.io.Raw
        The raw data object being displayed in the browser.
    selector_padding : float, optional
        Padding (in seconds) to apply when clamping the time selector to the
        current view range of the raw data browser, by default 0.1
    default_selector_position : float, optional
        The default relative position of the time selector in the raw data
        browser's view, given as a fraction between 0 and 1, by default 0.5
    parent : QObject, optional
        The parent QObject for this manager, by default None
    """

    def __init__(
        self,
        raw_browser: RawBrowserInterface,
        raw: mne.io.Raw,
        selector_padding=0.1,
        default_selector_position=0.5,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._browser = raw_browser
        self._raw = raw
        self._selector_padding = selector_padding
        # User can modify this by dragging the time selector.
        self._time_selector_fraction = default_selector_position

        # Bounds of the raw data browser's view in seconds
        self._raw_min_time = 0
        self._raw_max_time = self._browser.get_max_time()

        # Flag to prevent obsolete updates when time range is changed programmatically
        self._programmatic_time_range_change = False

        self._raw_time_selector = TimeSelector(parent=self)
        self._browser.add_item_to_plot(self._raw_time_selector.selector)

        # The selected time of the raw browser can change in two ways:
        # 1. When user modifies the raw browser view (e.g., zooms in/out or scrolls)
        # 2. When user drags the time selector
        # Connect the signals for both cases to handle them appropriately.

        # User modifies the raw browser view
        # --> move time selector to keep it in the same relative position
        # (excluding boundaries)
        self._browser.sigTimeRangeChanged.connect(self._handle_time_range_change)

        # User drags the time selector
        # --> update the value of it but keep the view unchanged
        self._raw_time_selector.sigSelectedTimeChanged.connect(
            self._handle_time_selector_change
        )

        # Initialize the time selector position
        self._handle_time_range_change(raw_browser.get_view_time_range())

    def set_position(
        self, position_idx: int, media_idx: int, signal: bool = True
    ) -> bool:
        """Set the current position for the raw data browser.

        Parameters
        ----------
        position_idx : int
            The sample index in the raw data to set the position to.
        media_idx : int
            Ignored.
        signal : bool, optional
            Whether to emit sigPositionChanged signal, by default True.

        Returns
        -------
        bool
            True if the position was set successfully, False if the position index
            is out of bounds.
        """
        self._warn_about_media_idx(media_idx)
        # Convert position index to time in seconds.
        raw_time_seconds = self._raw.times[position_idx]
        self._set_selected_time_no_signal(raw_time_seconds)

        if signal:
            self.sigPositionChanged.emit(0, position_idx)  # emit zero as media_idx

        return True

    def jump_to_start(self, media_idx: int, signal: bool = True) -> None:
        """Set browser's view and time selector to the beginning of the data."""
        self._warn_about_media_idx(media_idx)
        max_time = self._raw_min_time + self._browser.get_visible_duration()
        logger.debug(
            f"Setting raw view to range [{self._raw_min_time:.3f}, {max_time:.3f}] "
            "seconds at the start of the data."
        )
        self._browser.set_view_time_range(self._raw_min_time, max_time)
        self._raw_time_selector.set_selected_time_no_signal(self._raw_min_time)

        if signal:
            self.sigPositionChanged.emit(0, 0)  # emit zero as media_idx

    def jump_to_end(self, media_idx: int, signal: bool = True) -> None:
        """Set browser's view and time selector to the end of the data."""
        self._warn_about_media_idx(media_idx)
        min_time = self._raw_max_time - self._browser.get_visible_duration()
        logger.debug(
            f"Setting raw view to range [{min_time:.3f}, {self._raw_max_time:.3f}] "
            "seconds at the end of the data."
        )
        self._browser.set_view_time_range(min_time, self._raw_max_time)
        self._raw_time_selector.set_selected_time_no_signal(self._raw_max_time)

        if signal:
            # Emit zero as media_idx and last index as position_idx
            self.sigPositionChanged.emit(0, self._raw.n_times - 1)

    def get_current_position(self, media_idx: int) -> int:
        """Get the current position in the raw data as a sample index."""
        self._warn_about_media_idx(media_idx)
        raw_time_seconds = self._raw_time_selector.selected_time
        return int(self._raw.time_as_index(raw_time_seconds, use_rounding=True)[0])

    def show_browser(self) -> None:
        """Show the raw data browser."""
        self._browser.show()

    def hide_browser(self) -> None:
        """Hide the raw data browser."""
        self._browser.hide()

    def _set_selected_time_no_signal(self, time_seconds: float) -> None:
        """Set the raw time selector to a specific time point (in seconds).

        Also moves the raw data browser's view if it is required to keep the
        time selector in the visible range.
        Does NOT emit a signal for the selected time change.
        """
        logger.debug(f"Setting raw time selector to {time_seconds:.3f} seconds.")
        self._raw_time_selector.set_selected_time_no_signal(time_seconds)
        self._move_view_to_time_selector()

    @Slot()
    def _handle_time_selector_change(self) -> None:
        """Update the default position and emit signal when user drags time selector."""
        # Clamp the raw time selector to the current view range so that user cannot drag
        # it outside the visible range of the raw data browser.
        self._raw_time_selector.clamp_selected_time_to_range(
            self._browser.get_view_time_range(), padding=self._selector_padding
        )
        clamped_time = self._raw_time_selector.selected_time
        logger.debug(
            "Detected change in raw time selector, setting new default position."
        )
        self._update_default_time_selector_position(clamped_time)
        self.sigPositionChanged.emit(0, self.get_current_position(media_idx=0))

    @Slot(tuple)
    def _handle_time_range_change(self, new_xrange: tuple[float, float]) -> None:
        """Update raw time selector value and emit the new value with a signal.

        Updates the raw time selector value so that it remains at the same
        relative position in the raw data browser's view (excluding boundaries)

        Parameters
        ----------
        new_raw_xrange : tuple[float, float]
            The new view range of the raw data browser, given as (xmin, xmax).
        """
        if self._programmatic_time_range_change:
            logger.debug(
                "Ignoring time range change signal due to programmatic update."
            )
            return
        logger.debug(
            f"Detected change in raw view range: {new_xrange[0]:.3f} to "
            f"{new_xrange[1]:.3f} seconds. Updating time selector value."
        )
        raw_time_seconds = self._update_time_selector_based_on_view(new_xrange)
        logger.debug(
            f"New raw time selector value set to {raw_time_seconds:.3f} seconds."
        )
        logger.debug("Emitting signal for selected time change in raw data browser.")
        self.sigPositionChanged.emit(0, self.get_current_position(media_idx=0))

    def _update_default_time_selector_position(self, new_selector_value: float) -> None:
        """Update the default position of the time selector based on current view."""
        # Update the time selector fraction based on the new raw time selector value
        min_time, max_time = self._browser.get_view_time_range()
        window_len = max_time - min_time

        new_selector_fraction = (new_selector_value - min_time) / window_len
        logger.debug(
            f"Updating time selector fraction to {new_selector_fraction:.3f} "
            f"based on raw time selector value {new_selector_value:.3f} seconds."
        )
        self._time_selector_fraction = new_selector_fraction

    def _update_time_selector_based_on_view(
        self, new_raw_time_range: tuple[float, float]
    ) -> float:
        """Update time point selector's value using raw view and time selector fraction.

        This changes the value of the selector so that it remains at the same
        relative position in the raw data browser's view.

        Parameters
        ----------
        raw_xrange : tuple[float, float]
            The new view range of the raw data browser, given as (xmin, xmax).

        Returns
        -------
        float
            The new position of the time point selector in seconds
        """
        # Get the current view range of the raw data browser
        min_time = new_raw_time_range[0]
        max_time = new_raw_time_range[1]

        # Calculate the new position of the time point selector
        selector_time = min_time + (max_time - min_time) * self._time_selector_fraction
        logger.debug(f"Setting raw time point selector to {selector_time:.3f} seconds.")
        self._raw_time_selector.set_selected_time_no_signal(selector_time)

        return selector_time

    def _move_view_to_time_selector(self) -> None:
        """Ensure that the raw data browser's view contains the time selector.

        If the time selector is outside the current view range, move the view
        as many window lengths as needed to bring the time selector into view.
        """
        selected_time = self._raw_time_selector.selected_time
        window_min, window_max = self._browser.get_view_time_range()
        window_len = window_max - window_min

        if window_min <= selected_time <= window_max:
            # The time selector is already in the view range, no need to change it.
            logger.debug(
                f"Time selector {selected_time:.3f} seconds is already in the view "
                f"range [{window_min:.3f}, {window_max:.3f}] seconds. No change needed."
            )
            return

        if selected_time < window_min:
            moves_needed = int(np.ceil((window_min - selected_time) / window_len))
            new_window_min = window_min - moves_needed * window_len
            new_window_max = window_max - moves_needed * window_len
        else:  # selected_time > window_max
            moves_needed = int(np.ceil((selected_time - window_max) / window_len))
            new_window_min = window_min + moves_needed * window_len
            new_window_max = window_max + moves_needed * window_len

        logger.debug(
            f"Moving raw view to range [{new_window_min:.3f}, {new_window_max:.3f}] "
            f"seconds to include time selector {selected_time:.3f} seconds."
        )
        self._programmatic_time_range_change = True
        try:
            self._browser.set_view_time_range(new_window_min, new_window_max)
        finally:
            # Ensure that the flag is reset even if an exception occurs
            # during setting the view range.
            self._programmatic_time_range_change = False

    def _warn_about_media_idx(self, media_idx: int) -> None:
        """Log a warning if media_idx is not zero."""
        if media_idx != 0:
            logger.warning(
                f"RawBrowserManager does not support multiple media, but it was asked "
                f"set or get position for media index {media_idx}."
            )
