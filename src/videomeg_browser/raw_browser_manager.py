import logging

import numpy as np
from mne_qt_browser._pg_figure import MNEQtBrowser
from qtpy.QtCore import QObject, Signal, Slot

from .time_selector import RawTimeSelector

logger = logging.getLogger(__name__)


class RawBrowserInterface(QObject):
    """Interface for the MNE Qt raw browser that the manager depends on.

    Provides methods for getting and setting the time range (x-axis) of the raw data
    browser and for adding an item to the plot. Emits a signal when visible time
    range of the browser changes.

    NOTE: The correct way to modify view of the raw browser from outside should
    be considered.
    """

    sigTimeRangeChanged = Signal(tuple)

    def __init__(self, raw_browser: MNEQtBrowser, parent=None):
        super().__init__(parent=parent)
        self.browser = raw_browser
        self.plt = self.browser.mne.plt

        # Connect the signal for time range changes to emit a custom signal
        self.plt.sigXRangeChanged.connect(
            lambda _, xrange: self.sigTimeRangeChanged.emit(xrange)
        )

    def get_max_time(self) -> float:
        """Return the maximum time in the raw data browser."""
        return self.browser.mne.xmax

    def get_view_time_range(self) -> tuple[float, float]:
        """Return the bounds of currently visible time axis."""
        return self.plt.getViewBox().viewRange()[0]

    def set_view_time_range(
        self, min_time_seconds: float, max_time_seconds: float, padding: float = 0
    ):
        """Set the bounds for currently visible time axis."""
        self.plt.setXRange(min_time_seconds, max_time_seconds, padding=padding)

    def get_visible_duration(self) -> float:
        """Return the duration of the currently visible time range in seconds."""
        return self.browser.mne.duration

    def add_item_to_plot(self, item):
        """Add an item to the plot. Calls the addItem method for browser.mne.plt."""
        self.plt.addItem(item)

    def show(self):
        """Show the raw data browser."""
        self.browser.show()


class RawBrowserManager(QObject):
    """Manager for raw browser instance tailored for time syncing with video.

    Provides methods for manipulating the view and adds a 'time selector'
    (vertical line) that marks the time point used for syncing with video.
    Emits signal when the selected time is changed.
    """

    # Carries the currently selected time in seconds
    sigSelectedTimeChanged = Signal(float)

    def __init__(self, raw_browser: RawBrowserInterface, parent=None):
        super().__init__(parent=parent)
        self.browser = raw_browser
        # Padding to apply to user selected time selector values when clamping
        # them to the current view range of the raw data browser.
        self.selector_padding = 0.1

        # Bounds of the raw data browser's view in seconds
        self.raw_min_time = 0
        self.raw_max_time = self.browser.get_max_time()

        # Flag to prevent obsolete updates when time range is changed programmatically
        self.programmatic_time_range_change = False

        # Default relative position of the time selector in the raw data
        # browser's view. This will not be obeyed in the boundaries of raw data.
        # User can modify this by dragging the time selector.
        self.time_selector_fraction = 0.5
        self.raw_time_selector = RawTimeSelector(parent=self)
        self.browser.add_item_to_plot(self.raw_time_selector.get_selector())

        # The selected time of the raw browser can change in two ways:
        # 1. When user modifies the raw browser view (e.g., zooms in/out or scrolls)
        # 2. When user drags the time selector
        # Connect the signals for both cases to handle them appropriately.

        # User modifies the raw browser view
        # --> move time selector to keep it in the same relative position
        # (excluding boundaries)
        self.browser.sigTimeRangeChanged.connect(self._handle_time_range_change)

        # User drags the time selector
        # --> update the value of it but keep the view unchanged
        self.raw_time_selector.sigSelectedTimeChanged.connect(
            self._handle_time_selector_change
        )

        # Initialize the time selector position
        self._handle_time_range_change(raw_browser.get_view_time_range())

    def jump_to_start(self):
        """Set browser's view and time selector to the beginning of the data."""
        max_time = self.raw_min_time + self.browser.get_visible_duration()
        logger.debug(
            f"Setting raw view to range [{self.raw_min_time:.3f}, {max_time:.3f}] "
            "seconds at the start of the data."
        )
        self.browser.set_view_time_range(self.raw_min_time, max_time)
        self.raw_time_selector.set_selected_time_no_signal(self.raw_min_time)

    def jump_to_end(self):
        """Set browser's view and time selector to the end of the data."""
        min_time = self.raw_max_time - self.browser.get_visible_duration()
        logger.debug(
            f"Setting raw view to range [{min_time:.3f}, {self.raw_max_time:.3f}] "
            "seconds at the end of the data."
        )
        self.browser.set_view_time_range(min_time, self.raw_max_time)
        self.raw_time_selector.set_selected_time_no_signal(self.raw_max_time)

    def set_selected_time(self, time_seconds: float):
        """Set the raw time selector to a specific time point in seconds.

        This will also update the view of the raw data browser accordingly.
        """
        logger.debug(f"Setting raw time selector to {time_seconds:.3f} seconds.")
        self.raw_time_selector.set_selected_time_no_signal(time_seconds)
        self._update_view_based_on_time_selector()

    def get_selected_time(self) -> float:
        """Get the current position of the raw time selector in seconds."""
        return self.raw_time_selector.get_selected_time()

    def show_browser(self):
        """Show the raw data browser."""
        self.browser.show()

    @Slot()
    def _handle_time_selector_change(self):
        """Update the default position and emit signal when user drags time selector."""
        # Clamp the raw time selector to the current view range
        # (for some reason it is possible to drag it outside the view range)
        clamped_time = self._clamp_time_selector_to_current_view(
            self.raw_time_selector.get_selected_time(), padding=self.selector_padding
        )
        self.raw_time_selector.set_selected_time_no_signal(clamped_time)
        logger.debug(
            "Detected change in raw time selector, setting new default position."
        )
        self._update_default_time_selector_position(clamped_time)
        self.sigSelectedTimeChanged.emit(clamped_time)

    @Slot(tuple)
    def _handle_time_range_change(self, new_xrange: tuple[float, float]):
        """Update raw time selector value and emit the new value with a signal.

        Updates the raw time selector value so that it remains at the same
        relative position in the raw data browser's view (excluding boundaries)

        Parameters
        ----------
        new_raw_xrange : tuple[float, float]
            The new view range of the raw data browser, given as (xmin, xmax).
        """
        if self.programmatic_time_range_change:
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
        self.sigSelectedTimeChanged.emit(raw_time_seconds)

    def _clamp_time_selector_to_current_view(
        self, new_value: float, padding: float
    ) -> float:
        """Set raw time selector value, clamped to current raw view range.

        Used to ensure that user cannot drag the time selector outside
        the current view range of the raw data browser.

        Parameters
        ----------
        new_value : float
            The value to set the raw time selector to, in seconds.
        padding : float
            Padding to apply to the view range when clamping the value.
            This is useful to ensure that the time selector does not
            get too close to the edges of the view range.

        Returns
        -------
        float
            The clamped value of the raw time selector, in seconds.
        """
        # Get the current view range of the raw data browser
        min_time, max_time = self.browser.get_view_time_range()
        # Clamp the new value to the current view range
        clamped_value = np.clip(new_value, min_time + padding, max_time - padding)

        return clamped_value

    def _update_default_time_selector_position(self, new_selector_value: float):
        """Update the default position of the time selector based on current view."""
        # Update the time selector fraction based on the new raw time selector value
        min_time, max_time = self.browser.get_view_time_range()
        window_len = max_time - min_time

        new_selector_fraction = (new_selector_value - min_time) / window_len
        logger.debug(
            f"Updating time selector fraction to {new_selector_fraction:.3f} "
            f"based on raw time selector value {new_selector_value:.3f} seconds."
        )
        self.time_selector_fraction = new_selector_fraction

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
        selector_time = min_time + (max_time - min_time) * self.time_selector_fraction
        logger.debug(f"Setting raw time point selector to {selector_time:.3f} seconds.")
        self.raw_time_selector.set_selected_time_no_signal(selector_time)

        return selector_time

    def _update_view_based_on_time_selector(self):
        """Set raw view based on the raw time selector.

        The raw time selector will stay at the same relative position in the view,
        expect when the view is at the boundaries of the raw data.
        """
        window_len = self.browser.get_visible_duration()

        time_selector_pos = self.raw_time_selector.get_selected_time()
        logger.debug(
            f"Time selector position for raw view updating: {time_selector_pos:.3f} "
            "seconds."
        )

        # Calculate new xmin and xmax for the raw data browser's view
        min_time = time_selector_pos - window_len * self.time_selector_fraction
        max_time = time_selector_pos + window_len * (1 - self.time_selector_fraction)

        # Prevent the update from re-updating time selector position and emitting
        # a signal.
        self.programmatic_time_range_change = True

        if min_time < self.raw_min_time:
            logger.debug(
                f"Raw view xmin {min_time:.3f} is less than the minimum view time "
                f"{self.raw_min_time:.3f}. Setting view to range "
                f"[{self.raw_min_time:.3f}, {self.raw_min_time + window_len}] seconds."
            )
            self.browser.set_view_time_range(
                self.raw_min_time, self.raw_min_time + window_len
            )
        elif max_time > self.raw_max_time:
            logger.debug(
                f"Raw view xmax {max_time:.3f} is greater than the maximum view time "
                f"{self.raw_max_time:.3f}. Setting view to range "
                f"[{self.raw_max_time - window_len:.3f}, {self.raw_max_time:.3f}] "
                "seconds."
            )
            self.browser.set_view_time_range(
                self.raw_max_time - window_len, self.raw_max_time
            )
        else:
            logger.debug(
                f"Setting raw view to show video marker at {time_selector_pos:.3f} "
                f"seconds with range [{min_time:.3f}, {max_time:.3f}] seconds."
            )
            self.browser.set_view_time_range(min_time, max_time)

        self.programmatic_time_range_change = False
