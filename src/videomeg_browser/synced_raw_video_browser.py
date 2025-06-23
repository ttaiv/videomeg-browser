"""Contains the main class for synchronizing MNE raw data browser and video browser."""

import logging

from mne_qt_browser.figure import MNEQtBrowser
from qtpy import QtWidgets
from qtpy.QtCore import QElapsedTimer, QObject, Qt, QTimer, Signal, Slot

from .raw_browser_manager import RawBrowserInterface, RawBrowserManager
from .time_index_mapper import (
    MapFailureReason,
    MappingFailure,
    MappingSuccess,
    TimeIndexMapper,
)
from .video import VideoFile
from .video_browser import SyncStatus, VideoBrowser

logger = logging.getLogger(__name__)


class SyncedRawVideoBrowser(QObject):
    """Instantiates MNE raw data browser and video browser, and synchronizes them.

    Parameters
    ----------
    raw_browser : mne_qt_browser.figure.MNEQtBrowser
        The MNE raw data browser object to be synchronized with the video browser.
        This can be created with 'plot' method of MNE raw data object when using qt
        backend.
    video_file : VideoFile
        The video file object to be displayed in the video browser.
    time_mapper : TimeIndexMapper
        Mapper that provides the mapping between raw data time points and video
        frame indices.
    show : bool, optional
        Whether to show the raw data browser immediately upon instantiation,
        by default True.
    raw_update_max_fps : int, optional
        The maximum frames per second for updating the raw data browser view,
        by default 10. This has effect on the performance of the browser.
    """

    def __init__(
        self,
        raw_browser: MNEQtBrowser,
        video_file: VideoFile,
        time_mapper: TimeIndexMapper,
        show: bool = True,
        raw_update_max_fps: int = 10,
    ) -> None:
        super().__init__()
        self.video_file = video_file
        self.time_mapper = time_mapper
        # Flag to prevent infinite recursion during synchronization
        self._syncing = False

        self.raw_update_max_fps = raw_update_max_fps
        self.min_raw_update_interval_ms = int(1000 / self.raw_update_max_fps)
        self.raw_update_throttler = BufferedThrottler(
            self.min_raw_update_interval_ms, parent=self
        )

        # Wrap the raw browser to a class that exposes the necessary methods.
        raw_browser_interface = RawBrowserInterface(raw_browser)
        # Pass interface for manager that contains actual logic for managing the browser
        # in sync with the video browser.
        self.raw_browser_manager = RawBrowserManager(raw_browser_interface)
        # Make sure that raw browse visibility matches the `show` parameter.
        if show:
            self.raw_browser_manager.show_browser()
        else:
            self.raw_browser_manager.hide_browser()

        # Set up the video browser.
        self.video_browser = VideoBrowser(video_file, show_sync_status=True)

        # Dock the video browser to the raw data browser with Qt magic
        self.dock = QtWidgets.QDockWidget("Video Browser", raw_browser)
        self.dock.setWidget(self.video_browser)
        self.dock.setFloating(True)
        raw_browser.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.resize(1000, 800)  # Set initial size of the video browser
        if not show:
            self.dock.hide()

        # Set up synchronization

        # When video browser frame changes, update the raw data browser's view.
        # Connect the signal through a throttler to limit the update rate
        # of the raw data browser to `raw_update_max_fps`.
        self.video_browser.sigFrameChanged.connect(self.raw_update_throttler.trigger)
        self.raw_update_throttler.triggered.connect(self.sync_raw_to_video)

        # When either raw time selector value or raw data browser's view changes,
        # update the video browser
        self.raw_browser_manager.sigSelectedTimeChanged.connect(self.sync_video_to_raw)

        # Consider raw data browser to be the main browser and start by
        # synchronizing the video browser to the raw data browser's view
        initial_raw_time = self.raw_browser_manager.get_selected_time()
        # Also updates the raw time selector
        self.sync_video_to_raw(initial_raw_time)

    @Slot(float)
    def sync_video_to_raw(self, raw_time_seconds: float) -> None:
        """Update the displayed video frame when raw view changes."""
        if self._syncing:
            # Prevent infinite recursion
            logger.debug("Already syncing, skip updating video view.")
            return
        self._syncing = True
        logger.debug("")  # Clear debug log for clarity
        logger.debug(
            "Detected change in raw data browser's selected time, syncing video."
        )

        self._update_video(raw_time_seconds)
        self._syncing = False

    def _update_video(self, raw_time_seconds: float) -> None:
        """Update video browser view based on selected raw time point.

        Either shows the video frame that corresponds to the raw time point,
        or shows the first or last frame of the video if the raw time point
        is out of bounds of the video data.

        Parameters
        ----------
        raw_time_seconds : float
            The raw time point in seconds to which the video browser should be synced.
        """
        mapping = self.time_mapper.raw_time_to_video_frame_index(raw_time_seconds)

        match mapping:
            case MappingSuccess(result=video_idx):
                # Raw time point has a corresponding video frame index
                logger.debug(
                    f"Setting video browser to show frame with index: {video_idx}"
                )
                self.video_browser.display_frame_at(video_idx)
                self.video_browser.set_sync_status(SyncStatus.SYNCHRONIZED)

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_SMALL):
                # Raw time stamp is smaller than the first video frame timestamp
                logger.debug(
                    "No video data for this small raw time point, showing first frame."
                )
                self.video_browser.set_sync_status(SyncStatus.NO_VIDEO_DATA)
                self.video_browser.display_frame_at(0)

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_LARGE):
                # Raw time stamp is larger than the last video frame timestamp
                logger.debug(
                    "No video data for this large raw time point, showing last frame."
                )
                self.video_browser.set_sync_status(SyncStatus.NO_VIDEO_DATA)
                self.video_browser.display_frame_at(self.video_file.frame_count - 1)
            case _:
                raise ValueError(f"Unexpected mapping result: {mapping}. ")

    @Slot(int)
    def sync_raw_to_video(self, video_frame_idx: int) -> None:
        """Update raw data browser's view and time selector when video frame changes."""
        if self._syncing:
            # Prevent infinite recursion
            logger.debug("Already syncing, skip updating raw view.")
            return
        self._syncing = True

        logger.debug("")  # Clear debug log for clarity
        logger.debug(f"Syncing raw browser to video frame index: {video_frame_idx}")

        self._update_raw(video_frame_idx)
        self._syncing = False

    def _update_raw(self, video_frame_idx: int) -> None:
        """Update raw browser view based on selected video frame index.

        If the video frame index is out of bounds of the raw data, moves the raw view
        to the start or end of the raw data.

        Parameters
        ----------
        video_frame_idx : int
            The video frame index to which the raw browser should be synced.

        """
        mapping = self.time_mapper.video_frame_index_to_raw_time(video_frame_idx)

        match mapping:
            case MappingSuccess(result=raw_time):
                # Video frame index has a corresponding raw time point
                logger.debug(f"Corresponding raw time in seconds: {raw_time:.3f}")
                self.raw_browser_manager.set_selected_time(raw_time)
                self.video_browser.set_sync_status(SyncStatus.SYNCHRONIZED)

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_SMALL):
                # Video frame index is smaller than the first raw time point
                logger.debug(
                    "No raw data for this small video frame, moving raw view to start."
                )
                self.video_browser.set_sync_status(SyncStatus.NO_RAW_DATA)
                self.raw_browser_manager.jump_to_start()

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_LARGE):
                logger.debug(
                    "No raw data for this large video frame, moving raw view to end."
                )
                self.video_browser.set_sync_status(SyncStatus.NO_RAW_DATA)
                self.raw_browser_manager.jump_to_end()
            case _:
                raise ValueError(f"Unexpected mapping result: {mapping}. ")

    def show(self) -> None:
        """Show the synchronized raw and video browsers."""
        self.raw_browser_manager.show_browser()
        self.dock.show()


class BufferedThrottler(QObject):
    """Emits the most recent payload no more than once every `interval_ms`.

    If enough time has passed since last emit, emits the received payload immediately.
    Otherwise schedules the received payload to be emitted after the required time has
    passed.

    Parameters
    ----------
    interval_ms : int
        The minimum interval in milliseconds between emits.
    parent : QObject, optional
        The parent QObject for this throttler, by default None.
    """

    triggered = Signal(int)

    def __init__(self, interval_ms: int, parent=None) -> None:
        super().__init__(parent)

        self._emit_interval_ms = interval_ms
        self._latest_payload = None  # holds the next value to emit

        # Start a timer to count milliseconds since last emit.
        self._elapsed_timer = QElapsedTimer()
        self._elapsed_timer.start()

        # Initialize another timer to schedule emits to happen later.
        self._delayed_emit_timer = QTimer(parent=self)
        self._delayed_emit_timer.setSingleShot(True)
        self._delayed_emit_timer.timeout.connect(self._emit_now)

    @Slot(int)
    def trigger(self, payload: int) -> None:
        """Trigger the throttler with a new payload."""
        self._latest_payload = payload

        elapsed_time_ms = self._elapsed_timer.elapsed()
        remaining_time_ms = self._emit_interval_ms - elapsed_time_ms

        if remaining_time_ms <= 0:
            # Enough time has passed since last emit.
            self._emit_now()
        else:
            # Triggered too soon. Start delayed emit timer if its not already running.
            if not self._delayed_emit_timer.isActive():
                self._delayed_emit_timer.start(remaining_time_ms)

    @Slot()
    def _emit_now(self):
        # Start counting time since last emit again from zero.
        self._elapsed_timer.restart()
        # Make sure that no delayed emits will happen before new trigger.
        self._delayed_emit_timer.stop()
        # Fire!
        logger.debug(f"Emitting latest payload: {self._latest_payload}")
        self.triggered.emit(self._latest_payload)
