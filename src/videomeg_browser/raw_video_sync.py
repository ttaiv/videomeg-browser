"""Contains the main class for synchronizing MNE raw data browser and video browser."""

import logging

import mne
from qtpy import QtWidgets
from qtpy.QtCore import Qt, Slot

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


class SyncedRawVideoBrowser:
    """Instantiates MNE raw data browser and video browser, and synchronizes them."""

    def __init__(
        self,
        raw: mne.io.Raw,
        video_file: VideoFile,
        time_mapper: TimeIndexMapper,
    ) -> None:
        self.raw = raw
        self.video_file = video_file
        self.time_mapper = time_mapper
        # Flag to prevent infinite recursion during synchronization
        self._syncing = False

        # Set up Qt application
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        # Instantiate the MNE Qt Browser
        self.raw_browser = raw.plot(block=False)
        # Wrap it in a interface class that exposes the necessary methods
        self.raw_browser_interface = RawBrowserInterface(self.raw_browser)
        # Pass interface for manager that contains actual logic for managing the browser
        # in sync with the video browser
        self.raw_browser_manager = RawBrowserManager(self.raw_browser_interface)

        # Set up the video browser
        self.video_browser = VideoBrowser(video_file, show_sync_status=True)

        # Dock the video browser to the raw data browser with Qt magic
        self.dock = QtWidgets.QDockWidget("Video Browser", self.raw_browser)
        self.dock.setWidget(self.video_browser)
        self.dock.setFloating(True)
        self.raw_browser.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.resize(1000, 800)  # Set initial size of the video browser

        # Set up synchronization

        # When video browser frame changes, update the raw data browser's view
        self.video_browser.sigFrameChanged.connect(self.sync_raw_to_video)
        # When either raw time selector value or raw data browser's view changes,
        # update the video browser
        self.raw_browser_manager.sigSelectedTimeChanged.connect(self.sync_video_to_raw)

        # Consider raw data browser to be the main browser and start by
        # synchronizing the video browser to the raw data browser's view
        initial_raw_time = self.raw_browser_manager.get_selected_time()
        # Also updates the raw time selector
        self.sync_video_to_raw(initial_raw_time)

    @Slot(tuple)
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
        self.app.exec_()
