import logging
from abc import ABC
from dataclasses import dataclass
from enum import Enum

import mne
import numpy as np
from qtpy import QtWidgets
from qtpy.QtCore import Qt, Slot

from .comp_tstamps import comp_tstamps
from .raw_browser_manager import RawBrowserInterface, RawBrowserManager
from .video import VideoFileHelsinkiVideoMEG
from .video_browser import SyncStatus, VideoBrowser

logger = logging.getLogger(__name__)


class MapFailureReason(Enum):
    """Enum telling why mapping from frame index to raw time or vice versa failed."""

    # Index to map is smaller than the first frame or raw time point
    INDEX_TOO_SMALL = "index_too_small"
    # Index to map is larger than the last frame or raw time point
    INDEX_TOO_LARGE = "index_too_large"


class MappingResult(ABC):
    """Represents the result of mapping raw time to video frame index or vice versa."""

    pass


@dataclass(frozen=True)
class MappingSuccess(MappingResult):
    """Represents a successful mapping that yielded a raw time or video frame index."""

    result: int


@dataclass(frozen=True)
class MappingFailure(MappingResult):
    """Represents a failed mapping with a reason for the failure."""

    failure_reason: MapFailureReason


class TimeIndexMapper:
    """Maps time points from raw data to video frames and vice versa.

    Currently, this is tailored for the Helsinki Video MEG data format,
    but this could be extended to other formats as well.
    """

    def __init__(
        self, raw: mne.io.Raw, raw_timing_ch: str, video: VideoFileHelsinkiVideoMEG
    ) -> None:
        self.raw = raw
        self.vid_timestamps_ms = video.ts

        logger.info("Initializing mapping from raw data time points to video frames.")
        logger.info(
            f"Using timing channel '{raw_timing_ch}' for timestamp computation."
        )

        timing_data = raw.get_data(picks=raw_timing_ch, return_times=False)
        # Remove the channel dimension
        # Ignoring warning about timing_data possibly being tuple,
        # as we do not ask times from raw.get_data
        timing_data = timing_data.squeeze()  # type: ignore
        logger.debug(f"Timing channel data shape: {timing_data.shape}, ")

        self.raw_timestamps_ms = comp_tstamps(timing_data, raw.info["sfreq"])

        if len(self.raw_timestamps_ms) != len(raw.times):
            raise ValueError(
                "The number of timestamps in the raw data does not match "
                "the number of time points."
            )

        if not np.all(np.diff(self.raw_timestamps_ms) >= 0):
            raise ValueError(
                "Raw timestamps are not strictly increasing. "
                "This is required for the mapping to work correctly."
            )
        if not np.all(np.diff(self.vid_timestamps_ms) >= 0):
            raise ValueError(
                "Video timestamps are not strictly increasing. "
                "This is required for the mapping to work correctly."
            )

        logger.info(f"Number of raw timestamps: {len(self.raw_timestamps_ms)}")
        logger.info(f"Number of video timestamps: {len(self.vid_timestamps_ms)}")

    def raw_time_to_video_frame_index(self, raw_time_seconds: float) -> MappingResult:
        """Convert a time point from raw data (in seconds) to video frame index."""
        raw_idx = self.raw.time_as_index(raw_time_seconds, use_rounding=False)
        if len(raw_idx) > 1:
            raise ValueError(
                "Multiple indices found for raw timestamp "
                f"{raw_time_seconds}: {raw_idx}. This should not happen."
            )
        raw_idx = raw_idx[0]
        logger.debug(f"Raw index for time {raw_time_seconds}: {raw_idx}")

        # Convert raw index to unix timestamp in milliseconds
        raw_timestamp_ms = self.raw_timestamps_ms[raw_idx]
        logger.debug(f"Raw unix timestamp at index {raw_idx}: {raw_timestamp_ms} ms")

        # Now we have temporal location of the raw data point in same units as video
        # timestamps, so we can compare them directly.

        if raw_timestamp_ms < self.vid_timestamps_ms[0]:
            return MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_SMALL)
        if raw_timestamp_ms > self.vid_timestamps_ms[-1]:
            return MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_LARGE)

        # Find the first video frame index that is greater than
        # or equal to the raw timestamp
        # TODO: Consider what other methods could be used here
        idx = np.searchsorted(self.vid_timestamps_ms, raw_timestamp_ms)

        return MappingSuccess(result=int(idx))

    def video_frame_index_to_raw_time(self, vid_idx: int) -> MappingResult:
        """Convert a video frame index to a raw data time point (in seconds)."""
        if vid_idx < 0 or vid_idx >= len(self.vid_timestamps_ms):
            raise ValueError(
                f"Video frame index {vid_idx} is out of bounds. "
                f"Valid range is 0 to {len(self.vid_timestamps_ms) - 1}."
            )

        # Get unix timestamp of the video frame
        vid_timestamp_ms = self.vid_timestamps_ms[vid_idx]
        logger.debug(f"Video unix timestamp at index {vid_idx}: {vid_timestamp_ms} ms")

        if vid_timestamp_ms < self.raw_timestamps_ms[0]:
            return MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_SMALL)
        if vid_timestamp_ms > self.raw_timestamps_ms[-1]:
            return MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_LARGE)

        # Find the first raw timestamp that is greater than
        # or equal to the video timestamp
        # TODO: Consider what other methods could be used here
        raw_idx = np.searchsorted(self.raw_timestamps_ms, vid_timestamp_ms)
        logger.debug(f"Raw index for video unix timestamp {vid_idx}: {raw_idx}")

        raw_time_seconds = self.raw.times[raw_idx]
        logger.debug(f"Raw time at index {raw_idx}: {raw_time_seconds} seconds")

        return MappingSuccess(result=raw_time_seconds)


class SyncedRawVideoBrowser:
    """Instantiates MNE raw data browser and video browser, and synchronizes them."""

    def __init__(
        self,
        raw: mne.io.Raw,
        video_file: VideoFileHelsinkiVideoMEG,
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

        self._syncing = False

    def show(self) -> None:
        """Show the synchronized raw and video browsers."""
        self.raw_browser_manager.show_browser()
        self.app.exec_()
