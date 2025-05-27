import logging
import os.path as op
from dataclasses import dataclass
from enum import Enum, auto

import mne
import numpy as np
from qtpy import QtWidgets
from qtpy.QtCore import Qt

from .browser import SyncStatus, VideoBrowser
from .comp_tstamps import comp_tstamps
from .video import VideoFileHelsinkiVideoMEG

logger = logging.getLogger(__name__)


class MapFailureReason(Enum):
    """Enum telling why mapping from frame index to raw time or vice versa failed."""

    # Index to map is smaller than the first frame or raw time point
    INDEX_TOO_SMALL = auto()
    # Index to map is larger than the last frame or raw time point
    INDEX_TOO_LARGE = auto()


@dataclass
class MappingResult:
    """Result of mapping a video frame index to a raw time point or vice versa."""

    result: int | None
    failure_reason: MapFailureReason | None = None

    def __post_init__(self):
        """Check that mapping yielded either a result or a failure reason."""
        if (self.result is not None and self.failure_reason is not None) or (
            self.result is None and self.failure_reason is None
        ):
            raise ValueError("Exactly one of 'result' or 'failure_reason' must be set.")


class TimeIndexMapper:
    """Maps time points from raw data to video frames and vice versa.

    Currently, this is tailored for the Helsinki Video MEG data format,
    but this could be extended to other formats as well.
    """

    def __init__(
        self, raw: mne.io.Raw, raw_timing_ch: str, video: VideoFileHelsinkiVideoMEG
    ):
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
            return MappingResult(
                result=None, failure_reason=MapFailureReason.INDEX_TOO_SMALL
            )
        if raw_timestamp_ms > self.vid_timestamps_ms[-1]:
            return MappingResult(
                result=None, failure_reason=MapFailureReason.INDEX_TOO_LARGE
            )

        # Find the first video frame index that is greater than
        # or equal to the raw timestamp
        # TODO: Consider what other methods could be used here
        idx = np.searchsorted(self.vid_timestamps_ms, raw_timestamp_ms)

        return MappingResult(result=int(idx), failure_reason=None)

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
            return MappingResult(
                result=None, failure_reason=MapFailureReason.INDEX_TOO_SMALL
            )
        if vid_timestamp_ms > self.raw_timestamps_ms[-1]:
            return MappingResult(
                result=None, failure_reason=MapFailureReason.INDEX_TOO_LARGE
            )

        # Find the first raw timestamp that is greater than
        # or equal to the video timestamp
        # TODO: Consider what other methods could be used here
        raw_idx = np.searchsorted(self.raw_timestamps_ms, vid_timestamp_ms)
        logger.debug(f"Raw index for video unix timestamp {vid_idx}: {raw_idx}")

        raw_time_seconds = self.raw.times[raw_idx]
        logger.debug(f"Raw time at index {raw_idx}: {raw_time_seconds} seconds")

        return MappingResult(result=raw_time_seconds, failure_reason=None)


class SyncedRawVideoBrowser:
    """Run mne raw data browser in sync with video browser."""

    def __init__(
        self,
        raw: mne.io.Raw,
        video_file: VideoFileHelsinkiVideoMEG,
        time_mapper: TimeIndexMapper,
    ):
        self.raw = raw
        self.video_file = video_file
        self.time_mapper = time_mapper

        # Set up Qt application
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        # Instantiate the MNE Qt Browser
        self.raw_browser = raw.plot(block=False)

        # Set up the video browser
        self.video_browser = VideoBrowser(video_file, show_sync_status=True)

        # Dock the video browser to the raw data browser with Qt magic
        self.dock = QtWidgets.QDockWidget("Video Browser", self.raw_browser)
        self.dock.setWidget(self.video_browser)
        self.dock.setFloating(True)
        self.raw_browser.addDockWidget(Qt.RightDockWidgetArea, self.dock)

        # Extract the raw data browser's scroll bar and video browser's slider
        # for easy synchronization
        self.vid_slider = self.video_browser.frame_slider
        self.raw_scroll_bar = self.raw_browser.mne.ax_hscroll
        # Flag to prevent infinite recursion during synchronization
        self._syncing = False

        # Synch the bars
        self.raw_scroll_bar.valueChanged.connect(self.sync_video_to_raw)
        self.vid_slider.valueChanged.connect(self.sync_raw_to_video)

        # Consider raw data browser to be the main browser and start by
        # synchronizing the video browser to the raw data browser's scroll bar
        initial_value = self.raw_scroll_bar.value()
        self.sync_video_to_raw(initial_value)

    def sync_video_to_raw(self, value):
        """Update the video position based on the raw data browser's scroll bar."""
        if self._syncing:
            # Prevent infinite recursion
            logger.debug("Already syncing, skip updating video slider.")
            return
        self._syncing = True

        logger.debug("")  # Clear debug log for clarity
        logger.debug(f"Syncing video to raw scroll bar value: {value}")
        raw_time = value / self.raw_scroll_bar.step_factor
        logger.debug(f"Corresponding raw time in seconds: {raw_time:.3f}")

        mapping = self.time_mapper.raw_time_to_video_frame_index(raw_time)
        if mapping.result is not None:
            # Raw time point has a corresponding video frame index
            vid_idx = mapping.result
            logger.debug(f"Setting video slider to frame index: {vid_idx}")
            self.vid_slider.setValue(vid_idx)
            self.video_browser.set_sync_status(SyncStatus.SYNCHRONIZED)
        else:
            # Raw time point is out of bounds of the video bounds
            # Update the video slider to the closest valid index
            self.video_browser.set_sync_status(SyncStatus.NO_VIDEO_DATA)
            if mapping.failure_reason == MapFailureReason.INDEX_TOO_SMALL:
                logger.debug(
                    "Raw time is before the first video frame. "
                    "Setting video to the first frame."
                )
                self.vid_slider.setValue(0)
            elif mapping.failure_reason == MapFailureReason.INDEX_TOO_LARGE:
                logger.debug(
                    "Raw time is after the last video frame "
                    "Setting video to the last frame."
                )
                self.vid_slider.setValue(self.video_file.frame_count - 1)
            else:
                raise ValueError(
                    f"Unexpected mapping failure reason: {mapping.failure_reason}"
                )
        self._syncing = False

    def sync_raw_to_video(self, value):
        """Update the raw data browser's scroll bar based on the video slider."""
        if self._syncing:
            # Prevent infinite recursion
            logger.debug("Already syncing, skip updating raw scroll bar.")
            return
        self._syncing = True

        logger.debug("")  # Clear debug log for clarity
        logger.debug(f"Syncing raw to video slider value: {value}")
        mapping = self.time_mapper.video_frame_index_to_raw_time(value)
        if mapping.result is not None:
            # Video frame index has a corresponding raw time point
            raw_time = mapping.result
            logger.debug(f"Corresponding raw time in seconds: {raw_time:.3f}")
            scroll_value = int(raw_time * self.raw_scroll_bar.step_factor)
            logger.debug(f"Setting raw scroll bar value: {scroll_value}")
            self.raw_scroll_bar.setValue(scroll_value)
            self.video_browser.set_sync_status(SyncStatus.SYNCHRONIZED)
        else:
            # Video frame index is out of bounds of the raw data bounds
            # Update the raw scroll bar to the closest valid index
            self.video_browser.set_sync_status(SyncStatus.NO_RAW_DATA)
            if mapping.failure_reason == MapFailureReason.INDEX_TOO_SMALL:
                logger.debug(
                    "Video frame index is before the first raw time point. "
                    "Setting raw scroll bar to the first time point."
                )
                self.raw_scroll_bar.setValue(0)
            elif mapping.failure_reason == MapFailureReason.INDEX_TOO_LARGE:
                logger.debug(
                    "Video frame index is after the last raw time point. "
                    "Setting raw scroll bar to the last time point."
                )
                # Calculate maximum raw scroll bar value
                # Taken from raw data browser's scroll bar code
                raw_scroll_max = int(
                    (self.raw_scroll_bar.mne.xmax - self.raw_scroll_bar.mne.duration)
                    * self.raw_scroll_bar.step_factor
                )
                self.raw_scroll_bar.setValue(raw_scroll_max)
            else:
                raise ValueError(
                    f"Unexpected mapping failure reason: {mapping.failure_reason}"
                )

        self._syncing = False

    def show(self):
        """Show the synchronized raw and video browsers."""
        self.raw_browser.show()
        self.app.exec_()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    base_path = "/u/69/taivait1/unix/video_meg_testing/Subject_2_Luna"
    # Create a video file object
    video_file = VideoFileHelsinkiVideoMEG(
        op.join(base_path, "Video_MEG", "animal_meg_subject_2_240614.video.dat")
    )

    # Create a raw data object
    raw = mne.io.read_raw_fif(
        op.join(base_path, "Raw", "animal_meg_subject_2_240614.fif"), preload=True
    )

    # Set up mapping between time points of raw data and video frame indices
    # This is tailored for the Helsinki Video MEG data format
    time_mapper = TimeIndexMapper(raw, raw_timing_ch="STI016", video=video_file)

    browser = SyncedRawVideoBrowser(raw, video_file, time_mapper)
    browser.show()
