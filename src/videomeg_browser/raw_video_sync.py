import logging
from dataclasses import dataclass
from enum import Enum, auto

import mne
import numpy as np
import pyqtgraph as pg
from qtpy import QtWidgets
from qtpy.QtCore import Qt, Slot

from .comp_tstamps import comp_tstamps
from .video import VideoFileHelsinkiVideoMEG
from .video_browser import SyncStatus, VideoBrowser

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
    """Instantiates MNE raw data browser and video browser, and synchronizes them."""

    def __init__(
        self,
        raw: mne.io.Raw,
        video_file: VideoFileHelsinkiVideoMEG,
        time_mapper: TimeIndexMapper,
    ):
        self.raw = raw
        self.video_file = video_file
        self.time_mapper = time_mapper
        # Flag to prevent infinite recursion during synchronization
        self._syncing = False

        # Set up Qt application
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        # Instantiate the MNE Qt Browser
        self.raw_browser = raw.plot(block=False)

        # Add a vertical line marker for the video frame position
        self.video_marker = pg.InfiniteLine(
            pos=0, angle=90, movable=False, pen=pg.mkPen("r", width=2)
        )
        self.raw_browser.mne.plt.addItem(self.video_marker)

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
        self.video_browser.frame_changed.connect(self.sync_raw_to_video)
        # And when raw data browser's xRange (time axis) changes
        # update the video browser's frame
        self.raw_browser.mne.plt.sigXRangeChanged.connect(
            lambda _, xrange: self.sync_video_to_raw(xrange)
        )

        # Consider raw data browser to be the main browser and start by
        # synchronizing the video browser to the raw data browser's view
        initial_xrange = self.raw_browser.mne.viewbox.viewRange()[0]

        logger.debug(
            f"Initial raw data view range: {initial_xrange[0]:.3f} to "
            f"{initial_xrange[1]:.3f} seconds."
        )

        self.sync_video_to_raw(initial_xrange)

    @Slot(tuple)
    def sync_video_to_raw(self, raw_xrange: tuple[float, float]):
        """Update the displayed video frame based on raw data view."""
        if self._syncing:
            # Prevent infinite recursion
            logger.debug("Already syncing, skip updating video slider.")
            return
        self._syncing = True

        # Get the middle time of the raw data browser's view
        raw_middle_time_seconds = (raw_xrange[0] + raw_xrange[1]) / 2
        logger.debug(
            f"Syncing video to raw time: {raw_middle_time_seconds:.3f} seconds"
        )

        logger.debug("")  # Clear debug log for clarity
        logger.debug(
            f"Syncing video to raw middle time: {raw_middle_time_seconds:.3f} seconds"
        )

        mapping = self.time_mapper.raw_time_to_video_frame_index(
            raw_middle_time_seconds
        )
        if mapping.result is not None:
            # Raw time point has a corresponding video frame index
            video_idx = mapping.result
            logger.debug(f"Setting video browser to show frame with index: {video_idx}")
            self.video_browser.display_frame_at(video_idx)
            self.video_browser.set_sync_status(SyncStatus.SYNCHRONIZED)
            self.video_marker.setValue(raw_middle_time_seconds)
        else:
            # Raw time point is out of bounds of the video bounds
            # Update the video slider to the closest valid index
            raise RuntimeError("This should not happen with middle time point.")
            """
            self.video_browser.set_sync_status(SyncStatus.NO_VIDEO_DATA)
            if mapping.failure_reason == MapFailureReason.INDEX_TOO_SMALL:
                logger.debug(
                    "Raw time is before the first video frame. "
                    "Setting video to the first frame."
                )
                self.video_browser.display_frame_at(0)
            elif mapping.failure_reason == MapFailureReason.INDEX_TOO_LARGE:
                logger.debug(
                    "Raw time is after the last video frame "
                    "Setting video to the last frame."
                )
                self.video_browser.display_frame_at(
                    self.video_browser.video.frame_count - 1
                )
            else:
                raise ValueError(
                    f"Unexpected mapping failure reason: {mapping.failure_reason}"
                )
            """
        self._syncing = False

    @Slot(int)
    def sync_raw_to_video(self, video_frame_idx: int):
        """Update the raw data browser's view based on the displayed video frame."""
        if self._syncing:
            # Prevent infinite recursion
            logger.debug("Already syncing, skip updating raw view.")
            return
        self._syncing = True

        logger.debug("")  # Clear debug log for clarity
        logger.debug(f"Syncing raw to video frame index: {video_frame_idx}")
        mapping = self.time_mapper.video_frame_index_to_raw_time(video_frame_idx)
        if mapping.result is not None:
            # Video frame index has a corresponding raw time point
            raw_time = mapping.result
            logger.debug(f"Corresponding raw time in seconds: {raw_time:.3f}")
            self.set_raw_view_time(raw_time)
            self.video_marker.setValue(raw_time)
            self.video_browser.set_sync_status(SyncStatus.SYNCHRONIZED)
        else:
            # Video frame index is out of bounds of the raw data bounds
            # Update the raw view to the closest valid time point
            self.video_browser.set_sync_status(SyncStatus.NO_RAW_DATA)
            if mapping.failure_reason == MapFailureReason.INDEX_TOO_SMALL:
                logger.debug(
                    "Video frame index is before the first raw time point. "
                    "Setting raw view to the start."
                )
                self.set_raw_view_start()
            elif mapping.failure_reason == MapFailureReason.INDEX_TOO_LARGE:
                logger.debug(
                    "Video frame index is after the last raw time point. "
                    "Setting raw view to the end."
                )
                self.set_raw_view_to_end()
            else:
                raise ValueError(
                    f"Unexpected mapping failure reason: {mapping.failure_reason}"
                )

        self._syncing = False

    def set_raw_view_time(self, raw_time_seconds: float):
        """Set the raw data browser's view to show a specific time point.

        If possible, the view will be centered around the specified time point.
        """
        # Get specs for the raw data browser's view
        window_len_seconds = self.raw_browser.mne.duration
        view_xmin = 0
        view_xmax = self.raw_browser.mne.xmax
        logger.debug(
            f"Raw data browser view range: [{view_xmin:.3f}, {view_xmax:.3f}] seconds."
        )
        logger.debug(f"Raw data browser duration: {window_len_seconds:.3f} seconds.")

        xmin = max(view_xmin, raw_time_seconds - window_len_seconds / 2)
        # xmax = min(raw_time_seconds + window_len_seconds / 2, view_xmax)
        xmax = xmin + window_len_seconds
        if xmax > view_xmax:
            logger.warning(
                f"Setting raw view max to a time point {xmax} that exceeds the "
                "maximum time point of the raw data."
            )
        logger.debug(
            f"Setting raw view to show time point {raw_time_seconds:.3f} seconds "
            f"with range [{xmin:.3f}, {xmax:.3f}] seconds."
        )
        self.raw_browser.mne.plt.setXRange(xmin, xmax, padding=0)

    def set_raw_view_start(self):
        """Set the raw data browser's view to the beginning of the data."""
        xmin = self.raw_browser.mne.xmin
        xmax = xmin + self.raw_browser.mne.duration
        logger.debug(
            f"Setting raw view to beginning at time {xmin:.3f} seconds "
            f"with range [{xmin:.3f}, {xmax:.3f}] seconds."
        )
        self.raw_browser.mne.plt.setXRange(xmin, xmax, padding=0)

    def set_raw_view_to_end(self):
        """Set the raw data browser's view to the end of the data."""
        xmax = self.raw_browser.mne.xmax
        xmin = xmax - self.raw_browser.mne.duration
        logger.debug(
            f"Setting raw view to end at time {xmax:.3f} seconds "
            f"with range [{xmin:.3f}, {xmax:.3f}] seconds."
        )
        self.raw_browser.mne.plt.setXRange(xmin, xmax, padding=0)

    def show(self):
        """Show the synchronized raw and video browsers."""
        self.raw_browser.show()
        self.app.exec_()
