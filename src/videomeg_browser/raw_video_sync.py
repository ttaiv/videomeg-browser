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
        # Default relative position of the time selector (marker that shows the time
        # point that is used to determine which video frame to display) in the raw data
        # browser's view. In the boundaries of raw data, this will not be obeyed.
        self.time_selector_fraction = 0.5

        # Set up Qt application
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        # Instantiate the MNE Qt Browser
        self.raw_browser = raw.plot(block=False)

        # Vertical line that shows the time point in the raw data browser
        # that corresponds to the currently displayed video frame
        self.raw_time_selector = pg.InfiniteLine(
            pos=0, angle=90, movable=False, pen=pg.mkPen("r", width=3)
        )
        self.raw_browser.mne.plt.addItem(self.raw_time_selector)

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
        # And vice versa
        self.raw_browser.mne.plt.sigXRangeChanged.connect(
            lambda _, xrange: self.sync_video_to_raw(xrange)
        )

        # Consider raw data browser to be the main browser and start by
        # synchronizing the video browser to the raw data browser's view
        initial_xrange = self.raw_browser.mne.plt.getViewBox().viewRange()[0]
        self.sync_video_to_raw(initial_xrange)  # Also updates the raw time selector

    @Slot(tuple)
    def sync_video_to_raw(self, raw_xrange: tuple[float, float]):
        """Update raw time selector and displayed video frame when raw view changes."""
        if self._syncing:
            # Prevent infinite recursion
            logger.debug("Already syncing, skip updating video view.")
            return
        self._syncing = True
        logger.debug("")  # Clear debug log for clarity
        logger.debug("Detected change in raw data browser's xRange, syncing video.")

        logger.debug("Updating raw time selector value based on raw view.")
        # Returns the new time selector position in seconds
        raw_time_seconds = self._update_raw_time_selector_based_on_raw_view(raw_xrange)

        logger.debug("Using updated raw time selector value to sync video.")
        logger.debug(f"Syncing video to raw time: {raw_time_seconds:.3f} seconds")

        mapping = self.time_mapper.raw_time_to_video_frame_index(raw_time_seconds)
        if mapping.result is not None:
            # Raw time point has a corresponding video frame index
            video_idx = mapping.result
            logger.debug(f"Setting video browser to show frame with index: {video_idx}")
            self.video_browser.display_frame_at(video_idx)
            self.video_browser.set_sync_status(SyncStatus.SYNCHRONIZED)
        else:
            # Raw time point is out of bounds of the video bounds
            self.video_browser.set_sync_status(SyncStatus.NO_VIDEO_DATA)

        self._syncing = False

    @Slot(int)
    def sync_raw_to_video(self, video_frame_idx: int):
        """Update raw data browser's view and time selector when video frame changes."""
        if self._syncing:
            # Prevent infinite recursion
            logger.debug("Already syncing, skip updating raw view.")
            return
        self._syncing = True

        logger.debug("")  # Clear debug log for clarity
        logger.debug(f"Syncing raw browser to video frame index: {video_frame_idx}")
        mapping = self.time_mapper.video_frame_index_to_raw_time(video_frame_idx)
        if mapping.result is not None:
            # Video frame index has a corresponding raw time point
            raw_time = mapping.result
            logger.debug(f"Corresponding raw time in seconds: {raw_time:.3f}")
            # Set the time selector value based on the video frame
            self.raw_time_selector.setValue(raw_time)
            # And update the raw data browser's view so that the selector
            # remains at the same relative position in the view
            self.update_raw_view_based_on_raw_time_selector()
            self.video_browser.set_sync_status(SyncStatus.SYNCHRONIZED)
        else:
            # Video frame index is out of bounds of the raw data bounds
            # Signal video browser that there is no raw data for this frame
            # and move the raw view either to the start or end of the raw data
            self.video_browser.set_sync_status(SyncStatus.NO_RAW_DATA)
            if mapping.failure_reason == MapFailureReason.INDEX_TOO_SMALL:
                logger.debug(
                    "No raw data for this small video frame, moving raw view to start."
                )
                self.set_raw_view_start()
                self.raw_time_selector.setValue(0.0)
            elif mapping.failure_reason == MapFailureReason.INDEX_TOO_LARGE:
                logger.debug(
                    "No raw data for this large video frame, moving raw view to end."
                )
                self.set_raw_view_to_end()
                self.raw_time_selector.setValue(self.raw_browser.mne.xmax)
            else:
                raise ValueError(
                    f"Unexpected mapping failure reason: {mapping.failure_reason}"
                )

        self._syncing = False

    def _update_raw_time_selector_based_on_raw_view(
        self, new_raw_xrange: tuple[float, float]
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
        raw_xmin = new_raw_xrange[0]
        raw_xmax = new_raw_xrange[1]

        # Calculate the new position of the time point selector
        selector_pos = raw_xmin + (raw_xmax - raw_xmin) * self.time_selector_fraction
        logger.debug(f"Setting raw time point selector to {selector_pos:.3f} seconds.")
        self.raw_time_selector.setValue(selector_pos)

        return selector_pos

    def update_raw_view_based_on_raw_time_selector(self):
        """Set raw view based on the raw time selector.

        The raw time selector will stay at the same relative position in the view,
        expect when the view is at the boundaries of the raw data.
        """
        # Get specs for the raw data browser's view
        # All are in seconds
        window_len = self.raw_browser.mne.duration
        view_xmin = 0
        view_xmax = self.raw_browser.mne.xmax

        time_selector_pos = self.raw_time_selector.value()
        if not isinstance(time_selector_pos, float):
            raise TypeError(
                f"Expected raw time selector value to be a float, "
                f"but got {type(time_selector_pos)}."
            )
        logger.debug(
            f"Video marker position for raw view updating: {time_selector_pos:.3f} "
            "seconds."
        )

        # Calculate new xmin and xmax for the raw data browser's view
        xmin = time_selector_pos - window_len * self.time_selector_fraction
        xmax = time_selector_pos + window_len * (1 - self.time_selector_fraction)

        if xmin < view_xmin:
            logger.debug(
                f"Raw view xmin {xmin:.3f} is less than the minimum view time "
                f"{view_xmin:.3f}. Setting view to range "
                f"[{view_xmin:.3f}, {view_xmin + window_len}] seconds."
            )
            self.raw_browser.mne.plt.setXRange(
                view_xmin, view_xmin + window_len, padding=0
            )
        elif xmax > view_xmax:
            logger.debug(
                f"Raw view xmax {xmax:.3f} is greater than the maximum view time "
                f"{view_xmax:.3f}. Setting view to range "
                f"[{view_xmax - window_len:.3f}, {view_xmax:.3f}] seconds."
            )
            self.raw_browser.mne.plt.setXRange(
                view_xmax - window_len, view_xmax, padding=0
            )
        else:
            logger.debug(
                f"Setting raw view to show video marker at {time_selector_pos:.3f} seconds "
                f"with range [{xmin:.3f}, {xmax:.3f}] seconds."
            )
            self.raw_browser.mne.plt.setXRange(xmin, xmax, padding=0)

    def set_raw_view_start(self):
        """Set the raw data browser's view to the beginning of the data."""
        xmin = 0.0
        xmax = xmin + self.raw_browser.mne.duration
        logger.debug(
            f"Setting raw view to range [{xmin:.3f}, {xmax:.3f}] seconds "
            "at the start of the data."
        )
        self.raw_browser.mne.plt.setXRange(xmin, xmax, padding=0)

    def set_raw_view_to_end(self):
        """Set the raw data browser's view to the end of the data."""
        xmax = self.raw_browser.mne.xmax
        xmin = xmax - self.raw_browser.mne.duration
        logger.debug(
            f"Setting raw view to range [{xmin:.3f}, {xmax:.3f}] seconds "
            "at the end of the data."
        )
        self.raw_browser.mne.plt.setXRange(xmin, xmax, padding=0)

    def show(self):
        """Show the synchronized raw and video browsers."""
        self.raw_browser.show()
        self.app.exec_()
