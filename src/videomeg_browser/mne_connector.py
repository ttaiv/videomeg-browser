import logging
import os.path as op

import mne
import numpy as np
from mne_qt_browser._pg_figure import TimeScrollBar
from qtpy import QtWidgets
from qtpy.QtCore import Qt

from .browser import VideoBrowser
from .comp_tstamps import comp_tstamps
from .video import VideoFileHelsinkiVideoMEG

logger = logging.getLogger(__name__)


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

    def raw_time_to_video_frame_index(self, raw_time_seconds: float) -> int | None:
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

        if (
            raw_timestamp_ms < self.vid_timestamps_ms[0]
            or raw_timestamp_ms > self.vid_timestamps_ms[-1]
        ):
            logger.debug("Raw timestamp is out of video bounds, returning None.")
            return None

        # Find the first video frame index that is greater than
        # or equal to the raw timestamp
        # TODO: Consider what other methods could be used here
        idx = np.searchsorted(self.vid_timestamps_ms, raw_timestamp_ms)

        return int(idx)

    def video_frame_index_to_raw_time(self, vid_idx: int) -> float | None:
        """Convert a video frame index to a raw data time point (in seconds)."""
        if vid_idx < 0 or vid_idx >= len(self.vid_timestamps_ms):
            raise ValueError(
                f"Video frame index {vid_idx} is out of bounds. "
                f"Valid range is 0 to {len(self.vid_timestamps_ms) - 1}."
            )

        # Get unix timestamp of the video frame
        vid_timestamp_ms = self.vid_timestamps_ms[vid_idx]
        logger.debug(f"Video unix timestamp at index {vid_idx}: {vid_timestamp_ms} ms")

        if (
            vid_timestamp_ms < self.raw_timestamps_ms[0]
            or vid_timestamp_ms > self.raw_timestamps_ms[-1]
        ):
            logger.debug("Video timestamp is out of raw data bounds, returning None.")
            return None

        # Find the first raw timestamp that is greater than
        # or equal to the video timestamp
        # TODO: Consider what other methods could be used here
        raw_idx = np.searchsorted(self.raw_timestamps_ms, vid_timestamp_ms)
        logger.debug(f"Raw index for video unix timestamp {vid_idx}: {raw_idx}")

        raw_time_seconds = self.raw.times[raw_idx]
        logger.debug(f"Raw time at index {raw_idx}: {raw_time_seconds} seconds")

        return raw_time_seconds


class ScrollBarSynchronizer:
    """Synchronize the scroll bar of the raw data browser with the video browser."""

    def __init__(
        self,
        raw_scroll_bar: TimeScrollBar,
        vid_slider: QtWidgets.QSlider,
        time_mapper: TimeIndexMapper,
    ):
        self.raw_scroll_bar = raw_scroll_bar
        self.vid_slider = vid_slider
        self.time_mapper = time_mapper
        self._syncing = False

        self.raw_scroll_bar.valueChanged.connect(self._sync_video_to_raw)
        self.vid_slider.valueChanged.connect(self._sync_raw_to_video)

    def _sync_video_to_raw(self, value):
        """Update the video position based on the raw data browser's scroll bar."""
        if not self._syncing:
            self._syncing = True

            print()
            print(f"Syncing raw scroll bar value: {value}")
            raw_t = value / self.raw_scroll_bar.step_factor
            print(f"Corresponding raw time in seconds: {raw_t:.6f}")

            vid_idx = self.time_mapper.raw_time_to_video_frame_index(raw_t)
            print(f"Corresponding video frame index: {vid_idx}")

            if vid_idx is None:
                print("No corresponding video frame found for this raw timestamp.")
            else:
                self.vid_slider.setValue(vid_idx)

            self._syncing = False

    def _sync_raw_to_video(self, value):
        """Update the raw data browser's scroll bar based on the video slider."""
        if not self._syncing:
            self._syncing = True

            print()
            print(f"Syncing video slider value: {value}")
            raw_t = self.time_mapper.video_frame_index_to_raw_time(value)
            if raw_t is None:
                print("No corresponding raw time found for this video frame index.")
            else:
                print(f"Corresponding raw time in seconds: {raw_t:.6f}")
                # Convert raw time to scroll bar value
                scroll_value = int(raw_t * self.raw_scroll_bar.step_factor)
                print(f"Setting raw scroll bar value to: {scroll_value}")
                self.raw_scroll_bar.setValue(scroll_value)

            self._syncing = False


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
        self.video_browser = VideoBrowser(video_file)

        # Dock the video browser to the raw data browser with Qt magic
        self.dock = QtWidgets.QDockWidget("Video Browser", self.raw_browser)
        self.dock.setWidget(self.video_browser)
        self.dock.setFloating(True)
        self.raw_browser.addDockWidget(Qt.RightDockWidgetArea, self.dock)

        # Sync the scroll bars
        self.sync = ScrollBarSynchronizer(
            self.raw_browser.mne.ax_hscroll,
            self.video_browser.frame_slider,
            time_mapper,
        )

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
