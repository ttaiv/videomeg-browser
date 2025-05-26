import os.path as op
import sys

import mne
import numpy as np
from mne_qt_browser._pg_figure import TimeScrollBar
from qtpy import QtWidgets
from qtpy.QtCore import Qt

from .browser import VideoBrowser
from .comp_tstamps import comp_tstamps
from .video import VideoFile, VideoFileHelsinkiVideoMEG


class TimeStampMapper:
    """Maps time points from raw data to video frames."""

    def __init__(
        self, raw: mne.io.Raw, raw_timing_ch: str, video: VideoFileHelsinkiVideoMEG
    ):
        self.raw = raw
        picks_timing = mne.pick_types(raw.info, meg=False, include=[raw_timing_ch])
        dt_timing = raw[picks_timing, :][0].squeeze()

        print("Beginning timestamp computation...")
        self.raw_ts = comp_tstamps(dt_timing, raw.info["sfreq"])
        print("Timestamp computation done.")
        self.vid_ts = video.ts

        print(f"Number of raw timestamps: {len(self.raw_ts)}")
        print(f"Number of video timestamps: {len(self.vid_ts)}")

    def raw_time_to_video_frame_index(self, raw_t: float) -> int | None:
        """Convert a time point from raw data (in seconds) to video frame index."""
        raw_idx = self.raw.time_as_index(raw_t, use_rounding=False)
        if len(raw_idx) > 1:
            raise ValueError(
                f"Multiple indices found for raw timestamp {raw_t}: {raw_idx}. "
                "This should not happen."
            )
        raw_idx = raw_idx[0]
        print(f"Raw index for time {raw_t}: {raw_idx}")

        # Convert raw index to unix timestamp in milliseconds
        raw_ts = self.raw_ts[raw_idx]
        print(f"Raw timestamp at index {raw_idx}: {raw_ts}")

        # Now we have temporal location of the raw data point in same units as video
        # timestamps, so we can compare them directly.

        if raw_ts < self.vid_ts[0] or raw_ts > self.vid_ts[-1]:
            print("Raw timestamp is out of video bounds, returning None.")
            return None

        # Find the first video frame index that is greater than
        # or equal to the raw timestamp
        # TODO: Consider what other methods could be used here
        idx = np.searchsorted(self.vid_ts, raw_ts)

        return int(idx)


class ScrollBarSynchronizer:
    """Synchronize the scroll bar of the raw data browser with the video browser."""

    def __init__(
        self,
        raw_scroll_bar: TimeScrollBar,
        vid_slider: QtWidgets.QSlider,
        ts_mapper: TimeStampMapper,
    ):
        self.raw_scroll_bar = raw_scroll_bar
        self.vid_slider = vid_slider
        self.ts_mapper = ts_mapper
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

            vid_idx = self.ts_mapper.raw_time_to_video_frame_index(raw_t)
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
            self.raw_scroll_bar.setValue(value)
            self._syncing = False


def plot_raw_with_video(raw: mne.io.Raw, video: VideoFile):
    """Run mne raw data browser in sync with video browser."""
    # Set up Qt application
    app = QtWidgets.QApplication([])
    # Instantiate the MNE Qt Browser
    raw_browser = raw.plot(block=False)

    # Set up the video browser
    video_browser = VideoBrowser(video_file)

    # Dock the video browser to the raw data browser with Qt magic
    dock = QtWidgets.QDockWidget("Video Browser", raw_browser)
    dock.setWidget(video_browser)
    dock.setFloating(True)
    raw_browser.addDockWidget(Qt.RightDockWidgetArea, dock)

    ts_mapper = TimeStampMapper(raw, raw_timing_ch="STI016", video=video_file)

    # Sync the scroll bars
    sync = ScrollBarSynchronizer(
        raw_browser.mne.ax_hscroll, video_browser.frame_slider, ts_mapper
    )

    # Profit
    raw_browser.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    base_path = "/u/69/taivait1/unix/video_meg_testing/Subject_2_Luna"
    # Create a video file object
    video_file = VideoFileHelsinkiVideoMEG(
        op.join(base_path, "Video_MEG", "animal_meg_subject_2_240614.video.dat")
    )

    # Create a raw data object
    raw = mne.io.read_raw_fif(
        op.join(base_path, "Raw", "animal_meg_subject_2_240614.fif"), preload=True
    )

    plot_raw_with_video(raw, video_file)
