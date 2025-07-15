import logging
import os.path as op

import mne
import numpy as np
from mne.datasets import sample
from qtpy.QtWidgets import QApplication

from videomeg_browser.raw_video_aligner import RawVideoAligner
from videomeg_browser.synced_raw_video_browser import SyncedRawVideoBrowser
from videomeg_browser.video import VideoFileHelsinkiVideoMEG

if __name__ == "__main__":
    BASE_PATH = "/u/69/taivait1/unix/video_meg_testing/2025-07-11_MEG2MEG_test/"

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [%(levelname)s] %(name)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load sample data.
    data_path = sample.data_path()
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.crop(tmax=60)

    # Load videos.
    video1 = VideoFileHelsinkiVideoMEG(
        op.join(BASE_PATH, "2025-07-11--18-18-41_video_01.vid")
    )
    video2 = VideoFileHelsinkiVideoMEG(
        op.join(BASE_PATH, "2025-07-11--18-18-41_video_02.vid")
    )

    for video in [video1, video2]:
        print(f"Stats for video {video.fname}:")
        print(f"  Frame count: {video.frame_count}")
        print(f"  {video.fps} FPS")
        print(f"  Duration: {video.frame_count / video.fps:.2f} seconds")
        print(
            f"  Timestamps: {video.timestamps_ms[0] / 1000:.2f} ... {video.timestamps_ms[-1] / 1000:.2f}"
        )
        print(f"Frame size: {video.frame_width}x{video.frame_height}")

    # Create artificial timestamps for raw data.
    start_ts = video1.timestamps_ms[0]
    end_ts = video1.timestamps_ms[-1]
    raw_timestamps_ms = np.linspace(start_ts, end_ts, raw.n_times, endpoint=False)

    # Define function for converting raw time to index
    def raw_time_to_index(time: float) -> int:
        """Convert a time in seconds to the corresponding index in the raw data."""
        return raw.time_as_index(time, use_rounding=True)[0]

    # Create a separate aligner for both videos
    aligner1 = RawVideoAligner(
        raw_timestamps=raw_timestamps_ms,
        video_timestamps=video1.timestamps_ms,
        raw_times=raw.times,
        raw_time_to_index=raw_time_to_index,
        timestamp_unit="milliseconds",
    )
    aligner2 = RawVideoAligner(
        raw_timestamps=raw_timestamps_ms,
        video_timestamps=video2.timestamps_ms,
        raw_times=raw.times,
        raw_time_to_index=raw_time_to_index,
        timestamp_unit="milliseconds",
    )

    # Start the browser.

    app = QApplication([])
    raw_browser = raw.plot(block=False, show=False)
    browser = SyncedRawVideoBrowser(raw_browser, [video1, video2], [aligner1, aligner2])
    app.exec_()  # Start the Qt event loop
