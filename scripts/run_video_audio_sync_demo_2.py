"""Demonstrates browsing raw data along with two synchronized videos and audio.

Running this requires video and audio files recorded with Helsinki videoMEG project
software. File paths also need adjustment.
"""

import logging
import os.path as op

import mne
import numpy as np
from mne.datasets import sample
from qtpy.QtWidgets import QApplication

from videomeg_browser.audio import AudioFileHelsinkiVideoMEG
from videomeg_browser.raw_media_aligner import RawMediaAligner
from videomeg_browser.synced_raw_media_browser import browse_raw_with_video_and_audio
from videomeg_browser.video import VideoFileHelsinkiVideoMEG

BASE_PATH = "/u/69/taivait1/unix/video_meg_testing/2025-07-11_MEG2MEG_test/"


def main() -> None:
    """Run the video and audio sync demo."""
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

    # Load videos and audio.
    video1 = VideoFileHelsinkiVideoMEG(
        op.join(BASE_PATH, "2025-07-11--18-18-41_video_01.vid")
    )
    video2 = VideoFileHelsinkiVideoMEG(
        op.join(BASE_PATH, "2025-07-11--18-18-41_video_02.vid")
    )
    audio = AudioFileHelsinkiVideoMEG(
        op.join(BASE_PATH, "2025-07-11--18-18-41_audio_00.aud")
    )
    audio.unpack_audio()

    for video in [video1, video2]:
        video.print_stats()
    audio.print_stats()

    # Extract video and audio timestamps.
    video1_timestamps_ms = video1.timestamps_ms
    video2_timestamps_ms = video2.timestamps_ms
    audio_timestamps_ms = audio.get_audio_timestamps_ms()

    # Create artificial timestamps for raw data.
    start_ts = audio_timestamps_ms[0]  # Start at the first audio timestamp
    end_ts = start_ts + 60 * 1000  # End at 60 seconds later (convert to milliseconds)
    raw_timestamps_ms = np.linspace(start_ts, end_ts, raw.n_times, endpoint=False)

    # Define function for converting raw time to index
    def raw_time_to_index(time: float) -> int:
        """Convert a time in seconds to the corresponding index in the raw data."""
        return raw.time_as_index(time, use_rounding=True)[0]

    # Create an aligner for each video and audio.
    vid_aligner1 = RawMediaAligner(
        raw_timestamps=raw_timestamps_ms,
        media_timestamps=video1_timestamps_ms,
        raw_times=raw.times,
        raw_time_to_index=raw_time_to_index,
        timestamp_unit="milliseconds",
    )
    vid_aligner2 = RawMediaAligner(
        raw_timestamps=raw_timestamps_ms,
        media_timestamps=video2_timestamps_ms,
        raw_times=raw.times,
        raw_time_to_index=raw_time_to_index,
        timestamp_unit="milliseconds",
    )
    audio_aligner = RawMediaAligner(
        raw_timestamps=raw_timestamps_ms,
        media_timestamps=audio_timestamps_ms,
        raw_times=raw.times,
        raw_time_to_index=raw_time_to_index,
        timestamp_unit="milliseconds",
    )

    # Start the browser.
    app = QApplication([])
    raw_browser = raw.plot(block=False, show=False)
    browser = browse_raw_with_video_and_audio(
        raw_browser,
        [video1, video2],
        [vid_aligner1, vid_aligner2],
        audio,
        audio_aligner,
        video_splitter_orientation="vertical",
    )
    app.exec_()


if __name__ == "__main__":
    main()
