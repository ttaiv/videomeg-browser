"""Inspect MEG data and video recorded with Helsinki videoMEG project software.

Running this requires MEG data and video files recorded with the Helsinki videoMEG
project software. You also need to adjust file paths.
"""

import logging
import os.path as op

import mne
import numpy as np
from numpy.typing import NDArray

from videomeg_browser import (
    TimestampAligner,
    VideoFileHelsinkiVideoMEG,
    browse_raw_with_video,
    comp_tstamps,
)


def get_raw_timestamps(raw: mne.io.Raw, timing_channel: str) -> NDArray[np.floating]:
    """Get the timestamps from raw data having Helsinki videoMEG timing channel."""
    timing_data = raw.get_data(picks=timing_channel, return_times=False)
    # Remove the channel dimension
    # Ignoring warning about timing_data possibly being tuple,
    # as we do not ask times from raw.get_data.
    timing_data = timing_data.squeeze()  # type: ignore
    return comp_tstamps(timing_data, raw.info["sfreq"])


def main() -> None:
    """Run the demo."""
    BASE_PATH = "/u/69/taivait1/unix/video_meg_testing/Subject_2_Luna"
    RAW_TIMING_CHANNEL = "STI016"

    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [%(levelname)s] %(name)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create a video file object
    video_file = VideoFileHelsinkiVideoMEG(
        op.join(BASE_PATH, "Video_MEG", "animal_meg_subject_2_240614.video.dat"),
        magic_str="ELEKTA_VIDEO_FILE",
    )
    video_file.print_stats()

    # Create a raw data object
    raw = mne.io.read_raw_fif(
        op.join(BASE_PATH, "Raw", "animal_meg_subject_2_240614.fif"), preload=True
    )

    # Extract raw and video timestamps
    raw_timestamps_ms = get_raw_timestamps(raw, RAW_TIMING_CHANNEL)
    video_timestamps_ms = video_file.timestamps_ms

    # Set up mapping between raw data points and video frames
    aligner = TimestampAligner(
        raw_timestamps_ms,
        video_timestamps_ms,
        name_a="raw",
        name_b="video",
    )

    # Instantiate the browsers.
    raw_browser = raw.plot(block=False, show=False)
    browse_raw_with_video(
        raw_browser,
        raw,
        [video_file],
        [aligner],
    )
    video_file.close()


if __name__ == "__main__":
    main()
