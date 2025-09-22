"""Demonstrates browsing raw data along with two synchronized videos and audio.

Running this requires video and audio files recorded with Helsinki videoMEG project
software. File paths also need adjustment.
"""

import logging
import os.path as op

import mne
import numpy as np
from numpy.typing import NDArray

from videomeg_browser.audio import AudioFileHelsinkiVideoMEG
from videomeg_browser.comp_tstamps import comp_tstamps
from videomeg_browser.synced_raw_media_browser import browse_raw_with_video_and_audio
from videomeg_browser.timestamp_aligner import TimestampAligner
from videomeg_browser.video import VideoFileHelsinkiVideoMEG

BASE_PATH = (
    "/u/69/taivait1/unix/video_meg_testing/meg2meg_with_raw/2025-09-04--14-44-09_test_1"
)
# Assuming that base path contains all these files:
RAW_FNAME = "2025_09_04__14_44_09_MEG.fif"
VIDEO1_FNAME = "2025-09-04--14-44-09_video_01.vid"
VIDEO2_FNAME = "2025-09-04--14-44-09_video_02.vid"
AUDIO_FNAME = "2025-09-04--14-44-09_audio_00.aud"

# Channel in raw data that contains the timing information
RAW_TIMING_CHANNEL = "STI009"


def get_raw_timestamps(raw: mne.io.Raw, timing_channel: str) -> NDArray[np.floating]:
    """Get the timestamps from raw data that has Helsinki videoMEG timing channel."""
    timing_data = raw.get_data(picks=timing_channel, return_times=False)
    # Remove the channel dimension
    # Ignoring warning about timing_data possibly being tuple,
    # as we do not ask times from raw.get_data.
    timing_data = timing_data.squeeze()  # type: ignore
    return comp_tstamps(timing_data, raw.info["sfreq"])


def main() -> None:
    """Run the video and audio sync demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load MEG data.
    raw = mne.io.read_raw_fif(op.join(BASE_PATH, RAW_FNAME), preload=True)

    # Load videos and audio.
    video1 = VideoFileHelsinkiVideoMEG(op.join(BASE_PATH, VIDEO1_FNAME))
    video2 = VideoFileHelsinkiVideoMEG(op.join(BASE_PATH, VIDEO2_FNAME))
    audio = AudioFileHelsinkiVideoMEG(op.join(BASE_PATH, AUDIO_FNAME))
    audio.unpack_audio()

    # Print info about video and audio files.
    for video in [video1, video2]:
        video.print_stats()
    audio.print_stats()

    # Extract timestamps for videos, audio, and raw data.
    video1_timestamps_ms = video1.timestamps_ms
    video2_timestamps_ms = video2.timestamps_ms
    audio_timestamps_ms = audio.get_audio_timestamps_ms()
    raw_timestamps_ms = get_raw_timestamps(raw, RAW_TIMING_CHANNEL)

    # For some reason there are two audio timestamps that are smaller than the previous
    # one. Sort the timestamps to make them non-decreasing and avoid error in aligner
    # creation.
    audio_timestamps_ms.sort()

    # Use the timestamps to create aligners for videos and audio to sync them with
    # the MEG data.
    vid_aligner1 = TimestampAligner(
        timestamps_a=raw_timestamps_ms,
        timestamps_b=video1_timestamps_ms,
        timestamp_unit="milliseconds",
        name_a="raw",
        name_b="video1",
    )
    vid_aligner2 = TimestampAligner(
        timestamps_a=raw_timestamps_ms,
        timestamps_b=video2_timestamps_ms,
        timestamp_unit="milliseconds",
        name_a="raw",
        name_b="video2",
    )
    audio_aligner = TimestampAligner(
        timestamps_a=raw_timestamps_ms,
        timestamps_b=audio_timestamps_ms,
        timestamp_unit="milliseconds",
        name_a="raw",
        name_b="audio",
    )

    # Start the browser.
    raw_browser = raw.plot(block=False, show=False)
    browse_raw_with_video_and_audio(
        raw_browser,
        raw,
        [video1, video2],
        [vid_aligner1, vid_aligner2],
        audio,
        audio_aligner,
        video_splitter_orientation="vertical",
    )


if __name__ == "__main__":
    main()
