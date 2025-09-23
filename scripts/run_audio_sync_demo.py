"""Run audio browser in sync with MNE raw data browser.

Running this requires audio file recorded with the Helsinki videoMEG project software.
You also need to adjust file paths. Used MEG data is from MNE sample dataset.
"""

import logging

import mne
import numpy as np
from mne.datasets import sample

from videomeg_browser import (
    AudioFileHelsinkiVideoMEG,
    TimestampAligner,
    browse_raw_with_audio,
)

# Replace this with the path to your audio file.
AUDIO_PATH = (
    "/u/69/taivait1/unix/video_meg_testing/2025-07-11_MEG2MEG_test/"
    "2025-07-11--18-18-41_audio_00.aud"
)


def main() -> None:
    """Run the demo."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] [%(levelname)s] %(name)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Load sample raw data.
    data_path = sample.data_path()
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.crop(tmax=60)  # Crop to the first 60 seconds.

    # Load the audio.
    audio = AudioFileHelsinkiVideoMEG(AUDIO_PATH)
    audio.unpack_audio()
    audio.print_stats()

    # Extract timestamps for each audio sample.
    audio_timestamps_ms = audio.get_audio_timestamps_ms()

    # Create artificial timestamps for raw data.
    start_ts = audio_timestamps_ms[0]  # Start at the first audio timestamp
    end_ts = start_ts + 60 * 1000  # End at 60 seconds later (convert to milliseconds)
    raw_timestamps_ms = np.linspace(start_ts, end_ts, raw.n_times, endpoint=False)

    # Align the raw data with the audio.
    aligner = TimestampAligner(
        timestamps_a=raw_timestamps_ms,
        timestamps_b=audio_timestamps_ms,
        timestamp_unit="milliseconds",
        name_a="raw",
        name_b="audio",
    )

    # Start the synced raw and audio browsers.
    raw_browser = raw.plot(block=False, show=False)
    browse_raw_with_audio(raw_browser, raw, audio, aligner)


if __name__ == "__main__":
    main()
