"""Run video browser in sync with mne raw data browser using mne sample data.

Adds an artificial stimulus channel to the raw data and creates a fake video
that shows markers when the stimulus is active.
"""

import mne
import numpy as np
import numpy.typing as npt
import scipy
from demo_utils import create_fake_video_with_markers
from mne.datasets import sample

from videomeg_browser import TimestampAligner, browse_raw_with_video


def _create_binary_stimulus_vector(
    sampling_freq: float,
    n_times: int,
    stimulus_len_seconds: float,
    stimulus_inteval_seconds: float,
) -> npt.NDArray[np.bool_]:
    """Create a square wave stimulus vector."""
    # Convert the stimulus length and interval from seconds to number of samples.
    fs = sampling_freq
    stimulus_len_samples = int(stimulus_len_seconds * fs)
    stimulus_interval_samples = int(stimulus_inteval_seconds * fs)

    stimulus_vec = np.zeros(n_times, dtype=np.bool_)
    # Mark the stimulus events as specified by the interval and length.
    for i in range(0, n_times, stimulus_interval_samples):
        stimulus_vec[i : i + stimulus_len_samples] = 1

    return stimulus_vec


def _add_stimulus_channel(raw: mne.io.Raw, stimulus_vec: npt.NDArray[np.bool_]) -> None:
    """Add an artificial stimulus channel to the raw data."""
    info = mne.create_info(
        ch_names=["artificial_stimulus"],
        sfreq=raw.info["sfreq"],
        ch_types="stim",
    )
    stimulus_raw = mne.io.RawArray(stimulus_vec[np.newaxis, :], info)
    raw.add_channels([stimulus_raw], force_update_info=True)


def _downsample_binary_stimulus_nearest_neighbor(
    stimulus: npt.NDArray[np.bool_],
    stimulus_times: npt.NDArray[np.float64],
    new_times: npt.NDArray[np.floating],
) -> npt.NDArray[np.bool_]:
    """Downsample a binary stimulus vector to match the new times."""
    interpolator = scipy.interpolate.make_interp_spline(stimulus_times, stimulus, k=0)
    downsampled_stimulus = interpolator(new_times)
    # Make sure that the downsampled stimulus is binary (0 or 1).
    assert np.all(np.isin(downsampled_stimulus, [0, 1]))

    return downsampled_stimulus.astype(np.bool_)


def main() -> None:
    """Run the demo with MNE sample data."""
    # The duration to which the raw data is cropped and the video is created
    DURATION_SECONDS = 60
    # The frames per second for the video
    VIDEO_FPS = 30.0
    # The length of the stimulus
    STIMULUS_LEN_SECONDS = 0.5
    # The interval between the beginning of the stimulus events
    STIMULUS_INTERVAL_SECONDS = 2

    # Load sample data from MNE.
    data_path = sample.data_path()
    raw_fname = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = mne.io.read_raw_fif(raw_fname, preload=True)
    raw.crop(tmax=DURATION_SECONDS)

    # Create artificial stimulus channel that will be used to mark frames in the video.
    # The stimulus will be a square wave with values 0 and 1.
    stimulus_vec = _create_binary_stimulus_vector(
        raw.info["sfreq"], raw.n_times, STIMULUS_LEN_SECONDS, STIMULUS_INTERVAL_SECONDS
    )
    # Add the stimulus channel to the raw data
    _add_stimulus_channel(raw, stimulus_vec)

    # Create a fake video file that has marked frames for stimulus events.
    video_frame_count = int(DURATION_SECONDS * VIDEO_FPS)
    video_times = np.linspace(0, DURATION_SECONDS, video_frame_count, endpoint=False)
    stimulus_mask = _downsample_binary_stimulus_nearest_neighbor(
        stimulus_vec, stimulus_times=raw.times, new_times=video_times
    )
    video = create_fake_video_with_markers(video_frame_count, stimulus_mask)

    # Create mapping between raw data points and video frames

    # Both raw times and video times go from 0 to DURATION_SECONDS, so we can use them
    # directly as synchronization timestamps.
    aligner = TimestampAligner(
        timestamps_a=raw.times,
        timestamps_b=video_times,
        timestamp_unit="seconds",
        name_a="raw",
        name_b="video",
    )

    # Launch the synced raw and video browsers.
    raw_browser = raw.plot(block=False, show=False)
    browse_raw_with_video(raw_browser, raw, [video], [aligner])


if __name__ == "__main__":
    main()
