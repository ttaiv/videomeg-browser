from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest

from videomeg_browser.raw_video_aligner import (
    MapFailureReason,
    MappingFailure,
    MappingResult,
    MappingSuccess,
    RawVideoAligner,
)


def test_with_matching_timestamps() -> None:
    """Test mapping with matching raw and video timestamps (only successes).

    When raw and video timestamps match exactly, the mapping should be one-to-one
    and always succeed.
    """
    # All in milliseconds
    timestamps_start_time = 5000
    timestamps_end_time = 9000
    timestamps_step = 1000
    # Video timesamps corresponding to each video frame in milliseconds
    video_timestamps_ms = np.arange(
        timestamps_start_time,
        timestamps_end_time + 1,
        timestamps_step,
        dtype=np.float64,
    )
    # Raw timestamps corresponding to each raw data point in milliseconds
    raw_timestamps_ms = np.arange(
        timestamps_start_time,
        timestamps_end_time + 1,
        timestamps_step,
        dtype=np.float64,
    )

    # These could be anything just as long there as as many as the raw timestamps.
    raw_times = np.arange(len(raw_timestamps_ms), dtype=np.float64)
    raw_time_to_index = _make_simple_raw_time_to_index_function(raw_times)

    aligner = RawVideoAligner(
        raw_timestamps_ms,
        video_timestamps_ms,
        raw_times=raw_times,
        raw_time_to_index=raw_time_to_index,
    )

    # Test that mapping each raw time yields the correct video frame index.
    # As the timestamps match, each resulting video frame index should be the
    # same as the raw time index.
    for correct_video_idx, raw_time in enumerate(raw_times):
        mapping_to_video_idx = aligner.raw_time_to_video_frame_index(raw_time)
        _assert_mapping_success(mapping_to_video_idx, expected_value=correct_video_idx)

    # Test the other way around, mapping video frame index to raw time.
    for video_frame_idx, correct_raw_time in enumerate(raw_times):
        mapping_to_raw_time = aligner.video_frame_index_to_raw_time(video_frame_idx)
        _assert_mapping_success(mapping_to_raw_time, expected_value=correct_raw_time)


@pytest.mark.parametrize(
    "video_fps, raw_sfreq, duration_seconds, select_on_tie, timestamp_start_time_ms",
    [
        (
            30.0,  # 30 fps video
            1000.0,  # 1000 Hz raw data
            10,  # for 10 seconds
            "left",  # select left timestamp when distance is equal
            7000.0,  # start time of timestamps in milliseconds
        ),
        (
            60.0,  # 60 fps video
            1000.0,  # 1000 Hz raw data
            5,  # for 5 seconds
            "right",  # select right timestamp when distance is equal
            1.1,  # start time of timestamps in milliseconds
        ),
        (
            29.95,  # fps,
            1100.0,  # Hz
            15,  # for 15 seconds
            "left",
            999.9,  # ms
        ),
        (
            90,  # fps
            600,  # Hz
            20,  # for 20 seconds
            "right",
            0.0,  # ms
        ),
    ],
)
def test_with_constant_interval_timestamps(
    video_fps: float,
    raw_sfreq: float,
    duration_seconds: int,
    select_on_tie: Literal["left", "right"],
    timestamp_start_time_ms: float,  # this should not matter for the test
) -> None:
    """Test mapping with simulated constant interval raw and video timestamps."""
    # Simulate raw and video timestamps in milliseconds.
    video_timestamps_ms = np.linspace(
        timestamp_start_time_ms,
        timestamp_start_time_ms + duration_seconds * 1000,
        num=int(video_fps * duration_seconds),  # number of frames in video
        endpoint=False,
    )
    raw_timestamps_ms = np.linspace(
        timestamp_start_time_ms,
        timestamp_start_time_ms + duration_seconds * 1000,
        num=int(raw_sfreq * duration_seconds),  # number of raw samples
        endpoint=False,
    )
    # Create raw times that start from zero and correspond to sampling frequency.
    raw_times = np.linspace(
        0, duration_seconds, num=int(raw_sfreq * duration_seconds), endpoint=False
    )

    _run_alignment_test(
        video_timestamps_ms, raw_timestamps_ms, raw_times, select_on_tie
    )


@pytest.mark.parametrize(
    "video_fps, raw_sfreq, duration_seconds, select_on_tie, timestamp_start_time_ms",
    [
        (
            30.0,  # 30 fps video
            1000.0,  # 1000 Hz raw data
            10,  # for 10 seconds
            "left",  # select left timestamp when distance is equal
            7000.0,  # start time of timestamps in milliseconds
        ),
        (
            60.0,  # 60 fps video
            1000.0,  # 1000 Hz raw data
            5,  # for 5 seconds
            "right",  # select right timestamp when distance is equal
            1.1,  # start time of timestamps in milliseconds
        ),
        (
            29.95,  # fps,
            1100.0,  # Hz
            15,  # for 15 seconds
            "left",
            999.9,  # ms
        ),
        (
            90,  # fps
            600,  # Hz
            20,  # for 20 seconds
            "right",
            0.0,  # ms
        ),
    ],
)
def test_with_random_timestamps(
    video_fps: float,
    raw_sfreq: float,
    duration_seconds: int,
    select_on_tie: Literal["left", "right"],
    timestamp_start_time_ms: float,  # this should not matter for the test
) -> None:
    """Test mapping with random raw and video timestamps.

    Ensures that the mapping works even when timestamps are just some strictly
    increasing numbers that are not evenly spaced.
    """
    rng = np.random.default_rng(42)  # for reproducibility
    raw_timestamps_ms = np.sort(
        rng.uniform(
            low=timestamp_start_time_ms,
            high=timestamp_start_time_ms + duration_seconds * 1000,
            size=int(raw_sfreq * duration_seconds),
        )
    )
    video_timestamps_ms = np.sort(
        rng.uniform(
            low=timestamp_start_time_ms,
            high=timestamp_start_time_ms + duration_seconds * 1000,
            size=int(video_fps * duration_seconds),
        )
    )
    assert np.all(np.diff(raw_timestamps_ms) > 0), (
        "Raw timestamps must be strictly increasing. Fix the test."
    )
    assert np.all(np.diff(video_timestamps_ms) > 0), (
        "Video timestamps must be strictly increasing. Fix the test."
    )

    # Create raw times that start from zero and correspond to sampling frequency.
    raw_times = np.linspace(
        0, duration_seconds, num=int(raw_sfreq * duration_seconds), endpoint=False
    )

    _run_alignment_test(
        video_timestamps_ms, raw_timestamps_ms, raw_times, select_on_tie
    )


def _run_alignment_test(
    video_timestamps_ms: npt.NDArray[np.floating],
    raw_timestamps_ms: npt.NDArray[np.floating],
    raw_times: npt.NDArray[np.floating],
    select_on_tie: Literal["left", "right"],
) -> None:
    """Run the alignment test with given video and raw timestamps and raw times.

    Tests mapping all the video frame indices (indices of `video_timestamps_ms`) to raw
    times and all `raw_times` to video frame indices.
    """
    raw_time_to_index = _make_simple_raw_time_to_index_function(raw_times)

    # Create the aligner to test.
    aligner = RawVideoAligner(
        raw_timestamps_ms,
        video_timestamps_ms,
        raw_times=raw_times,
        raw_time_to_index=raw_time_to_index,
        select_on_tie=select_on_tie,
    )

    # Calculate range to which video and raw timestamps must fall in order to yield
    # successful mappings.
    video_timestamp_valid_range = _get_valid_source_timestamp_thresholds(
        target_timestamps=raw_timestamps_ms
    )
    raw_timestamp_valid_range = _get_valid_source_timestamp_thresholds(
        target_timestamps=video_timestamps_ms
    )

    # Test mapping each video frame index to a raw time and assert correct mapping.
    for test_video_frame_idx, video_ts in enumerate(video_timestamps_ms):
        # Use the aligner to get a raw time.
        mapping_to_raw_time = aligner.video_frame_index_to_raw_time(
            test_video_frame_idx
        )
        # Manually calculate the raw time closest to the video timestamp.
        closest_raw_time = _find_closest_raw_time(
            raw_timestamps_ms, video_ts, raw_times, side=select_on_tie
        )
        # Assert that mapping result is either failure with appropriate reason or a
        # success with the closest raw time, depending on min and max valid timestamps.
        _assert_correct_mapping(
            source_ts=video_ts,
            min_valid_source_ts=video_timestamp_valid_range[0],
            max_valid_source_ts=video_timestamp_valid_range[1],
            mapping_result=mapping_to_raw_time,
            result_if_success=closest_raw_time,
        )

    # Do the same but vice versa: map each raw time to video frame index and check
    # correctness.
    for test_raw_time, raw_ts in zip(raw_times, raw_timestamps_ms):
        mapping_to_video_frame_idx = aligner.raw_time_to_video_frame_index(
            test_raw_time
        )
        closest_video_frame_idx = _find_closest_video_frame_index(
            video_timestamps_ms, raw_ts, side=select_on_tie
        )
        _assert_correct_mapping(
            source_ts=raw_ts,
            min_valid_source_ts=raw_timestamp_valid_range[0],
            max_valid_source_ts=raw_timestamp_valid_range[1],
            mapping_result=mapping_to_video_frame_idx,
            result_if_success=closest_video_frame_idx,
        )


def _find_closest_video_frame_index(
    video_timestamps: npt.NDArray[np.floating],
    raw_timestamp: float,
    side: Literal["left", "right"],
) -> int:
    """Find the video index closest to a raw timestamp to be used as ground truth."""
    if side == "left":
        # Use argmin nomally, as it returns the first occurrence of the minimum value.
        closest_video_frame_idx = int(
            np.argmin(np.abs(video_timestamps - raw_timestamp))
        )
    elif side == "right":
        # Use argmin on a reversed array to get the rightmost occurrence of the minimum.
        reversed_idx = np.argmin(np.abs(video_timestamps[::-1] - raw_timestamp))
        closest_video_frame_idx = len(video_timestamps) - 1 - reversed_idx
    else:
        raise ValueError(f"Invalid side '{side}'. Use 'left' or 'right'.")

    return int(closest_video_frame_idx)


def _find_closest_raw_time(
    raw_timestamps: npt.NDArray[np.floating],
    video_timestamp: float,
    raw_times: npt.NDArray[np.floating],
    side: Literal["left", "right"],
) -> float:
    """Find the raw time closest to a video timestamp to be used as ground truth."""
    if side == "left":
        # Use argmin nomally, as it returns the first occurrence of the minimum value.
        closest_raw_idx = int(np.argmin(np.abs(raw_timestamps - video_timestamp)))
    elif side == "right":
        # Use argmin on a reversed array to get the rightmost occurrence of the minimum.
        reversed_idx = np.argmin(np.abs(raw_timestamps[::-1] - video_timestamp))
        closest_raw_idx = len(raw_timestamps) - 1 - reversed_idx
    else:
        raise ValueError(f"Invalid side '{side}'. Use 'left' or 'right'.")

    closest_raw_time = raw_times[closest_raw_idx]
    return closest_raw_time


def _assert_correct_mapping(
    source_ts: float,
    min_valid_source_ts: float,
    max_valid_source_ts: float,
    mapping_result: MappingResult,
    result_if_success: int | float,
) -> None:
    """Assert that mapping of raw time to video frame index or vice versa is correct.

    Parameters
    ----------
    source_ts : float
        The raw/video timestamp that corresponds to the raw time or video frame index
        that was mapped.
    min_valid_source_ts : float
        The minimum valid source timestamp. If `source_ts` is smaller than this, the
        mapping should fail with `MapFailureReason.INDEX_TOO_SMALL`.
    max_valid_source_ts : float
        The maximum valid source timestamp. If `source_ts` is larger than this, the
        mapping should fail with `MapFailureReason.INDEX_TOO_LARGE`.
    mapping_result : MappingResult
        The result of the mapping operation that will be tested.
    result_if_success : int | float
        The expected result of the mapping if it is successful.
    """
    if source_ts < min_valid_source_ts:
        _assert_mapping_failure(
            mapping_result,
            expected_failure_reason=MapFailureReason.INDEX_TOO_SMALL,
        )
    elif source_ts > max_valid_source_ts:
        _assert_mapping_failure(
            mapping_result,
            expected_failure_reason=MapFailureReason.INDEX_TOO_LARGE,
        )
    else:
        _assert_mapping_success(mapping_result, expected_value=result_if_success)


def _make_simple_raw_time_to_index_function(
    raw_times: npt.NDArray[np.floating],
) -> Callable[[float], int]:
    """Return a function that maps raw time to its index in `raw_times`."""

    def raw_time_to_index(time: float) -> int:
        """Convert a time in seconds to the corresponding index in the raw data."""
        # Assume that all input raw times are directly found in the raw timestamps.
        matches = (raw_times == time).nonzero()[0]
        if len(matches) == 0:
            raise ValueError(f"Time {time} not found in `raw_times`")
        return matches[0]  # return first matching index

    return raw_time_to_index


def _assert_mapping_success(
    mapping_result: MappingResult, expected_value: float | int
) -> None:
    """Assert that the mapping result is a success and matches the expected value."""
    assert isinstance(mapping_result, MappingSuccess), (
        "Mapping result should be a success."
    )
    assert mapping_result.result == expected_value, (
        f"Expected {expected_value}, but got {mapping_result.result}"
    )


def _assert_mapping_failure(
    mapping_result: MappingResult, expected_failure_reason: MapFailureReason
) -> None:
    """Assert that mapping result is failure and reason for failure matches expected."""
    assert isinstance(mapping_result, MappingFailure), (
        "Mapping result should be a failure."
    )
    assert mapping_result.failure_reason == expected_failure_reason, (
        f"Expected {expected_failure_reason}, but got {mapping_result.failure_reason}."
    )


def _get_valid_source_timestamp_thresholds(
    target_timestamps: npt.NDArray[np.floating],
) -> tuple[float, float]:
    """Return the minimum and maximum value for valid source timestamps.

    A valid source timestamp is a timestamp that yields MappingSuccess instead of
    MappingFailure.
    """
    target_interval = np.diff(target_timestamps).mean()
    return (
        target_timestamps[0] - target_interval / 2,
        target_timestamps[-1] + target_interval / 2,
    )
