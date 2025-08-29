from collections.abc import Callable
from typing import Literal

import numpy as np
import numpy.typing as npt
import pytest

from videomeg_browser.raw_media_aligner import (
    MapFailureReason,
    MappingFailure,
    MappingResult,
    MappingSuccess,
    TimestampAligner,
)


@pytest.mark.parametrize(
    "timestamps_start_time_ms, timestamps_end_time_ms, timestamps_step_ms",
    [
        (5000, 9000, 1000),  # 5 seconds to 9 seconds with 1 second step
        (1000, 2000, 100),  # 1 second to 2 seconds with 100 ms step
        (0, 5000, 500),  # 0 seconds to 5 seconds with 500 ms step
        (1234, 5678, 123),  # arbitrary start and end times with 123 ms step
    ],
)
def test_with_matching_timestamps(
    timestamps_start_time_ms: int, timestamps_end_time_ms: int, timestamps_step_ms: int
) -> None:
    """Test mapping with matching raw and video timestamps (only successes).

    When raw and video timestamps match exactly, the mapping should be one-to-one
    and always succeed.
    """
    # Video timesamps corresponding to each video frame in milliseconds
    video_timestamps_ms = np.arange(
        timestamps_start_time_ms,
        timestamps_end_time_ms + 1,  # +1 to include the end time
        timestamps_step_ms,
        dtype=np.float64,
    )
    # Raw timestamps corresponding to each raw data point in milliseconds
    raw_timestamps_ms = video_timestamps_ms.copy()

    aligner = TimestampAligner(
        raw_timestamps_ms,
        video_timestamps_ms,
    )

    # Test that mapping each raw index yields the correct video frame index.
    # As the timestamps match, each resulting video frame index should be the
    # same as the raw time index.
    for raw_idx in range(len(raw_timestamps_ms)):
        correct_video_idx = raw_idx
        mapping_to_media_idx = aligner.a_index_to_b_index(raw_idx)
        _assert_mapping_success(mapping_to_media_idx, expected_value=correct_video_idx)

    # Test the other way around, mapping video frame index to raw time.
    for video_frame_idx in range(len(video_timestamps_ms)):
        correct_raw_idx = video_frame_idx
        mapping_to_raw_time = aligner.b_index_to_a_index(video_frame_idx)
        _assert_mapping_success(mapping_to_raw_time, expected_value=correct_raw_idx)


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

    _run_alignment_test(video_timestamps_ms, raw_timestamps_ms, select_on_tie)


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

    _run_alignment_test(video_timestamps_ms, raw_timestamps_ms, select_on_tie)


def _run_alignment_test(
    video_timestamps_ms: npt.NDArray[np.floating],
    raw_timestamps_ms: npt.NDArray[np.floating],
    select_on_tie: Literal["left", "right"],
) -> None:
    """Run the alignment test with given video and raw timestamps and raw times.

    Tests mapping all the video frame indices (indices of `video_timestamps_ms`) to raw
    times and all `raw_times` to video frame indices.
    """
    # Create the aligner to test.
    aligner = TimestampAligner(
        raw_timestamps_ms,
        video_timestamps_ms,
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
        mapping_to_raw_idx = aligner.b_index_to_a_index(test_video_frame_idx)
        # Manually calculate the raw time closest to the video timestamp.
        closest_raw_idx = _find_closest_raw_idx(
            raw_timestamps_ms, video_ts, side=select_on_tie
        )
        # Assert that mapping result is either failure with appropriate reason or a
        # success with the closest raw time, depending on min and max valid timestamps.
        _assert_correct_mapping(
            source_ts=video_ts,
            min_valid_source_ts=video_timestamp_valid_range[0],
            max_valid_source_ts=video_timestamp_valid_range[1],
            mapping_result=mapping_to_raw_idx,
            result_if_success=closest_raw_idx,
        )

    # Do the same but vice versa: map each raw time to video frame index and check
    # correctness.
    for test_raw_idx, raw_ts in enumerate(raw_timestamps_ms):
        mapping_to_media_frame_idx = aligner.a_index_to_b_index(test_raw_idx)
        closest_video_frame_idx = _find_closest_video_frame_index(
            video_timestamps_ms, raw_ts, side=select_on_tie
        )
        _assert_correct_mapping(
            source_ts=raw_ts,
            min_valid_source_ts=raw_timestamp_valid_range[0],
            max_valid_source_ts=raw_timestamp_valid_range[1],
            mapping_result=mapping_to_media_frame_idx,
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


def _find_closest_raw_idx(
    raw_timestamps: npt.NDArray[np.floating],
    video_timestamp: float,
    side: Literal["left", "right"],
) -> int:
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

    return int(closest_raw_idx)


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
