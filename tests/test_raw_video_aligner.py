import numpy as np
import pytest

from videomeg_browser.raw_video_aligner import (
    MapFailureReason,
    MappingFailure,
    MappingResult,
    MappingSuccess,
    RawVideoAligner,
)


def assert_mapping_success(
    mapping_result: MappingResult, expected_value: float | int
) -> None:
    """Assert that the mapping result is a success and matches the expected value."""
    assert isinstance(mapping_result, MappingSuccess), "Mapping should be successful"
    assert mapping_result.result == expected_value, (
        f"Expected {expected_value}, but got {mapping_result.result}"
    )


def test_with_matching_timestamps() -> None:
    """Test mapping with matching raw and video timestamps."""
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

    def raw_time_to_index(time: float) -> int:
        """Convert a time in seconds to the corresponding index in the raw data."""
        # Assume that all input raw times are directly found in the raw timestamps.
        return (raw_times == time).nonzero()[0][0]  # return first matching index

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
        mapping_result = aligner.raw_time_to_video_frame_index(raw_time)
        assert_mapping_success(mapping_result, expected_value=correct_video_idx)

    # Test the other way around, mapping video frame index to raw time.
    for video_frame_idx, correct_raw_time in enumerate(raw_times):
        mapping_result = aligner.video_frame_index_to_raw_time(video_frame_idx)
        assert_mapping_success(mapping_result, expected_value=correct_raw_time)
