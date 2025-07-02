"""Contains a class for mapping between time points of raw data and video frames."""

import logging
from abc import ABC
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MapFailureReason(Enum):
    """Enum telling why mapping from frame index to raw time or vice versa failed."""

    # Index to map is smaller than the first frame or raw time point
    INDEX_TOO_SMALL = "index_too_small"
    # Index to map is larger than the last frame or raw time point
    INDEX_TOO_LARGE = "index_too_large"
    # A value that can be used as a placeholder before mapping is done
    NOT_MAPPED = "not_mapped"


class MappingResult(ABC):
    """Represents the result of mapping raw time to video frame index or vice versa."""

    pass


@dataclass(frozen=True)
class MappingSuccess(MappingResult):
    """Represents a successful mapping that yielded a raw time or video frame index."""

    result: int | float


@dataclass(frozen=True)
class MappingFailure(MappingResult):
    """Represents a failed mapping with a reason for the failure."""

    failure_reason: MapFailureReason


class TimeIndexMapper:
    """Maps time points from raw data to video frames and vice versa.

    Parameters
    ----------
    raw_timestamps : NDArray[np.floating]
        1-D sorted array of raw timestamps used for synchronization.
        In Helsinki VideoMEG project these are unix times in milliseconds.
    video_timestamps : NDArray[np.floating]
        1-D sorted array of video timestamps used for synchronization.
        In Helsinki VideoMEG project these are unix times in milliseconds.
    raw_times : NDArray[np.floating]
        1-D array of raw data times in seconds, used for converting raw indices
        to actual time points.
    raw_time_to_index : Callable[[float], int]
        A function that converts a raw time point in seconds to the corresponding
        index in the raw data. Required for converting arbitrary raw time
        to a discrete index in the raw data.
    timestamp_unit : Literal["milliseconds", "seconds"], optional
        The unit of the timestamps in `raw_timestamps` and `video_timestamps`.
        By default "milliseconds".
    """

    def __init__(
        self,
        raw_timestamps: NDArray[np.floating],
        video_timestamps: NDArray[np.floating],
        raw_times: NDArray[np.floating],
        raw_time_to_index: Callable[[float], int],
        timestamp_unit: Literal["milliseconds", "seconds"] = "milliseconds",
    ) -> None:
        self._raw_times = raw_times
        self._raw_time_to_index = raw_time_to_index
        self._timestamp_unit = timestamp_unit
        # Internally store timestamps in milliseconds.
        # Set _timestamp unit before calling _get_timestamps_in_milliseconds!
        self._raw_timestamps_ms = self._get_timestamps_in_milliseconds(raw_timestamps)
        self._video_timestamps_ms = self._get_timestamps_in_milliseconds(
            video_timestamps
        )

        self._validate_input_times()
        self._diagnose_timestamps()

        # Precompute mappings from raw indices to video frame indices
        # and from video frame indices to raw times.

        # NOTE: Video frame indices can be mapped straight to their corresponding
        # raw times as they are a dicrete set of values. But raw times are continuous,
        # so instead of precomputing mapping raw times --> video frame indices, we
        # precompute mapping raw indices --> video frame indices. And when asked to
        # convert a raw time to a video frame index, we do
        # raw time --> raw index --> video frame index.

        logger.info("Building mapping from raw indices to video frame indices.")
        self._raw_idx_to_video_frame_idx: list[MappingResult] = self._build_mapping(
            source_timestamps_ms=self._raw_timestamps_ms,
            target_timestamps_ms=self._video_timestamps_ms,
        )

        logger.info("Building mapping from video frame indices to raw times.")
        self._video_frame_idx_to_raw_time: list[MappingResult] = self._build_mapping(
            source_timestamps_ms=self._video_timestamps_ms,
            target_timestamps_ms=self._raw_timestamps_ms,
            convert_raw_results_to_seconds=True,
        )

        self._log_mapping_results(
            mapping_results=self._raw_idx_to_video_frame_idx,
            header="Mapping results from raw indices to video frame indices:",
        )
        self._log_mapping_results(
            mapping_results=self._video_frame_idx_to_raw_time,
            header="Mapping results from video frame indices to raw times:",
        )

    def raw_time_to_video_frame_index(self, raw_time_seconds: float) -> MappingResult:
        """Convert a time point from raw data (in seconds) to video frame index."""
        # Find the raw index that corresponds to the given time point.
        # We cannot use the given time directly, as it may not match exactly with raw
        # times.
        raw_idx = self._raw_time_to_index(raw_time_seconds)
        return self._raw_idx_to_video_frame_idx[raw_idx]

    def video_frame_index_to_raw_time(self, video_frame_idx: int) -> MappingResult:
        """Convert a video frame index to a raw data time point (in seconds)."""
        return self._video_frame_idx_to_raw_time[video_frame_idx]

    def _validate_input_times(self) -> None:
        if not np.all(np.diff(self._raw_timestamps_ms) >= 0):
            raise ValueError(
                "Raw timestamps are not strictly increasing. "
                "This is required for the mapping to work correctly."
            )
        if not np.all(np.diff(self._video_timestamps_ms) >= 0):
            raise ValueError(
                "Video timestamps are not strictly increasing. "
                "This is required for the mapping to work correctly."
            )
        if not len(self._raw_timestamps_ms) == len(self._raw_times):
            raise ValueError(
                "Length of raw timestamps does not match the length of raw times. "
                "This is required for the mapping to work correctly."
            )

    def _diagnose_timestamps(self) -> None:
        """Log some statistics about the raw and video timestamps."""
        # Convert to seconds for easier readability
        raw_timestamps_seconds = self._raw_timestamps_ms / 1000.0
        video_timestamps_seconds = self._video_timestamps_ms / 1000.0
        logger.info(
            f"Raw timestamps: {raw_timestamps_seconds[0]:.1f} s to "
            f"{raw_timestamps_seconds[-1]:.1f} s, "
            f"total {len(raw_timestamps_seconds)} timestamps."
        )
        logger.info(
            f"Video timestamps: {video_timestamps_seconds[0]:.1f} s to "
            f"{video_timestamps_seconds[-1]:.1f} s, "
            f"total {len(video_timestamps_seconds)} timestamps."
        )

        # Check the interval between timesamps
        raw_intervals_ms = np.diff(self._raw_timestamps_ms)
        logger.info(
            f"Raw timestamps intervals: min={np.min(raw_intervals_ms):.3f} ms, "
            f"max={np.max(raw_intervals_ms):.3f} ms, "
            f"mean={np.mean(raw_intervals_ms):.3f} ms, "
            f"std={np.std(raw_intervals_ms):.3f} ms"
        )
        video_intervals_ms = np.diff(self._video_timestamps_ms)
        logger.info(
            f"Video timestamps intervals: min={np.min(video_intervals_ms):.3f} ms, "
            f"max={np.max(video_intervals_ms):.3f} ms, "
            f"mean={np.mean(video_intervals_ms):.3f} ms, "
            f"std={np.std(video_intervals_ms):.3f} ms"
        )

        # Count timestamps that are out of bounds
        video_too_small_count = np.sum(
            self._video_timestamps_ms < self._raw_timestamps_ms[0]
        )
        video_too_large_count = np.sum(
            self._video_timestamps_ms > self._raw_timestamps_ms[-1]
        )
        logger.info(
            "Video timestamps smaller/larger than first/last raw timestamp: "
            f"{video_too_small_count}/{video_too_large_count}"
        )
        raw_too_small_count = np.sum(
            self._raw_timestamps_ms < self._video_timestamps_ms[0]
        )
        raw_too_large_count = np.sum(
            self._raw_timestamps_ms > self._video_timestamps_ms[-1]
        )
        logger.info(
            "Raw timestamps smaller/larger than first/last video timestamp: "
            f"{raw_too_small_count}/{raw_too_large_count}"
        )
        first_timestamp_diff_ms = (
            self._raw_timestamps_ms[0] - self._video_timestamps_ms[0]
        )
        logger.info(
            "Difference between first raw and video timestamps: "
            f"{first_timestamp_diff_ms:.3f} ms"
        )
        if first_timestamp_diff_ms > 1000:
            logger.warning(
                "The raw data timestamps start over a second later than the video "
                "timestamps."
            )
        elif first_timestamp_diff_ms < -1000:
            logger.warning(
                "Video timestamps start over a second later than the raw data "
                "timestamps."
            )
        last_timestamp_diff_ms = (
            self._raw_timestamps_ms[-1] - self._video_timestamps_ms[-1]
        )
        logger.info(
            "Difference between last raw and video timestamps: "
            f"{last_timestamp_diff_ms:.3f} ms"
        )
        if last_timestamp_diff_ms > 1000:
            logger.warning(
                "The video timestamps end over a second earlier than the raw data "
                "timestamps."
            )
        elif last_timestamp_diff_ms < -1000:
            logger.warning(
                "Raw data timestamps end over a second earlier than the video "
                "timestamps."
            )

    def _get_timestamps_in_milliseconds(
        self, timestamps: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Convert timestamps to milliseconds if they are in seconds."""
        if self._timestamp_unit == "milliseconds":
            return timestamps
        elif self._timestamp_unit == "seconds":
            return timestamps * 1000.0
        else:
            raise ValueError(
                f"Unknown timestamp unit: {self._timestamp_unit}. "
                "Expected 'milliseconds' or 'seconds'."
            )

    def _get_timestamp_unit_symbol(self) -> str:
        """Get the symbol for the timestamp unit."""
        if self._timestamp_unit == "milliseconds":
            return "ms"
        elif self._timestamp_unit == "seconds":
            return "s"
        else:
            raise ValueError(
                f"Unknown timestamp unit: {self._timestamp_unit}. "
                "Expected 'milliseconds' or 'seconds'."
            )

    def _find_indices_with_closest_values(
        self, source_times: NDArray[np.floating], target_times: NDArray[np.floating]
    ) -> NDArray[np.intp]:
        """Find indices of the closest target times for each source time.

        Parameters
        ----------
        source_times : NDArray[np.floating]
            I-D sorted array of source times
        target_times : NDArray[np.floating]
            I-D sorted array of target times

        Returns
        -------
        NDArray[np.intp]
            I-D array consisting of the indices of the closest target times
            for each source time.
        """
        # Find the indices where each source time would fit in the target array.
        insert_indices = np.searchsorted(target_times, source_times)
        # Ensure that the indices are within bounds
        insert_indices = np.clip(insert_indices, 1, len(target_times) - 1)

        # Get the target times around the insert position
        left_target = target_times[insert_indices - 1]
        right_target = target_times[insert_indices]

        # Calculate distances to the left and right target times
        left_distances = np.abs(source_times - left_target)
        right_distances = np.abs(source_times - right_target)

        # Choose the closest target time
        closest_indices = np.where(
            left_distances < right_distances, insert_indices - 1, insert_indices
        )

        return closest_indices

    def _log_mapping_errors(self, errors_ms: NDArray[np.floating]) -> None:
        """Log statistics about the distances between source and target timestamps."""
        logger.info(
            "    Statistics for mapping error (distances between source timestamps "
            "and their closest target timestamps):"
        )
        logger.info(
            f"    min={np.min(errors_ms):.3f} ms, "
            f"    max={np.max(errors_ms):.3f} ms, mean={np.mean(errors_ms):.3f} "
            f"    ms, std={np.std(errors_ms):.3f} ms"
        )
        if np.any(errors_ms < 0):
            logger.warning("Some distances between timestamps are negative.")
        if np.any(np.isnan(errors_ms)):
            logger.warning("Some distances between timestamps are NaN.")

    def _build_mapping(
        self,
        source_timestamps_ms: NDArray[np.floating],
        target_timestamps_ms: NDArray[np.floating],
        convert_raw_results_to_seconds: bool = False,
    ) -> list[MappingResult]:
        """Build a mapping from raw indices to video frame indices or vice versa.

        Parameters
        ----------
        source_timestamps_ms : NDArray[np.floating]
            I-D sorted array of source timestamps in milliseconds for which to
            compute the mapping.
        target_timestamps : NDArray[np.floating]
            I-D sorted array of target timestamps in milliseconds to which
            the source timestamps should be mapped.
        convert_raw_results_to_seconds : bool, optional
            If true, assume that the mapping is from video frame indices to raw indices,
            and convert the resulting raw indices to seconds.

        Returns
        -------
        list[MappingResult]
            List of mapping results, where each result corresponds to a source
            timestamp. If a source timestamp is out of bounds of the target timestamps,
            it will be marked as a failure with a reason for the failure (index too
            small or index too large).
        """
        # Initialize a list of mapping results with failures
        mapping: list[MappingResult] = [
            MappingFailure(MapFailureReason.NOT_MAPPED)
            for _ in range(len(source_timestamps_ms))
        ]

        # Find indices of source timestamps that are out of bounds
        # of the target timestamps.
        too_small_mask = source_timestamps_ms < target_timestamps_ms[0]
        too_small_source_indices = too_small_mask.nonzero()[0]
        too_large_mask = source_timestamps_ms > target_timestamps_ms[-1]
        too_large_source_indices = too_large_mask.nonzero()[0]

        # Add these to the mapping as failures
        for source_idx in too_small_source_indices:
            mapping[source_idx] = MappingFailure(
                failure_reason=MapFailureReason.INDEX_TOO_SMALL
            )
        for source_idx in too_large_source_indices:
            mapping[source_idx] = MappingFailure(
                failure_reason=MapFailureReason.INDEX_TOO_LARGE
            )

        # Map the rest of source timestamps to the closest target timestamps.

        valid_mask = ~(too_small_mask | too_large_mask)
        valid_source_timestamps_ms = source_timestamps_ms[valid_mask]
        valid_source_indices = valid_mask.nonzero()[0]

        closest_target_indices = self._find_indices_with_closest_values(
            source_times=valid_source_timestamps_ms, target_times=target_timestamps_ms
        )
        # Log mapping errors.
        errors_ms = np.abs(
            valid_source_timestamps_ms - target_timestamps_ms[closest_target_indices]
        )
        self._log_mapping_errors(errors_ms)

        if convert_raw_results_to_seconds:
            # Convert the raw indices to actual time points in seconds
            closest_raw_times = self._raw_times[closest_target_indices]
            mapping_results = closest_raw_times
        else:
            # The results are the plain indices of the target timestamps
            mapping_results = closest_target_indices

        for source_idx, result in zip(valid_source_indices, mapping_results):
            mapping[source_idx] = MappingSuccess(result=result)

        # Make sure that all indices were filled.
        for mapping_result in mapping:
            match mapping_result:
                case MappingFailure(failure_reason=MapFailureReason.NOT_MAPPED):
                    raise AssertionError(
                        "Not all source indices were mapped to target indices."
                    )
                case _:
                    pass

        return mapping

    def _log_mapping_results(
        self, mapping_results: list[MappingResult], header: str
    ) -> None:
        """Log the number of each mapping result for debugging purposes."""
        result_counts = self._count_mapping_results(mapping_results)
        logger.debug(f"{header}")
        for result, count in result_counts.items():
            logger.debug(f"    {result}: {count}")

    def _count_mapping_results(
        self, mapping_results: list[MappingResult]
    ) -> Counter[str]:
        """Count the number of each mapping results for debugging purposes."""
        counts = Counter()

        for mapping_result in mapping_results:
            match mapping_result:
                case MappingSuccess():
                    key = "MappingSuccess"
                case MappingFailure(failure_reason=reason):
                    key = f"MappingFailure({reason.name})"
                case _:
                    raise ValueError(f"Unexpected mapping result: {mapping_result}")
            counts[key] += 1

        return counts
