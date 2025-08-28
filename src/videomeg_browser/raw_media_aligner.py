"""Contains a class for mapping between time points of raw data and media indices."""

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
    """Enum telling why mapping from media index to raw time or vice versa failed."""

    # Index to map is too small
    INDEX_TOO_SMALL = "index_too_small"
    # Index to map is too large
    INDEX_TOO_LARGE = "index_too_large"


class MappingResult(ABC):
    """Represents the result of mapping raw time to media frame index or vice versa."""

    pass


@dataclass(frozen=True)
class MappingSuccess(MappingResult):
    """Represents a successful mapping that yielded a raw time or media frame index."""

    result: int | float


@dataclass(frozen=True)
class MappingFailure(MappingResult):
    """Represents a failed mapping with a reason for the failure."""

    failure_reason: MapFailureReason


class RawMediaAligner:
    """Maps time points from raw data to media frames and vice versa.

    Uses the provided timestamps to simply find the closest matching time.

    Parameters
    ----------
    raw_timestamps : NDArray[np.floating]
        1-D sorted array of raw timestamps used for synchronization.
        In Helsinki VideoMEG project these are unix times in milliseconds.
    media_timestamps : NDArray[np.floating]
        1-D sorted array of media timestamps used for synchronization.
        In Helsinki VideoMEG project these are unix times in milliseconds.
    timestamp_unit : Literal["milliseconds", "seconds"], optional
        The unit of the timestamps in `raw_timestamps` and `media_timestamps`.
        By default "milliseconds".
    select_on_tie : Literal["left", "right"], optional
        How to select the result when a source timestamp is exactly between two target
        timestamps. If "left", the raw time or media frame index corresponding
        to the left target timestamp is selected. If "right", the one corresponding
        to the right target timestamp is selected. By default "left".
    """

    def __init__(
        self,
        raw_timestamps: NDArray[np.floating],
        media_timestamps: NDArray[np.floating],
        timestamp_unit: Literal["milliseconds", "seconds"] = "milliseconds",
        select_on_tie: Literal["left", "right"] = "left",
    ) -> None:
        self._timestamp_unit = timestamp_unit
        self._select_on_tie = select_on_tie
        # Internally store timestamps in milliseconds.
        self._raw_timestamps_ms = self._get_timestamps_in_milliseconds(raw_timestamps)
        self._media_timestamps_ms = self._get_timestamps_in_milliseconds(
            media_timestamps
        )

        self._validate_input_times()
        self._diagnose_timestamps()

        # Precompute mappings from raw indices to media frame indices
        # and from media frame indices to raw times.

        # NOTE: Media indices can be mapped straight to their corresponding
        # raw times as they are a discrete set of values. But raw times are continuous,
        # so instead of precomputing mapping raw times --> media frame indices, we
        # precompute mapping raw indices --> media frame indices. And when asked to
        # convert a raw time to a media frame index, we do
        # raw time --> raw index --> media frame index.

        logger.info("Building mapping from raw indices to media frame indices.")
        self._raw_idx_to_media_frame_idx: dict[int, MappingResult] = (
            self._build_mapping(
                source_timestamps_ms=self._raw_timestamps_ms,
                target_timestamps_ms=self._media_timestamps_ms,
            )
        )
        logger.info("Building mapping from media frame indices to raw times.")
        self._media_frame_idx_to_raw_time: dict[int, MappingResult] = (
            self._build_mapping(
                source_timestamps_ms=self._media_timestamps_ms,
                target_timestamps_ms=self._raw_timestamps_ms,
            )
        )
        self._log_mapping_results(
            mapping_results=self._raw_idx_to_media_frame_idx,
            header="Mapping results from raw indices to media frame indices:",
        )
        self._log_mapping_results(
            mapping_results=self._media_frame_idx_to_raw_time,
            header="Mapping results from media frame indices to raw times:",
        )

    def raw_time_to_media_sample_index(self, raw_idx: int) -> MappingResult:
        """Convert an index of raw data to media index."""
        # Find the raw index that corresponds to the given time point.
        # We cannot use the given time directly, as it may not match exactly with raw
        # times.
        return self._raw_idx_to_media_frame_idx[raw_idx]

    def media_sample_index_to_raw_time(self, media_frame_idx: int) -> MappingResult:
        """Convert a media sample index to a raw data time point (in seconds).

        Media sample index can be the index of a frame in a video or the index of an
        audio sample.
        """
        return self._media_frame_idx_to_raw_time[media_frame_idx]

    def _validate_input_times(self) -> None:
        if not np.all(np.diff(self._raw_timestamps_ms) >= 0):
            raise ValueError(
                "Raw timestamps are not strictly increasing. "
                "This is required for the mapping to work correctly."
            )
        if not np.all(np.diff(self._media_timestamps_ms) >= 0):
            raise ValueError(
                "Media timestamps are not strictly increasing. "
                "This is required for the mapping to work correctly."
            )

    def _diagnose_timestamps(self) -> None:
        """Log some statistics about the raw and media timestamps."""
        # Convert to seconds for easier readability
        raw_timestamps_seconds = self._raw_timestamps_ms / 1000.0
        media_timestamps_seconds = self._media_timestamps_ms / 1000.0
        logger.info(
            f"Raw timestamps: {raw_timestamps_seconds[0]:.1f} s to "
            f"{raw_timestamps_seconds[-1]:.1f} s, "
            f"total {len(raw_timestamps_seconds)} timestamps."
        )
        logger.info(
            f"Media timestamps: {media_timestamps_seconds[0]:.1f} s to "
            f"{media_timestamps_seconds[-1]:.1f} s, "
            f"total {len(media_timestamps_seconds)} timestamps."
        )

        # Check the interval between timesamps
        raw_intervals_ms = np.diff(self._raw_timestamps_ms)
        logger.info(
            f"Raw timestamps intervals: min={np.min(raw_intervals_ms):.3f} ms, "
            f"max={np.max(raw_intervals_ms):.3f} ms, "
            f"mean={np.mean(raw_intervals_ms):.3f} ms, "
            f"std={np.std(raw_intervals_ms):.3f} ms"
        )
        media_intervals_ms = np.diff(self._media_timestamps_ms)
        logger.info(
            f"Media timestamps intervals: min={np.min(media_intervals_ms):.3f} ms, "
            f"max={np.max(media_intervals_ms):.3f} ms, "
            f"mean={np.mean(media_intervals_ms):.3f} ms, "
            f"std={np.std(media_intervals_ms):.3f} ms"
        )
        media_too_small_count = np.sum(
            self._media_timestamps_ms < self._raw_timestamps_ms[0]
        )
        media_too_large_count = np.sum(
            self._media_timestamps_ms > self._raw_timestamps_ms[-1]
        )
        logger.info(
            "Media timestamps smaller/larger than first/last raw timestamp: "
            f"{media_too_small_count}/{media_too_large_count}"
        )
        raw_too_small_count = np.sum(
            self._raw_timestamps_ms < self._media_timestamps_ms[0]
        )
        raw_too_large_count = np.sum(
            self._raw_timestamps_ms > self._media_timestamps_ms[-1]
        )
        logger.info(
            "Raw timestamps smaller/larger than first/last media timestamp: "
            f"{raw_too_small_count}/{raw_too_large_count}"
        )
        first_timestamp_diff_ms = (
            self._raw_timestamps_ms[0] - self._media_timestamps_ms[0]
        )
        logger.info(
            "Difference between first raw and media timestamps: "
            f"{first_timestamp_diff_ms:.3f} ms"
        )
        if first_timestamp_diff_ms > 1000:
            logger.warning(
                "The raw data timestamps start over a second later than the media "
                "timestamps."
            )
        elif first_timestamp_diff_ms < -1000:
            logger.warning(
                "Media timestamps start over a second later than the raw data "
                "timestamps."
            )
        last_timestamp_diff_ms = (
            self._raw_timestamps_ms[-1] - self._media_timestamps_ms[-1]
        )
        logger.info(
            "Difference between last raw and media timestamps: "
            f"{last_timestamp_diff_ms:.3f} ms"
        )
        if last_timestamp_diff_ms > 1000:
            logger.warning(
                "The media timestamps end over a second earlier than the raw data "
                "timestamps."
            )
        elif last_timestamp_diff_ms < -1000:
            logger.warning(
                "Raw data timestamps end over a second earlier than the media "
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

        # Determine which target time is closer and in case of a tie,
        # select based on the _select_on_tie attribute.
        if self._select_on_tie == "left":
            comparison = left_distances <= right_distances
        elif self._select_on_tie == "right":
            comparison = left_distances < right_distances
        else:
            raise ValueError(
                f"Unknown select_on_tie value: {self._select_on_tie}. "
                "Expected 'left' or 'right'."
            )
        closest_indices = np.where(comparison, insert_indices - 1, insert_indices)

        return closest_indices

    def _log_mapping_errors(self, errors_ms: NDArray[np.floating]) -> None:
        """Log statistics about the distances between source and target timestamps."""
        logger.info(
            "    Statistics for mapping error (distances between source timestamps "
            "and their closest target timestamps):"
        )
        logger.info(
            f"    min={np.min(errors_ms):.3f} ms, max={np.max(errors_ms):.3f} ms, "
            f"mean={np.mean(errors_ms):.3f} ms, std={np.std(errors_ms):.3f} ms"
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
    ) -> dict[int, MappingResult]:
        """Build a mapping from raw indices to media frame indices or vice versa.

        Parameters
        ----------
        source_timestamps_ms : NDArray[np.floating]
            I-D sorted array of source timestamps in milliseconds for which to
            compute the mapping.
        target_timestamps_ms : NDArray[np.floating]
            I-D sorted array of target timestamps in milliseconds to which
            the source timestamps should be mapped.
        convert_raw_results_to_seconds : bool, optional
            If true, assume that the mapping is from media frame indices to raw indices,
            and convert the resulting raw indices to seconds.

        Returns
        -------
        dict[int, MappingResult]
            A dictionary mapping source indices to mapping results.
            If a source timestamp is out of bounds of the target timestamps,
            it will be marked as a failure with a reason for the failure
            (index too small or index too large).
        """
        # Initialize a list of mapping results with failures
        mapping: dict[int, MappingResult] = {}
        # Find indices of source timestamps that are out of bounds of the target
        # timestamps. Use half of the average interval between target timestamps
        # as a threshold to determine if a source timestamp is too small or too large.
        average_target_interval_ms = np.diff(target_timestamps_ms).mean()

        too_small_mask = source_timestamps_ms < (
            target_timestamps_ms[0] - average_target_interval_ms / 2
        )
        too_small_source_indices = too_small_mask.nonzero()[0]
        too_large_mask = source_timestamps_ms > (
            target_timestamps_ms[-1] + average_target_interval_ms / 2
        )
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

        for source_idx, result in zip(valid_source_indices, closest_target_indices):
            # Use .item() to convert numpy scalar to a native Python int or float.
            mapping[source_idx] = MappingSuccess(result=result.item())

        # Make sure that all the source indices were mapped.
        assert len(mapping) == len(source_timestamps_ms), (
            "Mapping should cover all the source indices."
        )

        return mapping

    def _log_mapping_results(
        self, mapping_results: dict[int, MappingResult], header: str
    ) -> None:
        """Log the number of each mapping result for debugging purposes."""
        result_counts = self._count_mapping_results(mapping_results)
        logger.debug(f"{header}")
        for result, count in result_counts.items():
            logger.debug(f"    {result}: {count}")

    def _count_mapping_results(
        self, mapping_results: dict[int, MappingResult]
    ) -> Counter[str]:
        """Count the number of each mapping results for debugging purposes."""
        counts = Counter()

        for mapping_result in mapping_results.values():
            match mapping_result:
                case MappingSuccess():
                    key = "MappingSuccess"
                case MappingFailure(failure_reason=reason):
                    key = f"MappingFailure({reason.name})"
                case _:
                    raise ValueError(f"Unexpected mapping result: {mapping_result}")
            counts[key] += 1

        return counts
