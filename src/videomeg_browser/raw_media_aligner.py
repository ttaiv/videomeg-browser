"""Bidirectional timestamp alignment for synchronizing different data streams.

This module provides the TimestampAligner class for mapping indices between two
arrays of timestamps. It's designed to handle temporal alignment between any
types of time-series data (e.g., raw MEG data, video frames, audio samples).

The aligner uses closest-time matching with configurable tie-breaking.

Classes
-------
TimestampAligner
    Maps indices between two timestamp arrays bidirectionally.
MappingResult
    Abstract base class for alignment results.
MappingSuccess
    Represents successful alignment with a target index.
MappingFailure
    Represents failed alignment with a failure reason.
MapFailureReason
    Enumeration of possible alignment failure reasons.
"""

import logging
from abc import ABC
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MapFailureReason(Enum):
    """Enum telling why mapping failed."""

    # Index to map is too small
    INDEX_TOO_SMALL = "index_too_small"
    # Index to map is too large
    INDEX_TOO_LARGE = "index_too_large"


class MappingResult(ABC):
    """Represents the result of mapping one timestamp to another."""

    pass


@dataclass(frozen=True)
class MappingSuccess(MappingResult):
    """Represents a successful mapping that yielded an index."""

    result: int


@dataclass(frozen=True)
class MappingFailure(MappingResult):
    """Represents a failed mapping with a reason for the failure."""

    failure_reason: MapFailureReason


class TimestampAligner:
    """Maps indices between two arrays of timestamps bidirectionally.

    Parameters
    ----------
    timestamps_a : NDArray[np.floating]
        1-D sorted array of timestamps.
    timestamps_b : NDArray[np.floating]
        1-D sorted array of timestamps.
    timestamp_unit : Literal["milliseconds", "seconds"], optional
        The unit of the timestamps in `timestamps_a` and `timestamps_b`.
        By default "milliseconds".
    select_on_tie : Literal["left", "right"], optional
        How to select the result when a source timestamp is exactly between two target
        timestamps. If "left", the index corresponding to the left target timestamp
        is selected. If "right", the index corresponding to the right target timestamp
        is selected. By default "left".
    timestamps_a_name : str, optional
        Name for the timestamps in `timestamps_a`, used in logging messages.
    timestamps_b_name : str, optional
        Name for the timestamps in `timestamps_b`, used in logging messages.
    """

    # Threshold for warning about difference between start/end times of the timestamps
    _WARNING_THRESHOLD_MS = 1000

    def __init__(
        self,
        timestamps_a: NDArray[np.floating],
        timestamps_b: NDArray[np.floating],
        timestamp_unit: Literal["milliseconds", "seconds"] = "milliseconds",
        select_on_tie: Literal["left", "right"] = "left",
        name_a: str = "a",
        name_b: str = "b",
    ) -> None:
        self._timestamp_unit = timestamp_unit
        self._select_on_tie = select_on_tie
        # Internally store timestamps in milliseconds.
        self._timestamps_a_ms = self._get_timestamps_in_milliseconds(timestamps_a)
        self._timestamps_b_ms = self._get_timestamps_in_milliseconds(timestamps_b)

        self._name_a = name_a
        self._name_b = name_b

        self._validate_input_times()
        self._diagnose_timestamps()

        # Precompute mapping from timestamps a to b and vice versa.

        logger.info(f"Building mapping from {name_a} to {name_b}.")
        self._mapping_ab: dict[int, MappingResult] = self._build_mapping(
            source_timestamps_ms=self._timestamps_a_ms,
            target_timestamps_ms=self._timestamps_b_ms,
        )
        logger.info(f"Building mapping from {name_b} to {name_a}.")
        self._mapping_ba: dict[int, MappingResult] = self._build_mapping(
            source_timestamps_ms=self._timestamps_b_ms,
            target_timestamps_ms=self._timestamps_a_ms,
        )
        self._log_mapping_results(
            mapping_results=self._mapping_ab,
            header=f"Mapping results from {name_a} to {name_b}:",
        )
        self._log_mapping_results(
            mapping_results=self._mapping_ba,
            header=f"Mapping results from {name_b} to {name_a}:",
        )

    def a_index_to_b_index(self, a_idx: int) -> MappingResult:
        """Map an index in `timestamps_a` to the closest index in `timestamps_b`."""
        return self._mapping_ab[a_idx]

    def b_index_to_a_index(self, b_idx: int) -> MappingResult:
        """Map an index in `timestamps_b` to the closest index in `timestamps_a`."""
        return self._mapping_ba[b_idx]

    def _validate_input_times(self) -> None:
        if not np.all(np.diff(self._timestamps_a_ms) >= 0):
            raise ValueError(
                f"{self._name_a} timestamps are not strictly increasing. "
                "This is required for the mapping to work correctly."
            )
        if not np.all(np.diff(self._timestamps_b_ms) >= 0):
            raise ValueError(
                f"{self._name_b} timestamps are not strictly increasing. "
                "This is required for the mapping to work correctly."
            )

    def _diagnose_timestamps(self) -> None:
        """Log some statistics about the timestamps."""
        # Convert to seconds for easier readability
        timestamps_a_seconds = self._timestamps_a_ms / 1000.0
        timestamps_b_seconds = self._timestamps_b_ms / 1000.0
        logger.info(
            f"{self._name_a} timestamps: {timestamps_a_seconds[0]:.1f} s to "
            f"{timestamps_a_seconds[-1]:.1f} s, "
            f"total {len(timestamps_a_seconds)} timestamps."
        )
        logger.info(
            f"{self._name_b} timestamps: {timestamps_b_seconds[0]:.1f} s to "
            f"{timestamps_b_seconds[-1]:.1f} s, "
            f"total {len(timestamps_b_seconds)} timestamps."
        )

        # Check the interval between timesamps
        intervals_a_ms = np.diff(self._timestamps_a_ms)
        logger.info(
            f"{self._name_a} timestamps intervals: "
            f"min={np.min(intervals_a_ms):.3f} ms, "
            f"max={np.max(intervals_a_ms):.3f} ms, "
            f"mean={np.mean(intervals_a_ms):.3f} ms, "
            f"std={np.std(intervals_a_ms):.3f} ms"
        )
        intervals_b_ms = np.diff(self._timestamps_b_ms)
        logger.info(
            f"{self._name_b} timestamps intervals: "
            f"min={np.min(intervals_b_ms):.3f} ms, "
            f"max={np.max(intervals_b_ms):.3f} ms, "
            f"mean={np.mean(intervals_b_ms):.3f} ms, "
            f"std={np.std(intervals_b_ms):.3f} ms"
        )
        too_small_count_b = np.sum(self._timestamps_b_ms < self._timestamps_a_ms[0])
        too_large_count_b = np.sum(self._timestamps_b_ms > self._timestamps_a_ms[-1])
        logger.info(
            f"{self._name_b} timestamps smaller/larger than "
            f"first/last {self._name_a} timestamp: "
            f"{too_small_count_b}/{too_large_count_b}"
        )
        too_small_count_a = np.sum(self._timestamps_a_ms < self._timestamps_b_ms[0])
        too_large_count_a = np.sum(self._timestamps_a_ms > self._timestamps_b_ms[-1])
        logger.info(
            f"{self._name_a} timestamps smaller/larger than "
            f"first/last {self._name_b} timestamp: "
            f"{too_small_count_a}/{too_large_count_a}"
        )
        first_timestamp_diff_ms = self._timestamps_a_ms[0] - self._timestamps_b_ms[0]
        logger.info(
            f"Difference between first {self._name_a} and "
            f"{self._name_b} timestamps: "
            f"{first_timestamp_diff_ms:.3f} ms"
        )
        if first_timestamp_diff_ms > self._WARNING_THRESHOLD_MS:
            logger.warning(
                f"The {self._name_a} timestamps start over a second "
                f"later than the {self._name_b} timestamps."
            )
        elif first_timestamp_diff_ms < -self._WARNING_THRESHOLD_MS:
            logger.warning(
                f"{self._name_b} timestamps start over a second "
                f"later than the {self._name_a} timestamps."
            )
        last_timestamp_diff_ms = self._timestamps_a_ms[-1] - self._timestamps_b_ms[-1]
        logger.info(
            f"Difference between last {self._name_a} and "
            f"{self._name_b} timestamps: "
            f"{last_timestamp_diff_ms:.3f} ms"
        )
        if last_timestamp_diff_ms > self._WARNING_THRESHOLD_MS:
            logger.warning(
                f"The {self._name_b} timestamps end over a second "
                f"earlier than the {self._name_a} timestamps."
            )
        elif last_timestamp_diff_ms < -self._WARNING_THRESHOLD_MS:
            logger.warning(
                f"{self._name_a} timestamps end over a second "
                f"earlier than the {self._name_b} timestamps."
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
            1-D sorted array of source times
        target_times : NDArray[np.floating]
            1-D sorted array of target times

        Returns
        -------
        NDArray[np.intp]
            1-D array consisting of the indices of the closest target times
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
    ) -> dict[int, MappingResult]:
        """Build a mapping from source indices to target indices.

        Parameters
        ----------
        source_timestamps_ms : NDArray[np.floating]
            1-D sorted array of source timestamps in milliseconds for which to
            compute the mapping.
        target_timestamps_ms : NDArray[np.floating]
            1-D sorted array of target timestamps in milliseconds to which
            the source timestamps should be mapped.

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
