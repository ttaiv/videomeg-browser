import logging
from abc import ABC
from dataclasses import dataclass
from enum import Enum

import mne
import numpy as np
from numpy.typing import NDArray
from qtpy import QtWidgets
from qtpy.QtCore import Qt, Slot

from .comp_tstamps import comp_tstamps
from .raw_browser_manager import RawBrowserInterface, RawBrowserManager
from .video import VideoFileHelsinkiVideoMEG
from .video_browser import SyncStatus, VideoBrowser

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

    Currently, this is tailored for the Helsinki Video MEG data format,
    but this could be extended to other formats as well.
    """

    def __init__(
        self, raw: mne.io.Raw, raw_timing_ch: str, video: VideoFileHelsinkiVideoMEG
    ) -> None:
        self.raw = raw
        self.video_timestamps_ms = video.ts

        logger.info("Initializing mapping from raw data time points to video frames.")
        logger.info(
            f"Using timing channel '{raw_timing_ch}' for timestamp computation."
        )

        timing_data = raw.get_data(picks=raw_timing_ch, return_times=False)
        # Remove the channel dimension
        # Ignoring warning about timing_data possibly being tuple,
        # as we do not ask times from raw.get_data
        timing_data = timing_data.squeeze()  # type: ignore
        logger.debug(f"Timing channel data shape: {timing_data.shape}, ")

        self.raw_timestamps_ms = comp_tstamps(timing_data, raw.info["sfreq"])

        if len(self.raw_timestamps_ms) != len(raw.times):
            raise ValueError(
                "The number of timestamps in the raw data does not match "
                "the number of time points."
            )

        self._validate_timestamps()
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
        self.raw_idx_to_video_frame_idx: list[MappingResult] = self._build_mapping(
            source_timestamps=self.raw_timestamps_ms,
            target_timestamps=self.video_timestamps_ms,
        )

        logger.info("Building mapping from video frame indices to raw times.")
        self.video_frame_idx_to_raw_time: list[MappingResult] = self._build_mapping(
            source_timestamps=self.video_timestamps_ms,
            target_timestamps=self.raw_timestamps_ms,
            convert_raw_results_to_seconds=True,
        )

        self._log_mapping_results(
            mapping_results=self.raw_idx_to_video_frame_idx,
            header="Mapping results from raw indices to video frame indices:",
        )
        self._log_mapping_results(
            mapping_results=self.video_frame_idx_to_raw_time,
            header="Mapping results from video frame indices to raw times:",
        )

    def _validate_timestamps(self) -> None:
        """Validate that raw and video timestamps are strictly increasing."""
        if not np.all(np.diff(self.raw_timestamps_ms) >= 0):
            raise ValueError(
                "Raw timestamps are not strictly increasing. "
                "This is required for the mapping to work correctly."
            )
        if not np.all(np.diff(self.video_timestamps_ms) >= 0):
            raise ValueError(
                "Video timestamps are not strictly increasing. "
                "This is required for the mapping to work correctly."
            )

    def _diagnose_timestamps(self) -> None:
        """Log some statistics about the raw and video timestamps."""
        logger.info(
            f"Raw timestamps: {self.raw_timestamps_ms[0]} ms to "
            f"{self.raw_timestamps_ms[-1]} ms, "
            f"total {len(self.raw_timestamps_ms)} timestamps."
        )
        logger.info(
            f"Video timestamps: {self.video_timestamps_ms[0]} ms to "
            f"{self.video_timestamps_ms[-1]} ms, "
            f"total {len(self.video_timestamps_ms)} timestamps."
        )
        # Count timestamps that are out of bounds
        video_too_small_count = np.sum(
            self.video_timestamps_ms < self.raw_timestamps_ms[0]
        )
        video_too_large_count = np.sum(
            self.video_timestamps_ms > self.raw_timestamps_ms[-1]
        )
        logger.info(
            "Video timestamps smaller than first raw timestamp: "
            f"{video_too_small_count}"
        )
        logger.info(
            f"Video timestamps larger than last raw timestamp: {video_too_large_count}"
        )
        raw_too_small_count = np.sum(
            self.raw_timestamps_ms < self.video_timestamps_ms[0]
        )
        raw_too_large_count = np.sum(
            self.raw_timestamps_ms > self.video_timestamps_ms[-1]
        )
        logger.info(
            f"Raw timestamps smaller than first video timestamp: {raw_too_small_count}"
        )
        logger.info(
            f"Raw timestamps larger than last video timestamp: {raw_too_large_count}"
        )

    def _find_indices_with_closest_values(
        self, source_times: NDArray[np.floating], target_times: NDArray[np.floating]
    ) -> NDArray[np.intp]:
        """Find indices of the closest target times for each source time.

        Raises ValueError if any source time is out of bounds of the target times.

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

        if any(insert_indices == 0):
            raise ValueError(
                "Some source times are smaller than the first target time. "
                "This is not allowed."
            )
        if any(insert_indices >= len(target_times)):
            raise ValueError(
                "Some source times are larger than the last target time. "
                "This is not allowed."
            )

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

    def _build_mapping(
        self,
        source_timestamps: NDArray[np.floating],
        target_timestamps: NDArray[np.floating],
        convert_raw_results_to_seconds: bool = False,
    ) -> list[MappingResult]:
        """Build a mapping from raw indices to video frame indices or vice versa.

        Parameters
        ----------
        source_timestamps : NDArray[np.floating]
            I-D sorted array of source timestamps for which to compute the mapping
        target_timestamps : NDArray[np.floating]
            I-D sorted array of target timestamps to which the source timestamps
            should be mapped
        convert_raw_results_to_seconds : bool, optional
            If true, assume that the mapping is from video frame indices to raw indices,
            and convert the resulting raw indices to seconds.

        Returns
        -------
        list[MappingResult]
            List of mapping results, where each result corresponds to a source
            timestamp. If a source timestamp is out of bounds of the target timestamps,
            it will be marked as a failure with a reason for the failure.
        """
        # Initialize a list of mapping results with failures
        mapping: list[MappingResult] = [
            MappingFailure(MapFailureReason.NOT_MAPPED)
            for _ in range(len(source_timestamps))
        ]

        # Find indices of source timestamps that are out of bounds
        # of the target timestamps.
        too_small_mask = source_timestamps < target_timestamps[0]
        too_small_source_indices = too_small_mask.nonzero()[0]
        too_large_mask = source_timestamps > target_timestamps[-1]
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
        valid_source_timestamps_ms = source_timestamps[valid_mask]
        valid_source_indices = valid_mask.nonzero()[0]

        closest_target_indices = self._find_indices_with_closest_values(
            source_times=valid_source_timestamps_ms, target_times=target_timestamps
        )
        if convert_raw_results_to_seconds:
            # Convert the raw indices to actual time points in seconds
            closest_raw_times = self.raw.times[closest_target_indices]
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
                    raise ValueError(
                        "Not all source indices were mapped to target indices. "
                        "This should not happen."
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
            logger.debug(f"{result}: {count}")

    def _count_mapping_results(
        self, mapping_results: list[MappingResult]
    ) -> dict[str, int]:
        """Count the number of each mapping results for debugging purposes."""
        result_counts = {
            "MappingSuccess": 0,
            "MappingFailure(INDEX_TOO_SMALL)": 0,
            "MappingFailure(INDEX_TOO_LARGE)": 0,
            "MappingFailure(NOT_MAPPED)": 0,
        }
        for mapping_result in mapping_results:
            match mapping_result:
                case MappingSuccess():
                    result_counts["MappingSuccess"] += 1
                case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_SMALL):
                    result_counts["MappingFailure(INDEX_TOO_SMALL)"] += 1
                case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_LARGE):
                    result_counts["MappingFailure(INDEX_TOO_LARGE)"] += 1
                case MappingFailure(failure_reason=MapFailureReason.NOT_MAPPED):
                    result_counts["MappingFailure(NOT_MAPPED)"] += 1
                case _:
                    raise ValueError(f"Unexpected mapping result: {mapping_result}.")

        return result_counts

    def raw_time_to_video_frame_index(self, raw_time_seconds: float) -> MappingResult:
        """Convert a time point from raw data (in seconds) to video frame index."""
        # Find the raw index that corresponds to the given time point.
        # We cannot use the given time directly, as it may not match exactly with raw
        # times.
        raw_idx = self.raw.time_as_index(raw_time_seconds, use_rounding=True)[0]
        return self.raw_idx_to_video_frame_idx[raw_idx]

    def video_frame_index_to_raw_time(self, video_frame_idx: int) -> MappingResult:
        """Convert a video frame index to a raw data time point (in seconds)."""
        return self.video_frame_idx_to_raw_time[video_frame_idx]


class SyncedRawVideoBrowser:
    """Instantiates MNE raw data browser and video browser, and synchronizes them."""

    def __init__(
        self,
        raw: mne.io.Raw,
        video_file: VideoFileHelsinkiVideoMEG,
        time_mapper: TimeIndexMapper,
    ) -> None:
        self.raw = raw
        self.video_file = video_file
        self.time_mapper = time_mapper
        # Flag to prevent infinite recursion during synchronization
        self._syncing = False

        # Set up Qt application
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

        # Instantiate the MNE Qt Browser
        self.raw_browser = raw.plot(block=False)
        # Wrap it in a interface class that exposes the necessary methods
        self.raw_browser_interface = RawBrowserInterface(self.raw_browser)
        # Pass interface for manager that contains actual logic for managing the browser
        # in sync with the video browser
        self.raw_browser_manager = RawBrowserManager(self.raw_browser_interface)

        # Set up the video browser
        self.video_browser = VideoBrowser(video_file, show_sync_status=True)

        # Dock the video browser to the raw data browser with Qt magic
        self.dock = QtWidgets.QDockWidget("Video Browser", self.raw_browser)
        self.dock.setWidget(self.video_browser)
        self.dock.setFloating(True)
        self.raw_browser.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.resize(1000, 800)  # Set initial size of the video browser

        # Set up synchronization

        # When video browser frame changes, update the raw data browser's view
        self.video_browser.sigFrameChanged.connect(self.sync_raw_to_video)
        # When either raw time selector value or raw data browser's view changes,
        # update the video browser
        self.raw_browser_manager.sigSelectedTimeChanged.connect(self.sync_video_to_raw)

        # Consider raw data browser to be the main browser and start by
        # synchronizing the video browser to the raw data browser's view
        initial_raw_time = self.raw_browser_manager.get_selected_time()
        # Also updates the raw time selector
        self.sync_video_to_raw(initial_raw_time)

    @Slot(tuple)
    def sync_video_to_raw(self, raw_time_seconds: float) -> None:
        """Update the displayed video frame when raw view changes."""
        if self._syncing:
            # Prevent infinite recursion
            logger.debug("Already syncing, skip updating video view.")
            return
        self._syncing = True
        logger.debug("")  # Clear debug log for clarity
        logger.debug(
            "Detected change in raw data browser's selected time, syncing video."
        )

        self._update_video(raw_time_seconds)
        self._syncing = False

    def _update_video(self, raw_time_seconds: float) -> None:
        """Update video browser view based on selected raw time point.

        Either shows the video frame that corresponds to the raw time point,
        or shows the first or last frame of the video if the raw time point
        is out of bounds of the video data.

        Parameters
        ----------
        raw_time_seconds : float
            The raw time point in seconds to which the video browser should be synced.
        """
        mapping = self.time_mapper.raw_time_to_video_frame_index(raw_time_seconds)

        match mapping:
            case MappingSuccess(result=video_idx):
                # Raw time point has a corresponding video frame index
                logger.debug(
                    f"Setting video browser to show frame with index: {video_idx}"
                )
                self.video_browser.display_frame_at(video_idx)
                self.video_browser.set_sync_status(SyncStatus.SYNCHRONIZED)

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_SMALL):
                # Raw time stamp is smaller than the first video frame timestamp
                logger.debug(
                    "No video data for this small raw time point, showing first frame."
                )
                self.video_browser.set_sync_status(SyncStatus.NO_VIDEO_DATA)
                self.video_browser.display_frame_at(0)

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_LARGE):
                # Raw time stamp is larger than the last video frame timestamp
                logger.debug(
                    "No video data for this large raw time point, showing last frame."
                )
                self.video_browser.set_sync_status(SyncStatus.NO_VIDEO_DATA)
                self.video_browser.display_frame_at(self.video_file.frame_count - 1)
            case _:
                raise ValueError(f"Unexpected mapping result: {mapping}. ")

    @Slot(int)
    def sync_raw_to_video(self, video_frame_idx: int) -> None:
        """Update raw data browser's view and time selector when video frame changes."""
        if self._syncing:
            # Prevent infinite recursion
            logger.debug("Already syncing, skip updating raw view.")
            return
        self._syncing = True

        logger.debug("")  # Clear debug log for clarity
        logger.debug(f"Syncing raw browser to video frame index: {video_frame_idx}")
        mapping = self.time_mapper.video_frame_index_to_raw_time(video_frame_idx)

        match mapping:
            case MappingSuccess(result=raw_time):
                # Video frame index has a corresponding raw time point
                logger.debug(f"Corresponding raw time in seconds: {raw_time:.3f}")
                self.raw_browser_manager.set_selected_time(raw_time)
                self.video_browser.set_sync_status(SyncStatus.SYNCHRONIZED)

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_SMALL):
                # Video frame index is smaller than the first raw time point
                logger.debug(
                    "No raw data for this small video frame, moving raw view to start."
                )
                self.video_browser.set_sync_status(SyncStatus.NO_RAW_DATA)
                self.raw_browser_manager.jump_to_start()

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_LARGE):
                logger.debug(
                    "No raw data for this large video frame, moving raw view to end."
                )
                self.video_browser.set_sync_status(SyncStatus.NO_RAW_DATA)
                self.raw_browser_manager.jump_to_end()
            case _:
                raise ValueError(f"Unexpected mapping result: {mapping}. ")

        self._syncing = False

    def show(self) -> None:
        """Show the synchronized raw and video browsers."""
        self.raw_browser_manager.show_browser()
        self.app.exec_()
