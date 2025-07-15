"""Contains the main class for synchronizing MNE raw data browser and video browser."""

import logging

from mne_qt_browser.figure import MNEQtBrowser
from qtpy import QtWidgets
from qtpy.QtCore import QElapsedTimer, QObject, Qt, QTimer, Signal, Slot  # type: ignore

from .raw_browser_manager import RawBrowserInterface, RawBrowserManager
from .raw_video_aligner import (
    MapFailureReason,
    MappingFailure,
    MappingResult,
    MappingSuccess,
    RawVideoAligner,
)
from .video import VideoFile
from .video_browser import SyncStatus, VideoBrowser

logger = logging.getLogger(__name__)


class SyncedRawVideoBrowser(QObject):
    """Instantiates MNE raw data browser and video browser, and synchronizes them.

    Parameters
    ----------
    raw_browser : mne_qt_browser.figure.MNEQtBrowser
        The MNE raw data browser object to be synchronized with the video browser.
        This can be created with 'plot' method of MNE raw data object when using qt
        backend.
    videos : list[VideoFile]
        The video file object(s) to be displayed in the video browser.
    aligners : list[RawVideoAligner]
        A list of `RawVideoAligner` instances, one for each video file.
        Each aligner provides the mapping between raw data time points and video frames
        for the corresponding video file. The order of the aligners must match the order
        of the video files in the `videos` parameter.
    max_sync_fps : int, optional
        The maximum frames per second for synchronizing the raw data browser and video
        browser. This determines how often the synchronization updates can happen and
        has an effect on the performance.
    show : bool, optional
        Whether to show the raw data browser immediately upon instantiation,
        by default True.
    parent : QObject, optional
        The parent QObject for this synchronized browser, by default None.
    """

    def __init__(
        self,
        raw_browser: MNEQtBrowser,
        videos: list[VideoFile],
        aligners: list[RawVideoAligner],
        show: bool = True,
        max_sync_fps: int = 10,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._video = videos
        self._aligners = aligners
        self._raw_update_max_fps = max_sync_fps

        # Wrap the raw browser to a class that exposes the necessary methods.
        raw_browser_interface = RawBrowserInterface(raw_browser, parent=self)
        # Pass interface for manager that contains actual logic for managing the browser
        # in sync with the video browser.
        self._raw_browser_manager = RawBrowserManager(
            raw_browser_interface, parent=self
        )
        # Make sure that raw browser visibility matches the `show` parameter.
        if show:
            self._raw_browser_manager.show_browser()
        else:
            self._raw_browser_manager.hide_browser()

        # Set up the video browser.
        self._video_browser = VideoBrowser(
            videos,
            show_sync_status=True,
            parent=None,
            # Save space in the UI excluding histogram with multiple videos.
            display_method="image_item" if len(videos) > 1 else "image_view",
        )

        # Dock the video browser to the raw data browser with Qt magic
        self._dock = QtWidgets.QDockWidget("Video Browser", raw_browser)
        self._dock.setWidget(self._video_browser)
        self._dock.setFloating(True)
        raw_browser.addDockWidget(Qt.RightDockWidgetArea, self._dock)
        self._dock.resize(1000, 800)  # Set initial size of the video browser
        if not show:
            self._dock.hide()

        # Set up synchronization

        self._min_sync_interval_ms = int(1000 / max_sync_fps)
        # Create a throttler that limits the updates of both raw and other videos
        # due to fast change of one video (playback).
        self._throttler = BufferedThrottler(self._min_sync_interval_ms, parent=self)
        self._video_browser.sigFrameChanged.connect(self._throttler.trigger)
        self._throttler.triggered.connect(self.sync_all_to_video)

        # When either raw time selector value or raw data browser's view changes,
        # update the video browser (no throttling needed here).
        self._raw_browser_manager.sigSelectedTimeChanged.connect(
            self.sync_videos_to_raw
        )

        # Consider raw data browser to be the main browser and start by
        # synchronizing the videos to the raw data browser's view
        initial_raw_time = self._raw_browser_manager.get_selected_time()
        self.sync_videos_to_raw(initial_raw_time)

    @Slot(float)
    def sync_videos_to_raw(self, raw_time_seconds: float) -> None:
        """Update the displayed video frame(s) when raw view changes."""
        logger.debug("")  # Clear debug log for clarity
        logger.debug(
            "Detected change in raw data browser's selected time, syncing video(s)."
        )
        for video_idx, aligner in enumerate(self._aligners):
            logger.debug(
                f"Syncing video {video_idx + 1}/{len(self._aligners)} to raw time: "
                f"{raw_time_seconds:.3f} seconds."
            )
            mapping_to_video = aligner.raw_time_to_video_frame_index(raw_time_seconds)
            self._update_video(video_idx, mapping_to_video)

    def _update_video(self, video_idx: int, mapping: MappingResult) -> None:
        """Update a video view based on mapping from a raw time to video frame index.

        Either shows the video frame that corresponds to the raw time point,
        or shows the first or last frame of the video if the raw time point
        is out of bounds of the video data.

        Parameters
        ----------
        video_idx : int
            The index of the video to update.
        mapping : MappingResult
            The result of mapping the raw time point to a video frame index.
        """
        # NOTE: The signal=False is used to prevent the video browser from
        # emitting the frame changed signal, which would trigger update of the
        # raw browser and cause an infinite loop of updates.
        match mapping:
            case MappingSuccess(result=frame_idx):
                # Raw time point has a corresponding video frame index
                logger.debug(
                    f"Setting video on index {video_idx} to show frame: {frame_idx}"
                )
                self._video_browser.display_frame_for_video_with_idx(
                    frame_idx, video_idx, signal=False
                )
                # self._video_browser.set_sync_status(SyncStatus.SYNCHRONIZED)

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_SMALL):
                # Raw time stamp is smaller than the first video frame timestamp
                logger.debug(
                    f"Video on index {video_idx} has no data for this small raw time "
                    "point, showing first frame."
                )
                # self._video_browser.set_sync_status(SyncStatus.NO_VIDEO_DATA)
                self._video_browser.display_frame_for_video_with_idx(
                    0, video_idx, signal=False
                )

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_LARGE):
                # Raw time stamp is larger than the last video frame timestamp
                logger.debug(
                    f"Video on index {video_idx} has no data for this large raw time "
                    "point, showing last frame."
                )
                # self._video_browser.set_sync_status(SyncStatus.NO_VIDEO_DATA)
                self._video_browser.display_frame_for_video_with_idx(
                    self._video[video_idx].frame_count - 1, video_idx, signal=False
                )

            case _:
                raise ValueError(f"Unexpected mapping result: {mapping}. ")

    @Slot(int, int)
    def sync_all_to_video(self, video_idx: int, frame_idx: int) -> None:
        """Update raw data browser's view and other videos when video frame changes."""
        logger.debug("")  # Clear debug log for clarity
        logger.debug(
            f"Detected change in video {video_idx + 1} to frame index: {frame_idx}. "
            "Syncing raw data browser."
        )
        # Update the raw browser view based on the selected video frame index.
        mapping_to_raw = self._aligners[video_idx].video_frame_index_to_raw_time(
            frame_idx
        )
        self._update_raw(mapping_to_raw)
        # Get the resulting raw time by asking it from the browser and use
        # it to update other videos (if any).
        raw_time_seconds = self._raw_browser_manager.get_selected_time()
        for idx, aligner in enumerate(self._aligners):
            if idx == video_idx:
                # Skip the video that triggered the change
                continue
            logger.debug(
                f"Syncing video {idx + 1}/{len(self._aligners)} to raw time: "
                f"{raw_time_seconds:.3f} seconds."
            )
            mapping_to_video = aligner.raw_time_to_video_frame_index(raw_time_seconds)
            self._update_video(idx, mapping_to_video)

    def _update_raw(self, mapping: MappingResult) -> None:
        """Update raw browser view based on mapping from video frame index to raw time.

        If the video frame index is out of bounds of the raw data, moves the raw view
        to the start or end of the raw data.

        Parameters
        ----------
        mapping : MappingResult
            The result of mapping the video frame index to a raw time point.
        """
        match mapping:
            case MappingSuccess(result=raw_time):
                # Video frame index has a corresponding raw time point
                logger.debug(f"Setting raw browser to time: {raw_time:.3f} seconds.")
                self._raw_browser_manager.set_selected_time_no_signal(raw_time)
                self._video_browser.set_sync_status(SyncStatus.SYNCHRONIZED)

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_SMALL):
                # Video frame index is smaller than the first raw time point
                logger.debug(
                    "No raw data for this small video frame, moving raw view to start."
                )
                self._video_browser.set_sync_status(SyncStatus.NO_RAW_DATA)
                self._raw_browser_manager.jump_to_start()

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_LARGE):
                logger.debug(
                    "No raw data for this large video frame, moving raw view to end."
                )
                self._video_browser.set_sync_status(SyncStatus.NO_RAW_DATA)
                self._raw_browser_manager.jump_to_end()

            case _:
                raise ValueError(f"Unexpected mapping result: {mapping}. ")

    def show(self) -> None:
        """Show the synchronized raw and video browsers."""
        self._raw_browser_manager.show_browser()
        self._dock.show()


class BufferedThrottler(QObject):
    """Emits the most recent input payload no more than once every `interval_ms`.

    If enough time has passed since last emit, emits the received payload immediately.
    Otherwise schedules the received payload to be emitted after the required time has
    passed.

    Parameters
    ----------
    interval_ms : int
        The minimum interval in milliseconds between emits.
    parent : QObject, optional
        The parent QObject for this throttler, by default None.
    """

    triggered = Signal(int, int)  # hard coded for signal emitted by video browser

    def __init__(self, interval_ms: int, parent: QObject | None = None) -> None:
        super().__init__(parent=parent)

        self._emit_interval_ms = interval_ms
        self._latest_payload = None  # holds the next value to emit

        # Start a timer to count milliseconds since last emit.
        self._elapsed_timer = QElapsedTimer()
        self._elapsed_timer.start()

        # Initialize another timer to schedule emits to happen later.
        self._delayed_emit_timer = QTimer(parent=self)
        self._delayed_emit_timer.setSingleShot(True)
        self._delayed_emit_timer.timeout.connect(self._emit_now)

    @Slot(int, int)
    def trigger(self, payload1: int, payload2: int) -> None:
        """Trigger the throttler with a new payload."""
        self._latest_payload = (payload1, payload2)

        elapsed_time_ms = self._elapsed_timer.elapsed()
        remaining_time_ms = self._emit_interval_ms - elapsed_time_ms

        if remaining_time_ms <= 0:
            # Enough time has passed since last emit.
            self._emit_now()
        else:
            # Triggered too soon. Start delayed emit timer if its not already running.
            if not self._delayed_emit_timer.isActive():
                self._delayed_emit_timer.start(remaining_time_ms)

    @Slot()
    def _emit_now(self) -> None:
        # Start counting time since last emit again from zero.
        self._elapsed_timer.restart()
        # Make sure that no delayed emits will happen before new trigger.
        self._delayed_emit_timer.stop()
        # Fire!
        assert self._latest_payload is not None, "No payload to emit."
        logger.debug(f"Emitting latest payload: {self._latest_payload}")
        self.triggered.emit(self._latest_payload[0], self._latest_payload[1])
