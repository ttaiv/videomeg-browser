"""Code for syncing MNE raw data browser with video or audio browser."""

import logging
from typing import Literal

from mne_qt_browser.figure import MNEQtBrowser
from qtpy.QtCore import QElapsedTimer, QObject, Qt, QTimer, Signal, Slot  # type: ignore
from qtpy.QtWidgets import QDockWidget

from .audio import AudioFile
from .audio_browser import AudioBrowser
from .raw_browser_manager import RawBrowserInterface, RawBrowserManager
from .raw_media_aligner import (
    MapFailureReason,
    MappingFailure,
    MappingResult,
    MappingSuccess,
    RawMediaAligner,
)
from .syncable_media_browser import SyncableMediaBrowser, SyncStatus
from .video import VideoFile
from .video_browser import VideoBrowser

logger = logging.getLogger(__name__)


class SyncedRawMediaBrowser(QObject):
    """Synchronize MNE raw data browser with a media browser.

    Parameters
    ----------
    raw_browser : mne_qt_browser.figure.MNEQtBrowser
        The MNE raw data browser object to be synchronized with the media browser.
        This can be created with 'plot' method of MNE raw data object when using qt
        backend.
    media_browser : SyncableMediaBrowser
        The media browser object to be synchronized with the raw data browser.
    aligners : list[RawMediaAligner]
        A list of `RawMediaAligner` instances, one for each media file.
        Each aligner provides the mapping between raw data time points and media samples
        for the corresponding media file.
    media_browser_title : str
        The title of the media browser dock widget.
    max_sync_fps : int, optional
        The maximum frames per second for synchronizing the raw data browser and media
        browser. This determines how often the synchronization updates can happen and
        has an effect on the performance.
    show : bool, optional
        Whether to show the browsers immediately, by default True.
    parent : QObject, optional
        The parent QObject for this synchronized browser, by default None.
    """

    def __init__(
        self,
        raw_browser: MNEQtBrowser,
        media_browser: SyncableMediaBrowser,
        aligners: list[RawMediaAligner],
        media_browser_title: str,
        show: bool = True,
        max_sync_fps: int = 10,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._media_browser = media_browser
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

        # Dock the media browser to the raw data browser with Qt magic.
        self._dock = QDockWidget(media_browser_title, raw_browser)
        self._dock.setWidget(self._media_browser)
        self._dock.setFloating(True)
        raw_browser.addDockWidget(Qt.RightDockWidgetArea, self._dock)
        self._dock.resize(1000, 800)  # Set initial size of the video browser
        if not show:
            self._dock.hide()

        # Set up synchronization

        # Create a throttler that limits the updates of both raw and other media
        # due to fast change of one media.
        self._min_sync_interval_ms = int(1000 / max_sync_fps)
        self._throttler = BufferedThrottler(self._min_sync_interval_ms, parent=self)

        # When the position in a media changes
        # update the raw browser and other media through throttler.
        self._media_browser.sigPositionChanged.connect(self._throttler.trigger)
        self._throttler.triggered.connect(self._sync_all_to_media)

        # When selected time of raw browser changes,
        # update the media (no throttling needed here).
        self._raw_browser_manager.sigSelectedTimeChanged.connect(
            self._sync_media_to_raw
        )

        # Consider raw data browser to be the main browser and start by
        # synchronizing the media browser to the initial raw time.
        initial_raw_time = self._raw_browser_manager.get_selected_time()
        self._sync_media_to_raw(initial_raw_time)

    def show(self) -> None:
        """Show the synchronized raw and video browsers."""
        self._raw_browser_manager.show_browser()
        self._dock.show()

    @Slot(float)
    def _sync_media_to_raw(self, raw_time_seconds: float) -> None:
        """Update the media browser's view based on the selected raw time."""
        logger.debug(
            "Detected change in raw data browser's selected time, syncing media."
        )
        for media_idx, aligner in enumerate(self._aligners):
            logger.debug(
                f"Syncing media {media_idx + 1}/{len(self._aligners)} to raw time: "
                f"{raw_time_seconds:.3f} seconds."
            )
            mapping_to_media = aligner.raw_time_to_media_sample_index(raw_time_seconds)
            self._update_media(media_idx, mapping_to_media)

    def _update_media(self, media_idx: int, mapping: MappingResult) -> None:
        """Update media browser view based on mapping from raw time to frame/sample."""
        # NOTE: The signal=False is used to prevent the media browser from
        # emitting the sigPositionChanged signal, which would trigger update of the
        # raw browser and cause an infinite loop of updates.
        match mapping:
            case MappingSuccess(result=position_idx):
                # Raw time point has a corresponding media frame/sample.
                self._media_browser.set_position(position_idx, media_idx, signal=False)
                self._media_browser.set_sync_status(SyncStatus.SYNCHRONIZED, media_idx)
            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_SMALL):
                # Raw time stamp is smaller than the first media timestamp.
                logger.debug(
                    f"Media on index {media_idx} has no data for this small raw time "
                    "point, moving media to start."
                )
                self._media_browser.jump_to_start(media_idx, signal=False)
                self._media_browser.set_sync_status(SyncStatus.NO_MEDIA_DATA, media_idx)

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_LARGE):
                # Raw time stamp is larger than the last media frame timestamp
                logger.debug(
                    f"Media on index {media_idx} has no data for this large raw time "
                    "point, showing last frame."
                )
                self._media_browser.jump_to_end(media_idx, signal=False)
                self._media_browser.set_sync_status(SyncStatus.NO_MEDIA_DATA, media_idx)

            case _:
                raise ValueError(f"Unexpected mapping result: {mapping}. ")

    @Slot(int, int)
    def _sync_all_to_media(self, media_idx: int, position_idx: int) -> None:
        """Update raw data browser's view and other media when media changes."""
        logger.debug(
            f"Detected change in media {media_idx + 1} to position "
            f"index: {position_idx}. Syncing raw data browser and other media."
        )
        # Update the raw browser view based on the media.
        mapping_to_raw = self._aligners[media_idx].media_sample_index_to_raw_time(
            position_idx
        )
        mapping_success = self._update_raw(mapping_to_raw)
        if mapping_success:
            self._media_browser.set_sync_status(SyncStatus.SYNCHRONIZED, media_idx)
        else:
            # Signal that there is no raw data for this video frame index.
            self._media_browser.set_sync_status(SyncStatus.NO_RAW_DATA, media_idx)
        # Get the resulting raw time by asking it from the browser and use
        # it to update other media (if any).
        raw_time_seconds = self._raw_browser_manager.get_selected_time()
        for idx, aligner in enumerate(self._aligners):
            if idx == media_idx:
                # Skip the media that triggered the change.
                continue
            logger.debug(
                f"Syncing media {idx + 1}/{len(self._aligners)} to raw time: "
                f"{raw_time_seconds:.3f} seconds."
            )
            mapping_to_media = aligner.raw_time_to_media_sample_index(raw_time_seconds)
            self._update_media(idx, mapping_to_media)

    def _update_raw(self, mapping: MappingResult) -> bool:
        """Update raw browser view based on mapping from media frame index to raw time.

        If the media frame index is out of bounds of the raw data, moves the raw view
        to the start or end of the raw data.

        Parameters
        ----------
        mapping : MappingResult
            The result of mapping the media frame index to a raw time point.

        Returns
        -------
        bool
            True if the mapping was successful and yielded a valid raw time point,
            False if the media frame index was out of bounds of the raw data.
        """
        match mapping:
            case MappingSuccess(result=raw_time):
                # Video frame index has a corresponding raw time point
                logger.debug(f"Setting raw browser to time: {raw_time:.3f} seconds.")
                self._raw_browser_manager.set_selected_time_no_signal(raw_time)
                return True

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_SMALL):
                # Video frame index is smaller than the first raw time point
                logger.debug(
                    "No raw data for this small video frame, moving raw view to start."
                )
                self._raw_browser_manager.jump_to_start()
                return False

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_LARGE):
                logger.debug(
                    "No raw data for this large video frame, moving raw view to end."
                )
                self._raw_browser_manager.jump_to_end()
                return False

            case _:
                raise ValueError(f"Unexpected mapping result: {mapping}. ")


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


def browse_raw_with_video(
    raw_browser: MNEQtBrowser,
    videos: list[VideoFile],
    aligners: list[RawMediaAligner],
    video_splitter_orientation: Literal["horizontal", "vertical"] = "horizontal",
    video_display_method: Literal["image_item", "image_view"] | None = None,
    show: bool = True,
    max_sync_fps: int = 10,
    parent: QObject | None = None,
) -> SyncedRawMediaBrowser:
    """Synchronize MNE raw data browser with a video browser.

    Parameters
    ----------
    raw_browser : mne_qt_browser.figure.MNEQtBrowser
        The MNE raw data browser object to be synchronized with the video browser.
        This can be created with 'plot' method of MNE raw data object when using qt
        backend.
    videos : list[VideoFile]
        The video file object(s) to be displayed in the video browser.
    aligners : list[RawMediaAligner]
        A list of `RawMediaAligner` instances, one for each video file.
        Each aligner provides the mapping between raw data time points and video frames
        for the corresponding video file. The order of the aligners must match the order
        of the video files in the `videos` parameter.
    video_splitter_orientation : Literal["horizontal", "vertical"], optional
        Whether to show multiple videos in a horizontal or vertical layout.
        This has no effect if only one video is provided.
    video_display_method : Literal["image_item", "image_view"], optional
        The display method to use for the video browser. By default, "image_item" is
        used if more than one video is provided, otherwise "image_view".
    max_sync_fps : int, optional
        The maximum frames per second for synchronizing the raw data browser and video
        browser. This determines how often the synchronization updates can happen and
        has an effect on the performance.
    show : bool, optional
        Whether to show the raw data browser immediately upon instantiation,
        by default True.
    parent : QObject, optional
        The parent QObject for this synchronized browser, by default None.

    Returns
    -------
    SyncedRawMediaBrowser
        An instance of `SyncedRawMediaBrowser`, a Qt controller object that handles
        synchronization between the raw data browser and the video browser.
    """
    # Set up the video browser.
    if video_display_method is None:
        # Save space in the UI by excluding histogram with multiple videos.
        display_method = "image_item" if len(videos) > 1 else "image_view"
    else:
        display_method = video_display_method
    video_browser = VideoBrowser(
        videos,
        show_sync_status=True,
        parent=None,
        display_method=display_method,
        video_splitter_orientation=video_splitter_orientation,
    )
    return SyncedRawMediaBrowser(
        raw_browser,
        video_browser,
        aligners,
        media_browser_title="Video Browser",
        show=show,
        max_sync_fps=max_sync_fps,
        parent=parent,
    )


def browse_raw_with_audio(
    raw_browser: MNEQtBrowser,
    audio: AudioFile,
    aligner: RawMediaAligner,
    show: bool = True,
    max_sync_fps: int = 10,
    parent: QObject | None = None,
) -> SyncedRawMediaBrowser:
    """Synchronize MNE raw data browser with a audio browser.

    Parameters
    ----------
    raw_browser : mne_qt_browser.figure.MNEQtBrowser
        The MNE raw data browser object to be synchronized with the video browser.
        This can be created with 'plot' method of MNE raw data object when using qt
        backend.
    audio : AudioFile
        The audio file object to be displayed in the audio browser.
    aligner : RawMediaAligner
        A `RawMediaAligner` instance that provides the mapping between raw data time
        points and audio samples for the audio file.
    max_sync_fps : int, optional
        The maximum frames per second for synchronizing the raw data browser and audio
        browser. This determines how often the synchronization updates can happen and
        has an effect on the performance.
    show : bool, optional
        Whether to show the raw data browser immediately upon instantiation,
        by default True.
    parent : QObject, optional
        The parent QObject for this synchronized browser, by default None.

    Returns
    -------
    SyncedRawMediaBrowser
        An instance of `SyncedRawMediaBrowser`, a Qt controller object that handles
        synchronization between the raw data browser and the audio browser.
    """
    # Set up the audio browser.
    audio_browser = AudioBrowser(audio, parent=None)
    return SyncedRawMediaBrowser(
        raw_browser,
        audio_browser,
        [aligner],
        media_browser_title="Audio Browser",
        show=show,
        max_sync_fps=max_sync_fps,
        parent=parent,
    )
