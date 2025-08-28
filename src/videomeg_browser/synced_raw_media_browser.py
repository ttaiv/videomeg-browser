"""Code for syncing MNE raw data browser with video or audio browser."""

import functools
import logging
from typing import Literal

import mne
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


class BrowserSynchronizer(QObject):
    """Synchronizes MNE raw data browser with one or more media browsers.

    Parameters
    ----------
    raw_browser : mne_qt_browser.figure.MNEQtBrowser
        The MNE raw data browser object to be synchronized with the media browser.
        This can be created with 'plot' method of MNE raw data object when using qt
        backend.
    media_browsers : list[SyncableMediaBrowser]
        The media browsers to be synchronized with the raw data browser.
    aligners : list[list[RawMediaAligner]]
        A list of lists of `RawMediaAligner` instances. aligners[i][j] provides
        the mapping between raw data time points and media samples for the j-th media
        file in the i-th media browser.
    media_browser_titles : list[str]
        Titles for the media browsers. Each title corresponds to a media browser in
        `media_browsers`.
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
        primary_browser: RawBrowserManager,
        secondary_browsers: list[SyncableMediaBrowser],
        aligners: list[list[RawMediaAligner]],
        max_sync_fps: int = 10,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._primary_browser = primary_browser
        self._secondary_browsers = secondary_browsers
        self._aligners = aligners
        self._max_update_fps = max_sync_fps

        self._primary_media_idx = 0

        # Set up synchronization

        # Create throttlers that limit the updates due to media position changes.
        self._min_sync_interval_ms = int(1000 / max_sync_fps)
        # One throttler for each media browser
        self._throttlers = [
            BufferedThrottler(self._min_sync_interval_ms, parent=self)
            for _ in self._secondary_browsers
        ]

        # When the position in a media changes
        # update the raw browser and other media through throttler.
        for browser_idx, (browser, throttler) in enumerate(
            zip(self._secondary_browsers, self._throttlers)
        ):
            # Media browser emits (media_idx, position_idx)
            browser.sigPositionChanged.connect(throttler.trigger)
            # _sync_all_to_media slot takes (browser_idx, media_idx, position_idx)
            throttler.triggered.connect(
                functools.partial(self._on_secondary_browser_change, browser_idx)
            )

        self._primary_browser.sigPositionChanged.connect(
            lambda media, pos: self._on_primary_browser_change(pos)
        )

        # When one browser starts playing, pause all other media browsers
        # to avoid mess in synchronization.
        for browser_idx, browser in enumerate(self._secondary_browsers):
            browser.sigPlaybackStateChanged.connect(
                functools.partial(self._on_playback_state_changed, browser_idx)
            )

        # Consider raw data browser to be the main browser and start by
        # synchronizing the media browsers to the initial raw time.
        initial_position = self._primary_browser.get_current_position(
            self._primary_media_idx
        )
        self._on_primary_browser_change(initial_position)

    def _update_browser(
        self, browser: SyncableMediaBrowser, media_idx: int, mapping: MappingResult
    ) -> bool:
        """Update media browser view based on mapping from raw time to frame/sample."""
        # NOTE: The signal=False is used to prevent the media browser from
        # emitting the sigPositionChanged signal, which would trigger update of the
        # raw browser and cause an infinite loop of updates.
        match mapping:
            case MappingSuccess(result=position_idx):
                # Raw time point has a corresponding media frame/sample.
                assert isinstance(position_idx, int), (
                    f"Mapping success for media browser should contain an "
                    f"integer index, got {type(position_idx)}."
                )
                browser.set_position(position_idx, media_idx, signal=False)
                browser.set_sync_status(SyncStatus.SYNCHRONIZED, media_idx)
                return True

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_SMALL):
                # Raw time stamp is smaller than the first media timestamp.
                logger.debug(
                    f"Media on index {media_idx} has no data for this small raw time "
                    "point, moving media to start."
                )
                browser.jump_to_start(media_idx, signal=False)
                browser.set_sync_status(SyncStatus.NO_MEDIA_DATA, media_idx)
                return False

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_LARGE):
                # Raw time stamp is larger than the last media frame timestamp
                logger.debug(
                    f"Media on index {media_idx} has no data for this large raw time "
                    "point, showing last frame."
                )
                browser.jump_to_end(media_idx, signal=False)
                browser.set_sync_status(SyncStatus.NO_MEDIA_DATA, media_idx)
                return False

            case _:
                raise ValueError(f"Unexpected mapping result: {mapping}. ")

    @Slot(int)
    def _on_primary_browser_change(self, position_idx: int) -> None:
        """Update raw data browser's view and other media when media changes."""
        logger.debug(
            f"Detected change in primary raw browser position to index {position_idx}. "
            "Updating other browsers."
        )
        self.update_secondary_browsers(position_idx, exclude_idx=None)

    @Slot(int, int, int)
    def _on_secondary_browser_change(
        self, browser_idx: int, media_idx: int, position_idx: int
    ) -> None:
        # Fist sync main browser, then others.
        browser_that_changed = self._secondary_browsers[browser_idx]
        aligner = self._aligners[browser_idx][media_idx]

        mapping = aligner.media_sample_index_to_raw_time(position_idx)
        mapping_success = self._update_browser(
            self._primary_browser, self._primary_media_idx, mapping
        )
        if mapping_success:
            browser_that_changed.set_sync_status(SyncStatus.SYNCHRONIZED, media_idx)
        else:
            # Signal that there is no raw data for this video frame index.
            browser_that_changed.set_sync_status(SyncStatus.NO_RAW_DATA, media_idx)

        # Get the resulting raw time by asking it from the browser and use
        # it to update other media (if any).
        raw_index = self._primary_browser.get_current_position(self._primary_media_idx)
        self.update_secondary_browsers(raw_index, exclude_idx=(browser_idx, media_idx))

    def update_secondary_browsers(
        self, position_idx: int, exclude_idx: tuple[int, int] | None
    ) -> None:
        _exclude_idx = exclude_idx if exclude_idx is not None else (-1, -1)
        for browser_idx, aligners in enumerate(self._aligners):
            for media_idx, aligner in enumerate(aligners):
                if browser_idx == _exclude_idx[0] and media_idx == _exclude_idx[1]:
                    logger.debug(
                        f"Skipping update of media {media_idx} of secondary "
                        f"browser {browser_idx}."
                    )
                    continue
                logger.debug(
                    f"Syncing media {media_idx + 1}/{len(aligners)} of extra "
                    f"browser {browser_idx + 1}/{len(self._secondary_browsers)} "
                    f"to position index {position_idx}."
                )
                mapping_to_media = aligner.raw_time_to_media_sample_index(position_idx)
                self._update_browser(
                    self._secondary_browsers[browser_idx], media_idx, mapping_to_media
                )

    @Slot(int, int, bool)
    def _on_playback_state_changed(
        self, media_browser_idx: int, media_idx: int, is_playing: bool
    ) -> None:
        """Prevent other media browsers from playing when one starts playing."""
        logger.debug(
            "Received signal about playback state change "
            f"for media browser {media_browser_idx} to {is_playing}."
        )
        if is_playing:
            self._pause_other_media_browsers(media_browser_idx)

    def _pause_other_media_browsers(self, excluded_browser_idx: int) -> None:
        """Pause all other media browsers except the one with the given index."""
        for browser_idx, media_browser in enumerate(self._secondary_browsers):
            if excluded_browser_idx != browser_idx and media_browser.is_playing:
                logger.debug(f"Pausing media browser with index {browser_idx}.")
                media_browser.pause_playback()


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
    raw: mne.io.Raw,
    videos: list[VideoFile],
    aligners: list[RawMediaAligner],
    video_splitter_orientation: Literal["horizontal", "vertical"] = "horizontal",
    show: bool = True,
    max_sync_fps: int = 10,
    parent: QObject | None = None,
):
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
    video_browser = VideoBrowser(
        videos,
        show_sync_status=True,
        parent=None,
        video_splitter_orientation=video_splitter_orientation,
    )

    return Aapa([raw_browser], [raw], [video_browser], [aligners], ["Video Browser"])


def browse_raw_with_audio(
    raw_browser: MNEQtBrowser,
    audio: AudioFile,
    aligner: RawMediaAligner,
    show: bool = True,
    max_sync_fps: int = 10,
    parent: QObject | None = None,
) -> BrowserSynchronizer:
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
    return BrowserSynchronizer(
        raw_browser,
        [audio_browser],
        [[aligner]],
        media_browser_titles=["Audio Browser"],
        show=show,
        max_sync_fps=max_sync_fps,
        parent=parent,
    )


def browse_raw_with_video_and_audio(
    raw_browser: MNEQtBrowser,
    videos: list[VideoFile],
    video_aligners: list[RawMediaAligner],
    audio: AudioFile,
    audio_aligner: RawMediaAligner,
    video_splitter_orientation: Literal["horizontal", "vertical"] = "horizontal",
    max_sync_fps: int = 10,
    show: bool = True,
    parent: QObject | None = None,
) -> BrowserSynchronizer:
    """Synchronize MNE raw data browser with both video and audio browsers.

    Parameters
    ----------
    raw_browser : mne_qt_browser.figure.MNEQtBrowser
        The MNE raw data browser object to be synchronized with the media browser.
        This can be created with 'plot' method of MNE raw data object when using qt
        backend.
    videos : list[VideoFile]
        The video file object(s) to be displayed in the video browser.
    video_aligners : list[RawMediaAligner]
        A list of `RawMediaAligner` instances, one for each video file.
        Each aligner provides the mapping between raw data time points and video frames
        for the corresponding video file. The order of the aligners must match the order
        of the video files in the `videos` parameter.
    audio : AudioFile
        The audio file object to be displayed in the audio browser.
    audio_aligner : RawMediaAligner
        A `RawMediaAligner` instance that provides the mapping between raw data time
        points and audio samples for the audio file.
    video_splitter_orientation : Literal["horizontal", "vertical"], optional
        Whether to show multiple videos in a horizontal or vertical layout.
        This has no effect if only one video is provided.
    max_sync_fps : int, optional
        The maximum frames per second for synchronizing the raw data browser and media
        browser. This determines how often the synchronization updates can happen and
        has an effect on the performance.
    show : bool, optional
        Whether to show the browsers immediately, by default True.
    parent : QObject, optional
        The parent QObject for this synchronized browser, by default None.

    Returns
    -------
    SyncedRawMediaBrowser
        An instance of `SyncedRawMediaBrowser`, a Qt controller object that handles
        synchronization between the raw data browser and the video and audio browsers.
    """
    # Set up the video browser.
    video_browser = VideoBrowser(
        videos,
        show_sync_status=True,
        video_splitter_orientation=video_splitter_orientation,
        parent=None,
    )
    # Set up the audio browser.
    audio_browser = AudioBrowser(audio, parent=None)

    return BrowserSynchronizer(
        raw_browser,
        [video_browser, audio_browser],
        [video_aligners, [audio_aligner]],
        media_browser_titles=["Video Browser", "Audio Browser"],
        show=show,
        max_sync_fps=max_sync_fps,
        parent=parent,
    )


class Aapa(QObject):
    def __init__(
        self,
        raw_browsers: list[MNEQtBrowser],
        raws: list[mne.io.Raw],
        media_browsers: list[SyncableMediaBrowser],
        aligners: list[list[RawMediaAligner]],
        media_browser_titles: list[str],
        show: bool = True,
        max_sync_fps: int = 10,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)  # Add this line
        # Wrap the raw browser to a class that exposes the necessary methods.
        raw_browser_interfaces = [
            RawBrowserInterface(raw_browser) for raw_browser in raw_browsers
        ]
        # Pass interface for manager that contains actual logic for managing the browser
        # in sync with the video browser.
        self._raw_browser_managers = [
            RawBrowserManager(raw_browser_interface, raw)
            for (raw_browser_interface, raw) in zip(raw_browser_interfaces, raws)
        ]

        # Make sure that raw browser visibility matches the `show` parameter.
        for raw_browser_manager in self._raw_browser_managers:
            if show:
                raw_browser_manager.show_browser()
            else:
                raw_browser_manager.hide_browser()

        # Dock the media browsers to the raw data browser.
        self._docks = []
        for media_browser, media_browser_title in zip(
            media_browsers, media_browser_titles
        ):
            dock = QDockWidget(media_browser_title, raw_browsers[0])
            dock.setWidget(media_browser)
            dock.setFloating(True)
            raw_browsers[0].addDockWidget(Qt.RightDockWidgetArea, dock)
            dock.resize(1000, 800)
            if not show:
                dock.hide()
            self._docks.append(dock)

        self._synchronizer = BrowserSynchronizer(
            self._raw_browser_managers[0], media_browsers, aligners, parent=self
        )
