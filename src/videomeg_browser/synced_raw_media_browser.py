"""Code for syncing MNE raw data browser with video or audio browser."""

import logging
from typing import Literal

import mne
from mne_qt_browser.figure import MNEQtBrowser
from qtpy.QtCore import QObject, Qt  # type: ignore
from qtpy.QtWidgets import QDockWidget

from .audio import AudioFile
from .audio_browser import AudioBrowser
from .browser_synchronizer import BrowserSynchronizer
from .raw_browser_manager import RawBrowserInterface, RawBrowserManager
from .syncable_browser import SyncableBrowser
from .timestamp_aligner import TimestampAligner
from .video import VideoFile
from .video_browser import VideoBrowser

logger = logging.getLogger(__name__)


class SyncedRawMediaBrowser(QObject):
    """Synchronizes MNE raw data browser with one or more media browsers.

    Parameters
    ----------
    raw_browser : mne_qt_browser.figure.MNEQtBrowser
        The MNE raw data browser object to be synchronized with the media browser.
        This can be created with 'plot' method of MNE raw data object when using qt
        backend.
    raw : mne.io.Raw
        The MNE raw data object that was used to create the `raw_browser`.
    media_browsers : list[SyncableBrowser]
        The media browsers to be synchronized with the raw data browser.
    aligners : list[list[TimeStampAligner]]
        A list of lists of `TimestampAligner` instances. aligners[i][j] provides
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
        raw_browser: MNEQtBrowser,
        raw: mne.io.Raw,
        media_browsers: list[SyncableBrowser],
        aligners: list[list[TimestampAligner]],
        media_browser_titles: list[str],
        show: bool = True,
        max_sync_fps: int = 10,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._media_browsers = media_browsers

        # Wrap the raw browser to a class that exposes the necessary methods.
        raw_browser_interface = RawBrowserInterface(raw_browser, parent=self)
        # Pass interface for manager that contains actual logic for managing the browser
        # in sync with the video browser.
        self._raw_browser_manager = RawBrowserManager(
            raw_browser_interface, raw, parent=self
        )
        # Make sure that raw browser visibility matches the `show` parameter.
        if show:
            self._raw_browser_manager.show_browser()
        else:
            self._raw_browser_manager.hide_browser()

        # Dock the media browsers to the raw data browser.
        self._docks = []
        for media_browser, media_browser_title in zip(
            self._media_browsers, media_browser_titles
        ):
            dock = QDockWidget(media_browser_title, raw_browser)
            dock.setWidget(media_browser)
            dock.setFloating(True)
            raw_browser.addDockWidget(Qt.RightDockWidgetArea, dock)
            dock.resize(1000, 800)
            if not show:
                dock.hide()
            self._docks.append(dock)

        # Set up the synchronizer that keeps the raw and media browsers in sync.
        self._synchronizer = BrowserSynchronizer(
            self._raw_browser_manager,
            self._media_browsers,
            aligners,
            max_sync_fps,
            parent=self,
        )

    def show(self) -> None:
        """Show the synchronized raw and video browsers."""
        self._raw_browser_manager.show_browser()
        for dock in self._docks:
            dock.show()


def browse_raw_with_video(
    raw_browser: MNEQtBrowser,
    raw: mne.io.Raw,
    videos: list[VideoFile],
    aligners: list[TimestampAligner],
    video_splitter_orientation: Literal["horizontal", "vertical"] = "horizontal",
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
    raw : mne.io.Raw
        The MNE raw data object that was used to create the `raw_browser`.
    videos : list[VideoFile]
        The video file object(s) to be displayed in the video browser.
    aligners : list[TimestampAligner]
        A list of `TimestampAligner` instances, one for each video file.
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
    return SyncedRawMediaBrowser(
        raw_browser,
        raw,
        [video_browser],
        [aligners],
        media_browser_titles=["Video Browser"],
        show=show,
        max_sync_fps=max_sync_fps,
        parent=parent,
    )


def browse_raw_with_audio(
    raw_browser: MNEQtBrowser,
    raw: mne.io.Raw,
    audio: AudioFile,
    aligner: TimestampAligner,
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
    raw : mne.io.Raw
        The MNE raw data object that was used to create the `raw_browser`.
    audio : AudioFile
        The audio file object to be displayed in the audio browser.
    aligner : TimestampAligner
        A `TimestampAligner` instance that provides the mapping between raw data time
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
        raw,
        [audio_browser],
        [[aligner]],
        media_browser_titles=["Audio Browser"],
        show=show,
        max_sync_fps=max_sync_fps,
        parent=parent,
    )


def browse_raw_with_video_and_audio(
    raw_browser: MNEQtBrowser,
    raw: mne.io.Raw,
    videos: list[VideoFile],
    video_aligners: list[TimestampAligner],
    audio: AudioFile,
    audio_aligner: TimestampAligner,
    video_splitter_orientation: Literal["horizontal", "vertical"] = "horizontal",
    max_sync_fps: int = 10,
    show: bool = True,
    parent: QObject | None = None,
) -> SyncedRawMediaBrowser:
    """Synchronize MNE raw data browser with both video and audio browsers.

    Parameters
    ----------
    raw_browser : mne_qt_browser.figure.MNEQtBrowser
        The MNE raw data browser object to be synchronized with the media browser.
        This can be created with 'plot' method of MNE raw data object when using qt
        backend.
    raw : mne.io.Raw
        The MNE raw data object that was used to create the `raw_browser`.
    videos : list[VideoFile]
        The video file object(s) to be displayed in the video browser.
    video_aligners : list[TimestampAligner]
        A list of `TimestampAligner` instances, one for each video file.
        Each aligner provides the mapping between raw data time points and video frames
        for the corresponding video file. The order of the aligners must match the order
        of the video files in the `videos` parameter.
    audio : AudioFile
        The audio file object to be displayed in the audio browser.
    audio_aligner : TimestampAligner
        A `TimestampAligner` instance that provides the mapping between raw data time
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

    return SyncedRawMediaBrowser(
        raw_browser,
        raw,
        [video_browser, audio_browser],
        [video_aligners, [audio_aligner]],
        media_browser_titles=["Video Browser", "Audio Browser"],
        show=show,
        max_sync_fps=max_sync_fps,
        parent=parent,
    )
