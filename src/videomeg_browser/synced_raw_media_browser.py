"""Code for syncing MNE raw data browser with video or audio browser."""

import logging
from typing import Literal

import mne
from mne_qt_browser.figure import MNEQtBrowser
from qtpy.QtCore import QCoreApplication, QObject, Qt  # type: ignore
from qtpy.QtWidgets import QApplication, QDockWidget

from .browser_synchronizer import BrowserSynchronizer
from .browsers.audio_browser import AudioBrowser
from .browsers.raw_browser_manager import RawBrowserInterface, RawBrowserManager
from .browsers.syncable_browser import (
    SyncableBrowserObject,
    SyncableBrowserWidget,
)
from .browsers.video_browser import VideoBrowser
from .media.audio import AudioFile
from .media.video import VideoFile
from .timestamp_aligner import TimestampAligner

logger = logging.getLogger(__name__)


class SyncedRawMediaBrowser(QObject):
    """Synchronizes MNE raw data browser with one or more media browsers.

    Parameters
    ----------
    primary_raw_browser : mne_qt_browser.figure.MNEQtBrowser
        The primary MNE raw data browser object to be synchronized with other browsers.
        This can be created with 'plot' method of MNE raw data object when using qt
        backend. The primary raw browser will act as a master browser that controls the
        time point shown in the secondary browsers.
    primary_raw : mne.io.Raw
        The MNE raw data object that was used to create the `primary_raw_browser`.
    media_browsers : list[SyncableBrowserWidget]
        The media browsers to be synchronized with the primary raw data browser.
    media_aligners : list[list[TimeStampAligner]]
        A list of lists of `TimestampAligner` instances. aligners[i][j] provides
        the mapping between primary raw data samples and media samples for the j-th
        media file in the i-th media browser.
    media_browser_titles : list[str]
        Titles for the media browsers. Each title corresponds to a media browser in
        `media_browsers`.
    secondary_raw_browsers : list[mne_qt_browser.figure.MNEQtBrowser] | None, optional
        Optional list of secondary MNE raw data browsers to be synchronized with the
        primary raw data browser.
    secondary_raws : list[mne.io.Raw] | None, optional
        The MNE raw data objects that were used to create the `secondary_raw_browsers`.
    raw_aligners : list[TimestampAligner] | None, optional
        A list of `TimestampAligner` instances, one for each secondary raw data
        browser. Each aligner provides the mapping between primary raw data samples and
        secondary raw data samples for the corresponding secondary raw data browser.
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
        primary_raw_browser: MNEQtBrowser,
        primary_raw: mne.io.Raw,
        media_browsers: list[SyncableBrowserWidget],
        media_aligners: list[list[TimestampAligner]],
        media_browser_titles: list[str],
        secondary_raw_browsers: list[MNEQtBrowser] | None = None,
        secondary_raws: list[mne.io.Raw] | None = None,
        raw_aligners: list[TimestampAligner] | None = None,
        max_sync_fps: int = 10,
        show: bool = True,
        parent: QObject | None = None,
    ) -> None:
        _validate_secondary_raw_parameters(
            secondary_raw_browsers, secondary_raws, raw_aligners
        )
        super().__init__(parent=parent)

        # Wrap the primary raw browser in a manager that enables synchronization.
        self._raw_browser_manager = self._prepare_raw_browser_manager(
            primary_raw_browser, primary_raw, show
        )

        # Dock the media browsers to the primary raw data browser.
        self._docks = []
        for media_browser, media_browser_title in zip(
            media_browsers, media_browser_titles
        ):
            dock = QDockWidget(media_browser_title, primary_raw_browser)
            dock.setWidget(media_browser)
            dock.setFloating(True)
            primary_raw_browser.addDockWidget(Qt.RightDockWidgetArea, dock)
            dock.resize(1000, 800)
            if not show:
                dock.hide()
            self._docks.append(dock)

        # Add media browsers to a list of secondary browsers.
        secondary_browsers: list[SyncableBrowserObject | SyncableBrowserWidget] = list(
            media_browsers
        )
        aligners = media_aligners  # one aligner for each secondary browser
        # If secondary raw browsers are provided, wrap them in managers and add to the
        # list of secondary browsers.
        if (
            secondary_raw_browsers is not None
            and secondary_raws is not None
            and raw_aligners is not None
        ):
            for sec_raw_browser, sec_raw, aligner in zip(
                secondary_raw_browsers, secondary_raws, raw_aligners
            ):
                sec_raw_manager = self._prepare_raw_browser_manager(
                    sec_raw_browser, sec_raw, show
                )
                secondary_browsers.append(sec_raw_manager)
                aligners.append([aligner])

        # Set up the synchronizer that keeps the primary and secondary browsers in sync.
        self._synchronizer = BrowserSynchronizer(
            primary_browser=self._raw_browser_manager,
            secondary_browsers=secondary_browsers,
            aligners=aligners,
            max_sync_fps=max_sync_fps,
            throttle_primary_browser=False,
            parent=self,
        )

    def show(self) -> None:
        """Show the synchronized raw and video browsers."""
        self._raw_browser_manager.show_browser()
        for dock in self._docks:
            dock.show()

    def _prepare_raw_browser_manager(
        self, raw_browser: MNEQtBrowser, raw: mne.io.Raw, show: bool
    ) -> RawBrowserManager:
        """Wrap the raw browser in a manager that enables synchronization."""
        # Wrap the raw browser to a class that exposes the necessary methods.
        raw_browser_interface = RawBrowserInterface(raw_browser, parent=self)
        # Pass interface for manager that implements SyncableBrowserObject interface.
        raw_browser_manager = RawBrowserManager(raw_browser_interface, raw, parent=self)
        # Make sure that raw browser visibility matches the `show` parameter.
        if show:
            raw_browser_manager.show_browser()
        else:
            raw_browser_manager.hide_browser()

        return raw_browser_manager


def browse_raw_with_video(
    raw_browser: MNEQtBrowser,
    raw: mne.io.Raw,
    videos: list[VideoFile],
    video_aligners: list[TimestampAligner],
    secondary_raw_browsers: list[MNEQtBrowser] | None = None,
    secondary_raws: list[mne.io.Raw] | None = None,
    raw_aligners: list[TimestampAligner] | None = None,
    video_splitter_orientation: Literal["horizontal", "vertical"] = "horizontal",
    show: bool = True,
    max_sync_fps: int = 10,
    parent: QObject | None = None,
    block: bool = True,
) -> SyncedRawMediaBrowser:
    """Synchronize MNE raw data browser(s) with a video browser.

    Parameters
    ----------
    raw_browser : mne_qt_browser.figure.MNEQtBrowser
        The primary MNE raw data browser object to be synchronized with the video
        browser. This can be created with 'plot' method of MNE raw data object when
        using qt backend. This will act as the master browser controlling the
        synchronization.
    raw : mne.io.Raw
        The MNE raw data object that was used to create the `raw_browser`.
    videos : list[VideoFile]
        The video file object(s) to be displayed in the video browser.
    video_aligners : list[TimestampAligner]
        A list of `TimestampAligner` instances, one for each video file.
        Each aligner provides the mapping between raw data time points and video frames
        for the corresponding video file. The order of the aligners must match the order
        of the video files in the `videos` parameter.
    secondary_raw_browsers : list[mne_qt_browser.figure.MNEQtBrowser] | None, optional
        Optional list of secondary MNE raw data browsers to be synchronized with the
        primary raw data browser and video browser. If provided, `secondary_raws` and
        `raw_aligners` must also be provided.
    secondary_raws : list[mne.io.Raw] | None, optional
        The MNE raw data objects that were used to create the `secondary_raw_browsers`.
        Must have the same length as `secondary_raw_browsers` if provided.
    raw_aligners : list[TimestampAligner] | None, optional
        A list of `TimestampAligner` instances, one for each secondary raw data
        browser. Each aligner provides the mapping between primary raw data time points
        and secondary raw data time points for the corresponding secondary raw browser.
        Must have the same length as `secondary_raw_browsers` if provided.
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
    block : bool, optional
        Whether to automatically start the Qt event loop and block until the
        application is closed, by default True. If False, the caller is responsible
        for managing the QApplication event loop.

    Returns
    -------
    SyncedRawMediaBrowser
        An instance of `SyncedRawMediaBrowser`, a Qt controller object that handles
        synchronization between the raw data browser(s) and the video browser.
    """
    _validate_secondary_raw_parameters(
        secondary_raw_browsers, secondary_raws, raw_aligners
    )
    app = _get_existing_qapplication()

    # Set up the video browser.
    video_browser = VideoBrowser(
        videos,
        show_sync_status=True,
        parent=None,
        video_splitter_orientation=video_splitter_orientation,
    )
    browser = SyncedRawMediaBrowser(
        raw_browser,
        raw,
        [video_browser],
        [video_aligners],
        media_browser_titles=["Video Browser"],
        secondary_raw_browsers=secondary_raw_browsers,
        secondary_raws=secondary_raws,
        raw_aligners=raw_aligners,
        show=show,
        max_sync_fps=max_sync_fps,
        parent=parent,
    )

    if block:
        # Start the Qt event loop to display the browsers.
        app.exec_()

    return browser


def browse_raw_with_audio(
    raw_browser: MNEQtBrowser,
    raw: mne.io.Raw,
    audio: AudioFile,
    audio_aligner: TimestampAligner,
    secondary_raw_browsers: list[MNEQtBrowser] | None = None,
    secondary_raws: list[mne.io.Raw] | None = None,
    raw_aligners: list[TimestampAligner] | None = None,
    show: bool = True,
    max_sync_fps: int = 10,
    parent: QObject | None = None,
    block: bool = True,
) -> SyncedRawMediaBrowser:
    """Synchronize MNE raw data browser(s) with an audio browser.

    Parameters
    ----------
    raw_browser : mne_qt_browser.figure.MNEQtBrowser
        The primary MNE raw data browser object to be synchronized with the audio
        browser. This can be created with 'plot' method of MNE raw data object when
        using qt backend. This will act as the master browser controlling the
        synchronization.
    raw : mne.io.Raw
        The MNE raw data object that was used to create the `raw_browser`.
    audio : AudioFile
        The audio file object to be displayed in the audio browser.
    audio_aligner : TimestampAligner
        A `TimestampAligner` instance that provides the mapping between raw data time
        points and audio samples for the audio file.
    secondary_raw_browsers : list[mne_qt_browser.figure.MNEQtBrowser] | None, optional
        Optional list of secondary MNE raw data browsers to be synchronized with the
        primary raw data browser and audio browser. If provided, `secondary_raws` and
        `raw_aligners` must also be provided.
    secondary_raws : list[mne.io.Raw] | None, optional
        The MNE raw data objects that were used to create the `secondary_raw_browsers`.
        Must have the same length as `secondary_raw_browsers` if provided.
    raw_aligners : list[TimestampAligner] | None, optional
        A list of `TimestampAligner` instances, one for each secondary raw data
        browser. Each aligner provides the mapping between primary raw data time points
        and secondary raw data time points for the corresponding secondary raw browser.
        Must have the same length as `secondary_raw_browsers` if provided.
    max_sync_fps : int, optional
        The maximum frames per second for synchronizing the raw data browser and audio
        browser. This determines how often the synchronization updates can happen and
        has an effect on the performance.
    show : bool, optional
        Whether to show the raw data browser immediately upon instantiation,
        by default True.
    parent : QObject, optional
        The parent QObject for this synchronized browser, by default None.
    block : bool, optional
        Whether to automatically start the Qt event loop and block until the
        application is closed, by default True. If False, the caller is responsible
        for managing the QApplication event loop.

    Returns
    -------
    SyncedRawMediaBrowser
        An instance of `SyncedRawMediaBrowser`, a Qt controller object that handles
        synchronization between the raw data browser(s) and the audio browser.
    """
    _validate_secondary_raw_parameters(
        secondary_raw_browsers, secondary_raws, raw_aligners
    )
    app = _get_existing_qapplication()

    # Set up the audio browser.
    audio_browser = AudioBrowser(audio, parent=None)
    browser = SyncedRawMediaBrowser(
        raw_browser,
        raw,
        [audio_browser],
        [[audio_aligner]],
        media_browser_titles=["Audio Browser"],
        secondary_raw_browsers=secondary_raw_browsers,
        secondary_raws=secondary_raws,
        raw_aligners=raw_aligners,
        show=show,
        max_sync_fps=max_sync_fps,
        parent=parent,
    )

    if block:
        # Start the Qt event loop to display the browsers.
        app.exec_()

    return browser


def browse_raw_with_video_and_audio(
    raw_browser: MNEQtBrowser,
    raw: mne.io.Raw,
    videos: list[VideoFile],
    video_aligners: list[TimestampAligner],
    audio: AudioFile,
    audio_aligner: TimestampAligner,
    secondary_raw_browsers: list[MNEQtBrowser] | None = None,
    secondary_raws: list[mne.io.Raw] | None = None,
    raw_aligners: list[TimestampAligner] | None = None,
    video_splitter_orientation: Literal["horizontal", "vertical"] = "horizontal",
    max_sync_fps: int = 10,
    show: bool = True,
    parent: QObject | None = None,
    block: bool = True,
) -> SyncedRawMediaBrowser:
    """Synchronize MNE raw data browser(s) with both video and audio browsers.

    Parameters
    ----------
    raw_browser : mne_qt_browser.figure.MNEQtBrowser
        The primary MNE raw data browser object to be synchronized with the media
        browsers. This can be created with 'plot' method of MNE raw data object when
        using qt backend. This will act as the master browser controlling the
        synchronization.
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
    secondary_raw_browsers : list[mne_qt_browser.figure.MNEQtBrowser] | None, optional
        Optional list of secondary MNE raw data browsers to be synchronized with the
        primary raw data browser and media browsers. If provided, `secondary_raws` and
        `raw_aligners` must also be provided.
    secondary_raws : list[mne.io.Raw] | None, optional
        The MNE raw data objects that were used to create the `secondary_raw_browsers`.
        Must have the same length as `secondary_raw_browsers` if provided.
    raw_aligners : list[TimestampAligner] | None, optional
        A list of `TimestampAligner` instances, one for each secondary raw data
        browser. Each aligner provides the mapping between primary raw data time points
        and secondary raw data time points for the corresponding secondary raw browser.
        Must have the same length as `secondary_raw_browsers` if provided.
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
    block : bool, optional
        Whether to automatically start the Qt event loop and block until the
        application is closed, by default True. If False, the caller is responsible
        for managing the QApplication event loop.

    Returns
    -------
    SyncedRawMediaBrowser
        An instance of `SyncedRawMediaBrowser`, a Qt controller object that handles
        synchronization between the raw data browser(s) and the video and audio
        browsers.
    """
    _validate_secondary_raw_parameters(
        secondary_raw_browsers, secondary_raws, raw_aligners
    )
    app = _get_existing_qapplication()

    # Set up the video browser.
    video_browser = VideoBrowser(
        videos,
        show_sync_status=True,
        video_splitter_orientation=video_splitter_orientation,
        parent=None,
    )
    # Set up the audio browser.
    audio_browser = AudioBrowser(audio, parent=None)

    # Set up synchronized browser.
    browser = SyncedRawMediaBrowser(
        raw_browser,
        raw,
        [video_browser, audio_browser],
        [video_aligners, [audio_aligner]],
        media_browser_titles=["Video Browser", "Audio Browser"],
        secondary_raw_browsers=secondary_raw_browsers,
        secondary_raws=secondary_raws,
        raw_aligners=raw_aligners,
        show=show,
        max_sync_fps=max_sync_fps,
        parent=parent,
    )

    if block:
        # Start the Qt event loop to display the browsers.
        app.exec_()

    return browser


def _validate_secondary_raw_parameters(
    secondary_raw_browsers: list[MNEQtBrowser] | None,
    secondary_raws: list[mne.io.Raw] | None,
    raw_aligners: list[TimestampAligner] | None,
) -> None:
    """Validate secondary raw browser parameters.

    Parameters
    ----------
    secondary_raw_browsers : list[MNEQtBrowser] | None
        Optional list of secondary raw browsers.
    secondary_raws : list[mne.io.Raw] | None
        Optional list of secondary raw data objects.
    raw_aligners : list[TimestampAligner] | None
        Optional list of aligners for secondary raw browsers.

    Raises
    ------
    ValueError
        If parameter validation fails.
    """
    # Validate that either all secondary raw variables are None or all are given.
    secondary_vars = [secondary_raw_browsers, secondary_raws, raw_aligners]
    if not (
        all(x is None for x in secondary_vars)
        or all(x is not None for x in secondary_vars)
    ):
        raise ValueError(
            "Either all of secondary_raw_browsers, secondary_raws, and raw_aligners"
            " must be None, or all must be provided (not None)."
        )

    # Validate input lengths if secondary raw browsers are provided
    if secondary_raw_browsers is not None:
        assert secondary_raws is not None
        assert raw_aligners is not None
        if len(secondary_raw_browsers) != len(secondary_raws):
            raise ValueError(
                "secondary_raw_browsers and secondary_raws must have the same length"
            )
        if len(secondary_raw_browsers) != len(raw_aligners):
            raise ValueError(
                "secondary_raw_browsers and raw_aligners must have the same length"
            )


def _get_existing_qapplication() -> QCoreApplication:
    """Return the current QApplication instance or raise error if none exists."""
    app = QApplication.instance()
    if app is None:
        raise RuntimeError(
            "No existing QApplication instance found even though raw.plot() should "
            "create one. Ensure that you call raw.plot() with qt backend before "
            "this function."
        )
    return app
