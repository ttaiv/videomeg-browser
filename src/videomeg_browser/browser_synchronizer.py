"""Contains a QObject that synchronizes multiple SyncableBrowser instances.

Synchronization is based on choosing one primary browser and using TimestampAligner
instances to map positions between the primary browser and other secondary browsers.

There are two main synchronization scenarios:
1) When the position in the primary browser changes, all secondary browsers are updated
   to match the new position in the primary browser.
2) When the position in one of the secondary browsers changes, the first step is to
   update the primary browser to match the new position in the secondary browser.
   After that, all other secondary browsers are updated to match the new position in
   the primary browser.
"""

import functools
import logging

from qtpy.QtCore import QElapsedTimer, QObject, QTimer, Signal, Slot  # type: ignore

from .browsers.syncable_browser import (
    SyncableBrowser,
    SyncableBrowserObject,
    SyncableBrowserWidget,
    SyncStatus,
)
from .timestamp_aligner import (
    MapFailureReason,
    MappingFailure,
    MappingResult,
    MappingSuccess,
    TimestampAligner,
)

logger = logging.getLogger(__name__)


class BrowserSynchronizer(QObject):
    """Synchronizes browsers using provided timestamp aligners.

    Parameters
    ----------
    primary_browser : SyncableBrowser
        The primary browser used as the reference for synchronization.
    secondary_browsers : list[SyncableBrowser]
        The secondary browsers to be synchronized with the primary browser
        and each other through the primary browser.
    aligners : list[list[TimeStampAligner]]
        A list of lists of `TimestampAligner` instances. aligners[i][j] provides
        the mapping between the primary browser and media samples for the j-th media
        file in the i-th secondary browser.
    max_sync_fps : int, optional
        The maximum frames per second for synchronization updates.
        This determines how often the synchronization updates can happen and
        has an effect on the performance.
    throttle_primary_browser : bool, optional
        Whether to throttle updates due to changes in the primary browser.
        This can be useful if the primary browser is updated very frequently.
    parent : QObject, optional
        The parent QObject for this synchronizer, by default None.
    """

    def __init__(
        self,
        primary_browser: SyncableBrowserObject | SyncableBrowserWidget,
        secondary_browsers: list[SyncableBrowserObject | SyncableBrowserWidget],
        aligners: list[list[TimestampAligner]],
        max_sync_fps: int = 10,
        throttle_primary_browser: bool = False,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._primary_browser = primary_browser
        self._secondary_browsers = secondary_browsers
        self._aligners = aligners
        self._max_sync_fps = max_sync_fps

        # Create throttlers that limit the updates due to changes in secondary browsers.
        self._min_sync_interval_ms = int(1000 / max_sync_fps)
        # One throttler for each secondary browser
        self._secondary_throttlers = [
            BufferedThrottler(self._min_sync_interval_ms, parent=self)
            for _ in self._secondary_browsers
        ]

        # When the position in a secondary media browser changes,
        # update all other browsers through throttler.
        for secondary_browser_idx, (secondary_browser, throttler) in enumerate(
            zip(self._secondary_browsers, self._secondary_throttlers)
        ):
            # browser emits (media_idx, position_idx)
            secondary_browser.sigPositionChanged.connect(throttler.trigger)
            # _sync_all_browsers_to_secondary slot takes
            # (browser_idx, media_idx, position_idx)
            throttler.triggered.connect(
                functools.partial(
                    self._sync_all_browsers_to_secondary, secondary_browser_idx
                )
            )

        # When selected time in the primary browser changes, update all secondary
        # browsers. Use throttling if user requested it.
        # NOTE: We assume that primary browser has only one media (index 0).
        if throttle_primary_browser:
            self._primary_throttler = BufferedThrottler(
                self._min_sync_interval_ms, parent=self
            )
            self._primary_browser.sigPositionChanged.connect(
                self._primary_throttler.trigger
            )
            self._primary_throttler.triggered.connect(
                lambda media_idx, pos_idx: self._sync_secondary_browsers_to_primary(
                    pos_idx
                )
            )
        else:
            # Connect signal directly without throttling.
            self._primary_browser.sigPositionChanged.connect(
                lambda media_idx, pos_idx: self._sync_secondary_browsers_to_primary(
                    pos_idx
                )
            )
        # When one browser starts playing, pause all other media browsers
        # to avoid mess in synchronization.
        for browser in self._secondary_browsers + [self._primary_browser]:
            browser.sigPlaybackStateChanged.connect(
                functools.partial(self._on_playback_state_changed, browser)
            )

        # Synchronize secondary browsers to the initial position of the primary browser.
        initial_primary_pos = self._primary_browser.get_current_position(media_idx=0)
        self._sync_secondary_browsers_to_primary(initial_primary_pos)

    @Slot(int, int)
    def _sync_secondary_browsers_to_primary(self, position_idx: int) -> None:
        """Update all the secondary browsers when the position in primary changes."""
        logger.debug(
            "Detected change in primary browser's position, syncing secondary browsers."
        )
        for secondary_browser_idx, aligners in enumerate(self._aligners):
            for media_idx, aligner in enumerate(aligners):
                logger.debug(
                    f"Syncing media {media_idx + 1}/{len(aligners)} of secondary "
                    f"browser {secondary_browser_idx + 1}/"
                    f"{len(self._secondary_browsers)} to primary position idx: "
                    f"{position_idx}"
                )
                mapping = aligner.a_index_to_b_index(position_idx)
                self._update_media_in_browser(
                    self._secondary_browsers[secondary_browser_idx], media_idx, mapping
                )

    def _update_media_in_browser(
        self, browser_to_update: SyncableBrowser, media_idx: int, mapping: MappingResult
    ) -> bool:
        """Update the specified media in the specified browser using mapping result.

        Parameters
        ----------
        browser_to_update : SyncableBrowser
            The media browser to update.
        media_idx : int
            The index of the media in the browser to update.
        mapping : MappingResult
            The result of mapping the primary browser position index to the media
            position index.

        Returns
        -------
        bool
            True if the mapping was successful, False if the mapping failed due to
            source index being out of bounds.

        """
        # NOTE: The signal=False is used to prevent the browser from emitting the
        # sigPositionChanged signal, which would cause an infinite loop of updates.
        match mapping:
            case MappingSuccess(result=position_idx):
                browser_to_update.set_position(position_idx, media_idx, signal=False)
                browser_to_update.set_sync_status(SyncStatus.SYNCHRONIZED, media_idx)
                return True

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_SMALL):
                logger.debug(
                    f"Media on index {media_idx} has no data for this small position, "
                    "moving media to start."
                )
                browser_to_update.jump_to_start(media_idx, signal=False)
                browser_to_update.set_sync_status(SyncStatus.NO_DATA_HERE, media_idx)
                return False

            case MappingFailure(failure_reason=MapFailureReason.INDEX_TOO_LARGE):
                logger.debug(
                    f"Media on index {media_idx} has no data for this large position, "
                    "moving media to end."
                )
                browser_to_update.jump_to_end(media_idx, signal=False)
                browser_to_update.set_sync_status(SyncStatus.NO_DATA_HERE, media_idx)
                return False

            case _:
                raise ValueError(f"Unexpected mapping result: {mapping}. ")

    @Slot(int, int, int)
    def _sync_all_browsers_to_secondary(
        self, browser_idx: int, media_idx: int, position_idx: int
    ) -> None:
        """Update all other browsers when one of the secondary browsers changes."""
        # Get the secondary browser that changed and its aligners.
        browser_that_changed = self._secondary_browsers[browser_idx]
        browser_aligners = self._aligners[browser_idx]

        logger.debug(
            f"Detected change in media {media_idx + 1}/{len(browser_aligners)} of "
            f"secondary browser {browser_idx + 1}/{len(self._secondary_browsers)} to "
            f"position index: {position_idx}. Syncing primary and other secondary "
            "browsers."
        )

        # Update the primary browser view based on the media.
        mapping_to_primary = browser_aligners[media_idx].b_index_to_a_index(
            position_idx
        )
        mapping_success = self._update_media_in_browser(
            self._primary_browser, media_idx=0, mapping=mapping_to_primary
        )
        if mapping_success:
            browser_that_changed.set_sync_status(SyncStatus.SYNCHRONIZED, media_idx)
        else:
            # Signal that there is no data in the primary browser for this position.
            browser_that_changed.set_sync_status(SyncStatus.NO_DATA_THERE, media_idx)
        # Get the resulting primary browser position index and use
        # it to update other media (if any).
        primary_idx = self._primary_browser.get_current_position(media_idx=0)
        for secondary_browser_idx, aligners in enumerate(self._aligners):
            for secondary_media_idx, aligner in enumerate(aligners):
                if (
                    secondary_media_idx == media_idx
                    and secondary_browser_idx == browser_idx
                ):
                    # Skip the media that triggered the change.
                    continue
                logger.debug(
                    f"Syncing media {secondary_media_idx + 1}/{len(aligners)} of "
                    f"secondary browser {secondary_browser_idx + 1}/"
                    f"{len(self._secondary_browsers)} to primary browser index: "
                    f"{primary_idx}"
                )
                mapping = aligner.a_index_to_b_index(primary_idx)
                self._update_media_in_browser(
                    self._secondary_browsers[secondary_browser_idx],
                    secondary_media_idx,
                    mapping,
                )

    @Slot(SyncableBrowser, int, bool)
    def _on_playback_state_changed(
        self, browser: SyncableBrowser, media_idx: int, is_playing: bool
    ) -> None:
        """Prevent other browsers from playing when one starts playing."""
        logger.debug(
            "Received signal about playback state change "
            f"for browser {browser}, is playing: {is_playing}."
        )
        if is_playing:
            self._pause_other_media_browsers(excluded_browser=browser)

    def _pause_other_media_browsers(self, excluded_browser: SyncableBrowser) -> None:
        """Pause all other browsers except the one given."""
        for browser in self._secondary_browsers + [self._primary_browser]:
            if browser is not excluded_browser and browser.is_playing:
                logger.debug(f"Pausing browser {browser}.")
                browser.pause_playback()


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
