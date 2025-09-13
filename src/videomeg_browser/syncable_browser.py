"""Contains interface for syncable browser widgets."""

from enum import Enum

from qtpy.QtCore import QObject, Signal  # type: ignore
from qtpy.QtWidgets import QWidget


class SyncStatus(Enum):
    """Synchronization status for data item in a browser."""

    # Displayed data is in sync with its counterpart
    SYNCHRONIZED = "synchronized"
    # There is no suitable data for the selected position in the counterpart
    NO_DATA_HERE = "no_data_here"
    # There is no suitable data in the counterpart for the selected position here
    NO_DATA_THERE = "no_data_there"


class SyncableBrowser:
    """Abstract base interface for browser widgets that can be synchronized.

    Defines methods that must be implemented by subclasses. Required signals
    are defined in two helper classes `SyncableBrowserObject` and
    `SyncableBrowserWidget` below. Actual subclass should inherit from one of these!

    NOTE: This class does not inherit from abc.ABC to prevent metaclass conflicts
    with Qt widgets. Instead, it uses `__init_subclass__` to enforce
    implementation of required methods in subclasses.
    """

    def __init_subclass__(cls) -> None:
        """Ensure that subclasses implement required methods."""
        # Skip validation for mixin classes that are meant to be inherited from
        if cls.__name__ in ("SyncableBrowserObject", "SyncableBrowserWidget"):
            return

        if cls.set_position is SyncableBrowser.set_position:
            raise TypeError(f"{cls.__name__} must implement set_position method.")
        if cls.jump_to_end is SyncableBrowser.jump_to_end:
            raise TypeError(f"{cls.__name__} must implement jump_to_end method.")
        if cls.jump_to_start is SyncableBrowser.jump_to_start:
            raise TypeError(f"{cls.__name__} must implement jump_to_start method.")
        if cls.get_current_position is SyncableBrowser.get_current_position:
            raise TypeError(
                f"{cls.__name__} must implement get_current_position method."
            )

    def set_position(
        self, position_idx: int, media_idx: int, signal: bool = True
    ) -> bool:
        """Set the current position (frame/sample) for the specified media.

        Parameters
        ----------
        position_idx : int
            The position index to display
            (frame index for video, sample index for audio).
        media_idx : int
            Index of the media to update.
        signal : bool, optional
            Whether to emit sigPositionChanged signal, by default True.

        Returns
        -------
        bool
            True if the position was set successfully, False if the position index
            is out of bounds.
        """
        raise NotImplementedError(
            "set_position method must be implemented by subclasses."
        )

    def jump_to_end(self, media_idx: int, signal: bool = True) -> None:
        """Display the last frame/sample of the specified media.

        Parameters
        ----------
        media_idx : int
            Index of the media to jump to the end.
        signal : bool, optional
            Whether to emit sigPositionChanged signal, by default True.
        """
        raise NotImplementedError(
            "jump_to_end method must be implemented by subclasses."
        )

    def jump_to_start(self, media_idx: int, signal: bool = True) -> None:
        """Display the first frame/sample of the specified media.

        Parameters
        ----------
        media_idx : int
            Index of the media to jump to the start.
        signal : bool, optional
            Whether to emit sigPositionChanged signal, by default True.
        """
        raise NotImplementedError(
            "jump_to_start method must be implemented by subclasses."
        )

    def set_sync_status(self, status: SyncStatus, media_idx: int) -> None:
        """Set the synchronization status for the specified media.

        Parameters
        ----------
        status : SyncStatus
            The synchronization status to set.
        media_idx : int
            Index of the media to update.
        """
        # Default implementation does nothing.
        pass

    def start_playback(self, media_idx: int) -> None:
        """Start playing the specified media.

        Parameters
        ----------
        media_idx : int
            Index of the media to start playing.
        """
        pass  # Empty default implementation

    def pause_playback(self) -> None:
        """Pause playback of the currently playing media."""
        pass  # Empty default implementation

    def get_current_position(self, media_idx: int) -> int:
        """Return the current position index of the specified media."""
        raise NotImplementedError(
            "get_current_position method must be implemented by subclasses."
        )

    @property
    def is_playing(self) -> bool:
        """Return whether the media is currently playing."""
        return False  # Default implementation


# Below are helper classes to combine SyncableBrowser with QObject or QWidget.
# Signals are defined in these classes as they should be defined in QObject subclasses.


class SyncableBrowserObject(SyncableBrowser, QObject):
    """A helper class to combine SyncableBrowser and QObject."""

    # Signal for change in displayed media position
    # (video frame, audio sample, raw data sample, etc.)
    sigPositionChanged = Signal(int, int)  # media index, sample index
    # Signal for change in playback state (playing/paused) of media
    sigPlaybackStateChanged = Signal(int, bool)  # media index, is playing

    def __init__(self, parent: QObject | None = None) -> None:
        SyncableBrowser.__init__(self)
        QObject.__init__(self, parent)


class SyncableBrowserWidget(SyncableBrowser, QWidget):
    """A helper class to combine SyncableBrowser and QWidget."""

    # Signal for change in displayed media position
    # (video frame, audio sample, raw data sample, etc.)
    sigPositionChanged = Signal(int, int)  # media index, sample index
    # Signal for change in playback state (playing/paused) of media
    sigPlaybackStateChanged = Signal(int, bool)  # media index, is playing

    def __init__(self, parent: QWidget | None = None) -> None:
        SyncableBrowser.__init__(self)
        QWidget.__init__(self, parent)
