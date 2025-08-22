"""Contains the base class for syncable media browser widgets."""

from enum import Enum

from qtpy.QtCore import Signal  # type: ignore
from qtpy.QtWidgets import QWidget


class SyncStatus(Enum):
    """Tells the sync status of the video and raw data."""

    SYNCHRONIZED = "synchronized"  # Video and raw data are synchronized
    NO_RAW_DATA = "no_raw_data"  # No raw (meg/eeg) data available for the current frame
    NO_MEDIA_DATA = "no_media_data"  # No media data available for the current raw data


class SyncableMediaBrowser(QWidget):
    """Base class for syncable video and audio browser widgets.

    Defines methods and signal that subclasses must implement.

    NOTE: This class does not inherit from abc.ABC to prevent metaclass conflicts
    with Qt widgets. Instead, it uses `__init_subclass__` to enforce
    implementation of required methods in subclasses.
    """

    # Emits a signal when the displayed frame or sample of any shown media changes.
    sigPositionChanged = Signal(int, int)  # media index, sample index
    sigPlaybackStateChanged = Signal(int, bool)  # media index, is playing

    def __init_subclass__(cls) -> None:
        """Ensure that subclasses implement required methods."""
        if cls.set_position is SyncableMediaBrowser.set_position:
            raise TypeError(f"{cls.__name__} must implement set_position method.")
        if cls.jump_to_end is SyncableMediaBrowser.jump_to_end:
            raise TypeError(f"{cls.__name__} must implement jump_to_end method.")
        if cls.jump_to_start is SyncableMediaBrowser.jump_to_start:
            raise TypeError(f"{cls.__name__} must implement jump_to_start method.")
        if cls.start_playback is SyncableMediaBrowser.start_playback:
            raise TypeError(f"{cls.__name__} must implement start_playback method.")
        if cls.pause_playback is SyncableMediaBrowser.pause_playback:
            raise TypeError(f"{cls.__name__} must implement pause_playback method.")
        if cls.is_playing is SyncableMediaBrowser.is_playing:
            raise TypeError(f"{cls.__name__} must implement is_playing property.")

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
        raise NotImplementedError(
            "start_playback method must be implemented by subclasses."
        )

    def pause_playback(self) -> None:
        """Pause playback of the currently playing media."""
        raise NotImplementedError(
            "pause_playback method must be implemented by subclasses."
        )

    @property
    def is_playing(self) -> bool:
        """Return whether the media is currently playing."""
        raise NotImplementedError(
            "is_playing property must be implemented by subclasses."
        )
