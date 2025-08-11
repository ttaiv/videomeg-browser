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
    """

    # Emits a signal when the displayed frame of any shown video changes.
    sigPositionChanged = Signal(int, int)  # media index, sample index

    def __init_subclass__(cls) -> None:
        """Ensure that subclasses implement required methods."""
        if cls.set_position is SyncableMediaBrowser.set_position:
            raise TypeError(f"{cls.__name__} must implement set_position method.")
        if cls.jump_to_end is SyncableMediaBrowser.jump_to_end:
            raise TypeError(f"{cls.__name__} must implement jump_to_end method.")
        if cls.jump_to_start is SyncableMediaBrowser.jump_to_start:
            raise TypeError(f"{cls.__name__} must implement jump_to_start method.")

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
            True if the position was set successfully, False if the index
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
