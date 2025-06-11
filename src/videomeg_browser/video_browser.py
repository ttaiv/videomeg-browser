try:
    from qtpy.QtCore import Qt
except Exception as exc:
    if exc.__class__.__name__ == "QtBindingsNotFoundError":
        raise ImportError(
            "No Qt binding found, please install PyQt6, PyQt5, PySide6, or PySide2"
        ) from None
    else:
        raise


import logging
from enum import Enum, auto

import pyqtgraph as pg
from qtpy.QtCore import QTimer, Signal, Slot
from qtpy.QtWidgets import (
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .video import VideoFile

logger = logging.getLogger(__name__)

pg.setConfigOptions(imageAxisOrder="row-major")


class SyncStatus(Enum):
    """Tells the sync status of the video and raw data."""

    SYNCHRONIZED = auto()  # Video and raw data are synchronized
    NO_RAW_DATA = auto()  # No raw data available for the current frame
    NO_VIDEO_DATA = auto()  # No video data available for the current raw data


class VideoBrowser(QWidget):
    """A browser for viewing video frames.

    Parameters
    ----------
    video : VideoFile
        The video file to be displayed.
    show_sync_status : bool, optional
        Whether to show a label indicating the synchronization status of the video and
        and raw data, by default False
    parent : QWidget, optional
        The parent widget for this browser, by default None
    """

    # Emits a signal when the displayed frame changes.
    # The signal carries the index of the new currently displayed frame.
    sigFrameChanged = Signal(int)

    def __init__(
        self,
        video: VideoFile,
        show_sync_status: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self.video = video
        self.show_sync_status = show_sync_status

        self.current_frame_idx = 0
        self.is_playing = False  # Whether the frame updates are currently automatic

        # Set up timer that allow automatic frame updates (playing the video)
        self.play_timer = QTimer(parent=self)
        # Milliseconds between frame updates so that video is played with original fps
        self.play_timer_interval_ms = int(1000 / video.fps)
        self.play_timer.setInterval(self.play_timer_interval_ms)
        self.play_timer.timeout.connect(self._play_next_frame)

        self.setWindowTitle("Video Browser Prototype")
        self.resize(1000, 800)  # Set initial size of the window

        # Create layout that will hold widgets that make up the browser
        layout = QVBoxLayout(self)

        # Create widgets for displaying video frames and navigation controls

        # Create an widget for displaying video frames
        self.im_view = pg.ImageView()
        layout.addWidget(self.im_view)

        # Create a button for navigating to the next frame
        self.button = QPushButton("Next Frame")
        self.button.clicked.connect(self.display_next_frame)
        layout.addWidget(self.button)

        # Create a button for navigating to the previous frame
        self.prev_button = QPushButton("Previous Frame")
        self.prev_button.clicked.connect(self.display_previous_frame)
        layout.addWidget(self.prev_button)

        # Create a slider for navigating to a specific frame
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.video.frame_count - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.valueChanged.connect(self.display_frame_at)
        layout.addWidget(self.frame_slider)

        # Create a label to display the current frame index
        self.frame_label = QLabel()
        layout.addWidget(self.frame_label)

        # Add play button to start/stop video playback
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        layout.addWidget(self.play_pause_button)

        if show_sync_status:
            self.sync_status_label = QLabel()
            layout.addWidget(self.sync_status_label)
        else:
            self.sync_status_label = None

        # Set up initial state

        first_frame = self.video.get_frame_at(0)
        if first_frame is None:
            raise ValueError("Could not read the first frame of the video.")
        self.im_view.setImage(first_frame)

        self.frame_label.setText(f"Current Frame: 1/{self.video.frame_count}")
        self._update_play_button_enabled()

    @Slot(int)
    def display_frame_at(self, frame_idx: int) -> bool:
        """Display the frame at the specified index.

        Parameters
        ----------
        frame_idx : int
            The index of the frame to display.

        Returns
        -------
        bool
            True if the frame was displayed, False if the index is out of bounds.
        """
        if frame_idx < 0 or frame_idx >= self.video.frame_count:
            logger.debug(
                f"Skipping updating to frame {frame_idx}, index is out of bounds "
                f"(0 to {self.video.frame_count - 1})."
            )
            return False

        self.current_frame_idx = frame_idx
        frame = self.video.get_frame_at(frame_idx)
        if frame is None:
            raise ValueError(
                f"Could not retrieve frame at index {frame_idx} event even though the "
                f"index is within bounds (0 to {self.video.frame_count - 1})."
            )

        self.im_view.setImage(frame)
        self._update_frame_label()
        self._update_slider_internal()
        self._update_play_button_enabled()

        # Emit signal that the frame has changed
        self.sigFrameChanged.emit(self.current_frame_idx)

        return True

    @Slot()
    def display_next_frame(self) -> bool:
        """Display the next frame in the video.

        Returns
        -------
        bool
            True if the next frame was displayed, False if next frame could not be
            retrieved (end of video?)
        """
        return self.display_frame_at(self.current_frame_idx + 1)

    @Slot()
    def display_previous_frame(self) -> bool:
        """Display the previous frame in the video.

        Returns
        -------
        bool
            True if the previous frame was displayed, False if previous frame could not
            be retrieved (beginning of video?)
        """
        return self.display_frame_at(self.current_frame_idx - 1)

    def _update_frame_label(self) -> None:
        """Update the frame label to show the current frame number."""
        # Use one-based index for display
        self.frame_label.setText(
            f"Current Frame: {self.current_frame_idx + 1}/{self.video.frame_count}"
        )

    def _update_slider_internal(self) -> None:
        """Update the slider to reflect the current frame index.

        This is a helper method to update the slider value without
        triggering the valueChanged signal of the slider.
        """
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame_idx)
        self.frame_slider.blockSignals(False)

    @Slot(SyncStatus)
    def set_sync_status(self, status: SyncStatus) -> None:
        """Set the sync status label and color."""
        if not self.show_sync_status or self.sync_status_label is None:
            return
        if status == SyncStatus.SYNCHRONIZED:
            self.sync_status_label.setText("Synchronized")
            self.sync_status_label.setStyleSheet("color: green; font-weight: bold;")
        elif status == SyncStatus.NO_RAW_DATA:
            self.sync_status_label.setText("No raw data for this frame")
            self.sync_status_label.setStyleSheet("color: red; font-weight: bold;")
        elif status == SyncStatus.NO_VIDEO_DATA:
            self.sync_status_label.setText("No video for this raw data")
            self.sync_status_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            raise ValueError(f"Unknown sync status: {status}")

    @Slot()
    def play_video(self) -> None:
        """Play the video frame by frame with its original fps."""
        if self.is_playing:
            logger.debug(
                "Received signal to play video even though video should be "
                "already playing. Skipping action."
            )
            return
        logger.debug("Playing video.")
        self.is_playing = True
        # Start the timer that controls automatic frame updates
        self.play_timer.start()
        self.play_pause_button.setText("Pause")  # Change play button to pause button

    @Slot()
    def pause_video(self) -> None:
        """Pause video playing and stop at current frame."""
        if not self.is_playing:
            logger.debug(
                "Received signal to pause video even though video should not "
                "be playing. Skipping action."
            )
        logger.debug("Pausing video.")
        self.is_playing = False
        self.play_timer.stop()
        self.play_pause_button.setText("Play")

    @Slot()
    def toggle_play_pause(self) -> None:
        """Either play or pause the video based on the current state."""
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()

    @Slot()
    def _play_next_frame(self) -> None:
        """Play next frame when play timer timeouts."""
        success = self.display_next_frame()
        if not success:
            # Pause the video if we are in the end
            self.pause_video()

    def _update_play_button_enabled(self) -> None:
        """Enable play button unless at the last frame."""
        if self.current_frame_idx >= self.video.frame_count - 1:
            self.play_pause_button.setEnabled(False)
        else:
            self.play_pause_button.setEnabled(True)
