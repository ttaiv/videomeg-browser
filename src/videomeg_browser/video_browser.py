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
from qtpy.QtCore import QTimer, Slot
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
    """

    def __init__(self, video: VideoFile, show_sync_status: bool = False):
        super().__init__()
        self.video = video
        self.current_frame_idx = 0

        self.setWindowTitle("Video Browser Prototype")

        layout = QVBoxLayout(self)

        self._has_sync_status_label = show_sync_status
        if show_sync_status:
            self.sync_status_label = QLabel("Synchronized")
            self.sync_status_label.setStyleSheet("color: green; font-weight: bold;")
            layout.addWidget(self.sync_status_label)
        else:
            self.sync_status_label = None

        # Create an ImageView widget and display first frame of the video
        self.im_view = pg.ImageView()
        layout.addWidget(self.im_view)
        first_frame = self.video.get_frame_at(0)
        if first_frame is None:
            raise ValueError("Could not read the first frame of the video.")
        self.im_view.setImage(first_frame)

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
        self.frame_slider.valueChanged.connect(self.slider_frame_changed)
        layout.addWidget(self.frame_slider)

        # Create a label to display the current frame index
        self.frame_label = QLabel()
        self.frame_label.setText(f"Current Frame: 1/{self.video.frame_count}")
        layout.addWidget(self.frame_label)

        # Initialize attributes for video playing
        self.is_playing = False
        self.play_timer = QTimer(parent=self)
        # How many milliseconds between frame updates so that
        # video is played with original fps
        self.play_timer_interval_ms = int(1000 / video.fps)

        self.play_timer.setInterval(self.play_timer_interval_ms)
        self.play_timer.timeout.connect(self._play_next_frame)

        # Add play button to start/stop video playback
        self.play_pause_button = QPushButton("Play")
        self.play_pause_button.clicked.connect(self.toggle_play_pause)
        layout.addWidget(self.play_pause_button)

        self.update_play_button_enabled()

    @Slot()
    def display_next_frame(self) -> bool:
        """Display the next frame in the video.

        Returns
        -------
        bool
            True if the next frame was displayed, False if next frame could not be
            retrieved (end of video?)
        """
        frame = self.video.get_frame_at(self.current_frame_idx + 1)
        if frame is None:
            logger.debug(
                "Skipping updating to the next frame, already at the last frame."
            )
            return False

        self.current_frame_idx += 1
        self.im_view.setImage(frame)
        self.update_frame_label()
        self.update_slider()
        self.update_play_button_enabled()

        return True

    @Slot()
    def display_previous_frame(self):
        """Display the previous frame in the video."""
        frame = self.video.get_frame_at(self.current_frame_idx - 1)
        if frame is None:
            logger.debug(
                "Skipping updating to the previous frame, already at the first frame."
            )
            return

        self.current_frame_idx -= 1
        self.im_view.setImage(frame)
        self.update_frame_label()
        self.update_slider()
        self.update_play_button_enabled()

    @Slot(int)
    def slider_frame_changed(self, value: int):
        """Update view to display the frame corresponding to the slider's position."""
        self.current_frame_idx = value
        frame = self.video.get_frame_at(self.current_frame_idx)
        if frame is None:
            raise ValueError(f"Invalid frame index {value} selected with the slider.")

        self.im_view.setImage(frame)
        self.update_frame_label()
        self.update_play_button_enabled()

    @Slot()
    def update_frame_label(self):
        """Update the frame label to show the current frame number."""
        # Use one-based index for display
        self.frame_label.setText(
            f"Current Frame: {self.current_frame_idx + 1}/{self.video.frame_count}"
        )

    @Slot()
    def update_slider(self):
        """Update the slider to reflect the current frame index."""
        # Use one-based index for display
        self.frame_slider.setValue(self.current_frame_idx)

    @Slot(SyncStatus)
    def set_sync_status(self, status: SyncStatus):
        """Set the sync status label and color."""
        if not self._has_sync_status_label or self.sync_status_label is None:
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
    def play_video(self):
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
    def pause_video(self):
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
    def toggle_play_pause(self):
        """Either play or pause the video based on the current state."""
        if self.is_playing:
            self.pause_video()
        else:
            self.play_video()

    @Slot()
    def _play_next_frame(self):
        """Play next frame when play timer timeouts."""
        success = self.display_next_frame()
        if not success:
            # Pause the video if we are in the end
            self.pause_video()

    def update_play_button_enabled(self):
        """Enable play button unless at the last frame."""
        if self.current_frame_idx >= self.video.frame_count - 1:
            self.play_pause_button.setEnabled(False)
        else:
            self.play_pause_button.setEnabled(True)
