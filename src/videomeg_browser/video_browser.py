"""Contains VideoBrowser Qt widget for displaying video."""

import collections
import logging
import time
from enum import Enum, auto

import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer, Signal, Slot  # type: ignore
from qtpy.QtWidgets import (
    QHBoxLayout,
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
        self._video = video
        self._show_sync_status = show_sync_status

        self._current_frame_idx = 0
        self._is_playing = False  # Whether the frame updates are currently automatic

        # Set up timer that allow automatic frame updates (playing the video)
        self._play_timer = QTimer(parent=self)
        # Milliseconds between frame updates so that video is played with original fps
        self._play_timer_interval_ms = round(1000 / video.fps)
        self._play_timer.setInterval(self._play_timer_interval_ms)
        self._play_timer.timeout.connect(self._play_next_frame)

        # Instantiate frame tracker for monitoring video fps when playing.
        self._frame_rate_tracker = FrameRateTracker(
            max_intervals_to_average=2 * round(video.fps)  # average over two seconds
        )
        # Update the displayed frame rate once per second.
        self.n_frames_between_fps_updates = round(video.fps)
        self.n_frames_since_last_fps_update = 0

        self.setWindowTitle(self._video.fname)
        self.resize(1000, 800)  # Set initial size of the window

        # Create layout that will hold widgets that make up the browser
        layout = QVBoxLayout(self)

        # Create widgets for displaying video frames and navigation controls

        # Widget for displaying video frames
        self._image_view = pg.ImageView(parent=self)
        layout.addWidget(self._image_view)

        # Label to display the current frame index
        self._frame_label = QLabel()
        layout.addWidget(self._frame_label)

        # Slider for navigating to a specific frame
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(self._video.frame_count - 1)
        self._frame_slider.setValue(0)
        self._frame_slider.valueChanged.connect(self.display_frame_at)
        layout.addWidget(self._frame_slider)

        # Navigation bar with buttons: previous frame, play/pause, next frame
        navigation_layout = QHBoxLayout()

        self._prev_button = QPushButton("Previous Frame")
        self._prev_button.clicked.connect(self.display_previous_frame)
        navigation_layout.addWidget(self._prev_button)

        self._play_pause_button = QPushButton("Play")
        self._play_pause_button.clicked.connect(self.toggle_play_pause)
        navigation_layout.addWidget(self._play_pause_button)

        self._button = QPushButton("Next Frame")
        self._button.clicked.connect(self.display_next_frame)
        navigation_layout.addWidget(self._button)

        layout.addLayout(navigation_layout)

        self._fps_label = QLabel()
        self._fps_label.setText("FPS: -")
        layout.addWidget(self._fps_label)

        if show_sync_status:
            self._sync_status_label = QLabel()
            layout.addWidget(self._sync_status_label)
        else:
            self._sync_status_label = None

        # Set up initial state

        first_frame = self._video.get_frame_at(0)
        if first_frame is None:
            raise ValueError("Could not read the first frame of the video.")
        self._image_view.setImage(first_frame)

        self._frame_label.setText(f"Current Frame: 1/{self._video.frame_count}")
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
        frame = self._video.get_frame_at(frame_idx)
        if frame is None:
            logger.info(
                f"Could not retrieve frame at index {frame_idx}. "
                "Skipping updating the frame."
            )
            return False

        self._current_frame_idx = frame_idx
        self._image_view.setImage(frame)
        self._update_frame_label()
        self._update_slider_internal()
        self._update_play_button_enabled()

        # Emit signal that the frame has changed
        self.sigFrameChanged.emit(self._current_frame_idx)

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
        return self.display_frame_at(self._current_frame_idx + 1)

    @Slot()
    def display_previous_frame(self) -> bool:
        """Display the previous frame in the video.

        Returns
        -------
        bool
            True if the previous frame was displayed, False if previous frame could not
            be retrieved (beginning of video?)
        """
        return self.display_frame_at(self._current_frame_idx - 1)

    @Slot(SyncStatus)
    def set_sync_status(self, status: SyncStatus) -> None:
        """Set the sync status label and color."""
        if not self._show_sync_status or self._sync_status_label is None:
            return
        if status == SyncStatus.SYNCHRONIZED:
            self._sync_status_label.setText("Synchronized")
            self._sync_status_label.setStyleSheet("color: green; font-weight: bold;")
        elif status == SyncStatus.NO_RAW_DATA:
            self._sync_status_label.setText("No raw data for this frame")
            self._sync_status_label.setStyleSheet("color: red; font-weight: bold;")
        elif status == SyncStatus.NO_VIDEO_DATA:
            self._sync_status_label.setText("No video for this raw data")
            self._sync_status_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            raise ValueError(f"Unknown sync status: {status}")

    @Slot()
    def play_video(self) -> None:
        """Play the video frame by frame with its original fps."""
        if self._is_playing:
            logger.warning(
                "Received signal to play video even though video should be "
                "already playing. Skipping action."
            )
            return
        logger.debug("Playing video.")
        self._is_playing = True
        # Start the timer that controls automatic frame updates
        self._play_timer.start()
        self._play_pause_button.setText("Pause")  # Change play button to pause button

    @Slot()
    def pause_video(self) -> None:
        """Pause video playing and stop at current frame."""
        if not self._is_playing:
            logger.warning(
                "Received signal to pause video even though video should not "
                "be playing. Skipping action."
            )
        logger.debug("Pausing video.")
        self._is_playing = False
        self._play_timer.stop()
        self._play_pause_button.setText("Play")
        self._fps_label.setText("FPS: -")
        # Reset the frame tracker to start fresh with the next play.
        self._frame_rate_tracker.reset()

    @Slot()
    def toggle_play_pause(self) -> None:
        """Either play or pause the video based on the current state."""
        if self._is_playing:
            self.pause_video()
        else:
            self.play_video()

    @Slot()
    def _play_next_frame(self) -> None:
        """Play next frame when play timer timeouts."""
        success = self.display_next_frame()
        if success:
            self._update_frame_rate()
        else:
            # Pause the video if we are in the end
            self.pause_video()

    def _update_frame_rate(self) -> None:
        """Update frame rate state and possibly also displayed fps."""
        # Tell frame rate tracker that a new frame was displayed.
        self._frame_rate_tracker.notify_new_frame()
        self.n_frames_since_last_fps_update += 1
        if self.n_frames_since_last_fps_update > self.n_frames_between_fps_updates:
            # Update the displayed frame rate.
            self._fps_label.setText(
                f"FPS: {round(self._frame_rate_tracker.get_current_frame_rate())}"
            )
            self.n_frames_since_last_fps_update = 0

    def _update_play_button_enabled(self) -> None:
        """Enable play button unless at the last frame."""
        if self._current_frame_idx >= self._video.frame_count - 1:
            self._play_pause_button.setEnabled(False)
        else:
            self._play_pause_button.setEnabled(True)

    def _update_frame_label(self) -> None:
        """Update the frame label to show the current frame number."""
        # Use one-based index for display
        self._frame_label.setText(
            f"Current Frame: {self._current_frame_idx + 1}/{self._video.frame_count}"
        )

    def _update_slider_internal(self) -> None:
        """Update the slider to reflect the current frame index.

        This is a helper method to update the slider value without
        triggering the valueChanged signal of the slider.
        """
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(self._current_frame_idx)
        self._frame_slider.blockSignals(False)


class FrameRateTracker:
    """Tracks the frame rate (FPS) of playing video.

    Parameters
    ----------
    max_intervals_to_average: int
        The maximum number of frame intervals to average when estimating FPS.
    """

    def __init__(self, max_intervals_to_average: int) -> None:
        if max_intervals_to_average < 1:
            raise ValueError("Interval count must be a positive integer.")
        # When the tracker was notified of the last frame
        self._last_frame_time: float | None = None
        # Queue that holds most recent frame intervals
        self._frame_intervals: collections.deque[float] = collections.deque(
            maxlen=max_intervals_to_average
        )

    def notify_new_frame(self) -> None:
        """Notify the tracker that a new frame was displayed."""
        now = time.perf_counter()
        if self._last_frame_time is not None:
            # Calculate and store the interval between last frame and this frame.
            interval = now - self._last_frame_time
            self._frame_intervals.append(interval)

        self._last_frame_time = now

    def get_current_frame_rate(self) -> float:
        """Return the current frame rate estimated with average frame interval.

        Returns
        -------
        float
            The current frame rate (FPS). Will be zero if `notify_new_frame` has
            been called less than two times.
        """
        if not self._frame_intervals:
            logger.debug(
                "No frame intervals to use for current frame rate estimation. "
                "Returning zero."
            )
            return 0.0
        average_interval = sum(self._frame_intervals) / len(self._frame_intervals)
        if average_interval == 0:
            logger.warning(
                "Average frame interval is zero. Cannot estimate FPS. Returning zero."
            )
            return 0.0

        return 1.0 / average_interval

    def reset(self) -> None:
        """Forget the past frame intervals.

        Use this to start the tracking fresh with next call to `notify_new_frame`.
        """
        self._frame_intervals.clear()
        self._last_frame_time = None
