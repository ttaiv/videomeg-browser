"""Contains VideoBrowser Qt widget for displaying video."""

import collections
import logging
import os
import time
from enum import Enum, auto
from typing import Literal

import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer, Signal, Slot  # type: ignore
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSplitter,
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
    """A browser for viewing video frames from one or more video files.

    Parameters
    ----------
    videos : list[VideoFile]
        The video file(s) to be displayed.
    show_sync_status : bool, optional
        Whether to show a label indicating the synchronization status of each video,
        by default False.
    display_method : Literal["image_view", "image_item"], optional
        The method used to display the video frames. If "image_view", uses
        `pyqtgraph.ImageView` with histogram and extra controls. If "image_item", uses
        plain 'pyqtgraph.ImageItem' inside a `pyqtgraph.ViewBox`.
    video_splitter_orientation : Literal["horizontal", "vertical"], optional
        The orientation of the video splitter that separates multiple video views,
        by default "horizontal". Has no effect if only one video is provided.
    parent : QWidget, optional
        The parent widget for this browser, by default None
    """

    # Emits a signal when the displayed frame of any shown video changes.
    sigFrameChanged = Signal(int, int)  # video index, frame index

    def __init__(
        self,
        videos: list[VideoFile],
        show_sync_status: bool = False,
        display_method: Literal["image_view", "image_item"] = "image_view",
        video_splitter_orientation: Literal["horizontal", "vertical"] = "horizontal",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._videos = videos
        self._show_sync_status = show_sync_status

        self._multiple_videos = len(videos) > 1
        # To which video the navigation controls currently apply
        self._selected_video_idx = 0
        self._is_playing = False  # Whether the frame updates are currently automatic

        # Set up timer that allow automatic frame updates (playing the video)
        self._play_timer = QTimer(parent=self)
        self._set_play_timer_interval()
        self._play_timer.timeout.connect(self._play_next_frame)

        # Instantiate frame tracker for monitoring video fps when playing.
        self._frame_rate_tracker = FrameRateTracker(max_intervals_to_average=30)
        # Define helper variables for tracking when to update the displayed fps.
        self._n_frames_between_fps_updates = 30
        self._n_frames_since_last_fps_update = 0

        self.setWindowTitle("Video Browser")

        # Create the main layout for the browser.
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # Add video view(s).
        self._video_views = [
            VideoView(
                video,
                show_sync_status=show_sync_status,
                display_method=display_method,
                parent=self,
            )
            for video in videos
        ]
        if self._multiple_videos:
            self._add_multi_video_view(video_splitter_orientation)
        else:
            self._layout.addWidget(self._video_views[0])

        # Slider for navigating to a specific frame
        self._frame_slider = QSlider(Qt.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(
            self._videos[self._selected_video_idx].frame_count - 1
        )
        self._frame_slider.setValue(0)
        self._frame_slider.valueChanged.connect(self.display_frame_for_selected_video)
        self._layout.addWidget(self._frame_slider)

        # Navigation bar with buttons: previous frame, play/pause, next frame
        # and possibly a video selector if multiple videos are shown.
        navigation_layout = QHBoxLayout()
        self._layout.addLayout(navigation_layout)

        play_prev_pause_width = 150  # Fixed width for play/pause buttons

        self._prev_button = QPushButton("Previous Frame")
        self._prev_button.clicked.connect(
            self.display_previous_frame_for_selected_video
        )
        self._prev_button.setMinimumWidth(play_prev_pause_width)
        navigation_layout.addWidget(self._prev_button)

        self._play_pause_button = QPushButton("Play")
        self._play_pause_button.clicked.connect(self.toggle_play_pause)
        self._play_pause_button.setMinimumWidth(play_prev_pause_width)
        navigation_layout.addWidget(self._play_pause_button)

        self._next_button = QPushButton("Next Frame")
        self._next_button.clicked.connect(self.display_next_frame_for_selected_video)
        self._next_button.setMinimumWidth(play_prev_pause_width)
        navigation_layout.addWidget(self._next_button)

        # Add drop-down menu for selecting which video to control.
        if self._multiple_videos:
            self._video_selector = QComboBox()
            self._video_selector.addItems(
                [os.path.basename(video.fname) for video in self._videos]
            )
            self._video_selector.setCurrentIndex(self._selected_video_idx)
            self._video_selector.currentIndexChanged.connect(
                self._on_selected_video_change
            )
            navigation_layout.addStretch()  # Push the selector to the right
            navigation_layout.addWidget(self._video_selector)

        # Label to display the current frame rate (FPS)
        self._fps_label = QLabel()
        self._fps_label.setText("FPS: -")
        self._layout.addWidget(self._fps_label)

        self._update_buttons_enabled()

    @Slot(int)
    def display_frame_for_selected_video(self, frame_idx: int) -> bool:
        """Display the frame at the specified index for the selected video view.

        Parameters
        ----------
        frame_idx : int
            The index of the frame to display.

        Returns
        -------
        bool
            True if the frame was displayed, False if the index is out of bounds.
        """
        return self.display_frame_for_video_with_idx(
            frame_idx, self._selected_video_idx
        )

    @Slot(int)
    def display_frame_for_video_with_idx(
        self, frame_idx: int, video_idx: int, signal: bool = True
    ) -> bool:
        """Display the frame at the specified index for a specific video view.

        Parameters
        ----------
        frame_idx : int
            The index of the frame to display.
        video_idx : int
            The index of the video view to update.
        signal : bool, optional
            Whether to emit the frame changed signal, by default True.
            Setting this to False is useful when setting the view programmatically
            and you do not want to trigger any additional actions that might be
            connected to the signal.

        Returns
        -------
        bool
            True if the frame was displayed, False if the index is out of bounds.
        """
        frame_shown = self._video_views[video_idx].display_frame_at(frame_idx)
        if not frame_shown:
            logger.debug(
                f"Could not display frame at index {frame_idx} for video {video_idx}."
            )
            return False

        self._update_slider_internal()
        self._update_buttons_enabled()

        if signal:
            self.sigFrameChanged.emit(video_idx, frame_idx)

        return True

    @Slot()
    def display_next_frame_for_selected_video(self) -> bool:
        """Display the next frame for the currently selected video.

        Returns
        -------
        bool
            True if the next frame was displayed, False if next frame could not be
            retrieved (end of video?)
        """
        return self.display_frame_for_selected_video(
            self._get_current_frame_index_of_selected_video() + 1
        )

    @Slot()
    def display_previous_frame_for_selected_video(self) -> bool:
        """Display the previous frame for the currently selected video.

        Returns
        -------
        bool
            True if the previous frame was displayed, False if previous frame could not
            be retrieved (beginning of video?)
        """
        return self.display_frame_for_selected_video(
            self._get_current_frame_index_of_selected_video() - 1
        )

    @Slot()
    def play_video(self) -> None:
        """Play the selected video with its original frame rate."""
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

    def set_sync_status_for_video_with_idx(
        self, video_idx: int, status: SyncStatus
    ) -> None:
        """Set the sync status for a specific video view.

        Parameters
        ----------
        video_idx : int
            The index of the video view to update.
        status : SyncStatus
            The synchronization status to set.
        """
        self._video_views[video_idx].set_sync_status(status)

    @Slot()
    def _play_next_frame(self) -> None:
        """Play next frame of currently selected video when play timer timeouts."""
        frame_shown = self.display_next_frame_for_selected_video()
        if frame_shown:
            self._update_frame_rate()
        else:
            # Pause the video if we are in the end
            self.pause_video()

    def _update_frame_rate(self) -> None:
        """Update frame rate state and possibly also displayed fps."""
        # Tell frame rate tracker that a new frame was displayed.
        self._frame_rate_tracker.notify_new_frame()
        self._n_frames_since_last_fps_update += 1
        if self._n_frames_since_last_fps_update >= self._n_frames_between_fps_updates:
            # Update the displayed frame rate.
            self._fps_label.setText(
                f"FPS: {round(self._frame_rate_tracker.get_current_frame_rate())}"
            )
            self._n_frames_since_last_fps_update = 0

    def _update_buttons_enabled(self) -> None:
        """Enable/disable navigation buttons based on the frame of selected video."""
        current_frame_idx = self._get_current_frame_index_of_selected_video()
        max_frame_idx = self._videos[self._selected_video_idx].frame_count - 1

        self._prev_button.setEnabled(current_frame_idx > 0)
        self._next_button.setEnabled(current_frame_idx < max_frame_idx)
        self._play_pause_button.setEnabled(current_frame_idx < max_frame_idx)

    def _update_slider_internal(self, new_maximum: int | None = None) -> None:
        """Update the slider to the current frame index and optionally set new maximum.

        This is a helper method to update the slider value without
        triggering the valueChanged signal of the slider.
        """
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(self._get_current_frame_index_of_selected_video())
        if new_maximum is not None:
            self._frame_slider.setMaximum(new_maximum)
        self._frame_slider.blockSignals(False)

    def _get_current_frame_index_of_selected_video(self) -> int:
        """Get the current index for the currently selected video."""
        return self._video_views[self._selected_video_idx].current_frame_idx

    @Slot(int)
    def _on_selected_video_change(self, new_index: int) -> None:
        """Handle user changing the selected video."""
        self._selected_video_idx = new_index
        self._update_slider_internal(
            new_maximum=self._videos[new_index].frame_count - 1
        )
        self._update_buttons_enabled()
        self._set_play_timer_interval()

    def _set_play_timer_interval(self) -> None:
        """Set up the play timer interval based on currently selected video."""
        # Milliseconds between frame updates so that video is played with original fps
        self._play_timer_interval_ms = round(
            1000 / self._videos[self._selected_video_idx].fps
        )
        self._play_timer.setInterval(self._play_timer_interval_ms)

    def _add_multi_video_view(
        self, video_splitter_orientation: Literal["horizontal", "vertical"]
    ) -> None:
        """Add a splitter with video views to the layout."""
        if video_splitter_orientation == "horizontal":
            video_splitter = QSplitter(Qt.Horizontal, parent=self)
        elif video_splitter_orientation == "vertical":
            video_splitter = QSplitter(Qt.Vertical, parent=self)
        else:
            raise ValueError(
                f"Invalid video splitter orientation: {video_splitter_orientation}. "
                "Use 'horizontal' or 'vertical'."
            )
        for video_view in self._video_views:
            video_splitter.addWidget(video_view)
        self._layout.addWidget(video_splitter, stretch=1)


class VideoView(QWidget):
    """A widget for displaying video.

    Parameters
    ----------
    video : VideoFile
        The video file to be displayed.
    show_sync_status : bool, optional
        Whether to show a label indicating the synchronization status of the video,
        by default False.
    display_method : Literal["image_view", "image_item"], optional
        The method used to display the video frames. If "image_view", uses
        `pyqtgraph.ImageView` with histogram and extra controls. If "image_item",
        uses plain 'pyqtgraph.ImageItem' inside a `pyqtgraph.ViewBox`.
        By default "image_view".
    parent : QWidget, optional
        The parent widget for this view, by default None
    """

    # Emits a signal with the index of the new currently displayed frame
    # when the displayed frame changes.
    sigFrameChanged = Signal(int)

    def __init__(
        self,
        video: VideoFile,
        show_sync_status: bool = False,
        display_method: Literal["image_view", "image_item"] = "image_view",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._video = video
        self._current_frame_idx = 0

        self._layout = QVBoxLayout(self)

        if display_method == "image_view":
            self._image_view = pg.ImageView(parent=self)
            self._layout.addWidget(self._image_view)
        elif display_method == "image_item":
            # Manually create a GraphicsLayoutWidget with ViewBox and ImageItem.
            graphics_widget = pg.GraphicsLayoutWidget(parent=self)
            self._layout.addWidget(graphics_widget)
            view_box = graphics_widget.addViewBox()
            view_box.setAspectLocked(True)
            # Inverting Y makes the orientation the same as in ImageView.
            view_box.invertY(True)
            # The ImageItem has method `setImage` just like ImageView!
            self._image_view = pg.ImageItem()
            view_box.addItem(self._image_view)
        else:
            raise ValueError(
                f"Invalid display method: {display_method}. "
                "Use 'image_view' or 'image_item'."
            )

        # Add a horizontal layout for info labels
        info_layout = QHBoxLayout()
        self._layout.addLayout(info_layout)

        # Label to display the current frame index
        self._frame_label = QLabel()
        info_layout.addWidget(self._frame_label)

        if show_sync_status:
            self._sync_status_label = QLabel()
            info_layout.addStretch()  # Push the sync status label to the right
            info_layout.addWidget(self._sync_status_label)
        else:
            self._sync_status_label = None

        # Show the first frame of the video by default.
        first_frame = self._video.get_frame_at(0)
        if first_frame is None:
            raise ValueError("Could not read the first frame of the video.")
        self._image_view.setImage(first_frame)
        self._frame_label.setText(f"Current Frame: 1/{self._video.frame_count}")

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

        # Emit signal that the frame has changed
        self.sigFrameChanged.emit(self._current_frame_idx)

        return True

    def set_sync_status(self, status: SyncStatus) -> None:
        """Set the sync status label and color."""
        if self._sync_status_label is None:
            logger.warning(
                "No sync status label available. Skipping setting sync status."
            )
            return
        if status == SyncStatus.SYNCHRONIZED:
            self._sync_status_label.setText("Synchronized to raw")
            self._sync_status_label.setStyleSheet("color: green; font-weight: bold;")
        elif status == SyncStatus.NO_RAW_DATA:
            self._sync_status_label.setText("No raw data for this frame")
            self._sync_status_label.setStyleSheet("color: red; font-weight: bold;")
        elif status == SyncStatus.NO_VIDEO_DATA:
            self._sync_status_label.setText("No video for selected raw data")
            self._sync_status_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            raise ValueError(f"Unknown sync status: {status}")

    @property
    def current_frame_idx(self) -> int:
        """Get the index of the currently displayed frame."""
        return self._current_frame_idx

    def _update_frame_label(self) -> None:
        """Update the frame label to show the current frame number."""
        # Use one-based index for display
        self._frame_label.setText(
            f"Current Frame: {self._current_frame_idx + 1}/{self._video.frame_count}"
        )


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
