"""Contains VideoBrowser Qt widget for displaying video."""

import collections
import logging
import os
import time
from typing import Literal

import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer, Signal, Slot  # type: ignore
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import videomeg_browser.browsers.gui_utils as gui_utils

from ..media.video import VideoFile
from .syncable_browser import SyncableBrowserWidget, SyncStatus

logger = logging.getLogger(__name__)

pg.setConfigOptions(imageAxisOrder="row-major")


class VideoBrowser(SyncableBrowserWidget):
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

    def __init__(
        self,
        videos: list[VideoFile],
        show_sync_status: bool = False,
        video_splitter_orientation: Literal["horizontal", "vertical"] = "horizontal",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._videos = videos
        self._show_sync_status = show_sync_status

        self._multiple_videos = len(videos) > 1
        # To which video the navigation controls currently apply
        self._selected_video_idx = 0
        self._selected_video = videos[self._selected_video_idx]
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
                parent=self,
            )
            for video in videos
        ]
        if self._multiple_videos:
            self._add_multi_video_view(video_splitter_orientation)
        else:
            self._layout.addWidget(self._video_views[0])

        # Add a horizontal layout that has time label and a frame slider.
        slider_layout = QHBoxLayout()
        self._layout.addLayout(slider_layout)

        # Label that shows the current time of the selected video.
        self._time_label = gui_utils.ElapsedTimeLabel(
            current_time_seconds=0.0,
            max_time_seconds=self._selected_video.duration,
            parent=self,
        )
        slider_layout.addWidget(self._time_label)

        # Slider for navigating to a specific frame
        self._frame_slider = gui_utils.IndexSlider(
            min_value=0,
            max_value=self._selected_video.frame_count - 1,
            value=0,
            parent=self,
        )
        self._frame_slider.sigIndexChanged.connect(
            self.display_frame_for_selected_video
        )
        slider_layout.addWidget(self._frame_slider)

        # Navigation bar with buttons: previous frame, play/pause, next frame
        # and possibly a video selector if multiple videos are shown.
        navigation_layout = QHBoxLayout()
        self._layout.addLayout(navigation_layout)

        self._navigation_bar = gui_utils.NavigationBar(
            prev_button_text="Previous Frame",
            next_button_text="Next Frame",
            parent=self,
        )
        self._navigation_bar.sigPreviousClicked.connect(
            self.display_previous_frame_for_selected_video
        )
        self._navigation_bar.sigNextClicked.connect(
            self.display_next_frame_for_selected_video
        )
        self._navigation_bar.sigPlayPauseClicked.connect(self._toggle_play_pause)
        navigation_layout.addWidget(self._navigation_bar)

        # Add drop-down menu for selecting which video to control.
        if self._multiple_videos:
            self._video_selector = QComboBox()
            self._video_selector.addItems(
                [os.path.basename(video.fname) for video in self._videos]
            )
            self._video_selector.setCurrentIndex(self._selected_video_idx)
            self._video_selector.currentIndexChanged.connect(self._set_selected_video)
            navigation_layout.addStretch()  # Push the selector to the right
            navigation_layout.addWidget(self._video_selector)

        # Label to display the current frame rate (FPS)
        self._fps_label = QLabel()
        self._fps_label.setText("Playing FPS: -")
        self._layout.addWidget(self._fps_label)

        self._update_buttons_enabled()

    def get_current_position(self, media_idx: int) -> int:
        """Return the current position index of the specified video."""
        return self._video_views[media_idx].current_frame_idx

    @property
    def is_playing(self) -> bool:
        """Return whether the video is currently playing."""
        return self._is_playing

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
        return self.set_position(frame_idx, self._selected_video_idx)

    def set_position(
        self, position_idx: int, media_idx: int, signal: bool = True
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
        frame_shown = self._video_views[media_idx].display_frame_at(position_idx)
        if not frame_shown:
            logger.debug(
                f"Could not display frame at index {position_idx} for video "
                f"{media_idx}."
            )
            return False

        self._frame_slider.set_value(
            self._get_current_frame_index_of_selected_video(), signal=False
        )
        self._update_time_label()
        self._update_buttons_enabled()

        if signal:
            self.sigPositionChanged.emit(media_idx, position_idx)

        return True

    def jump_to_end(self, media_idx: int, signal: bool = True) -> None:
        """Display the last frame of the specified video.

        Parameters
        ----------
        media_idx : int
            Index of the video to jump to the end.
        signal : bool, optional
            Whether to emit sigPositionChanged signal, by default True.
        """
        last_frame_idx = self._videos[media_idx].frame_count - 1
        self.set_position(last_frame_idx, media_idx, signal=signal)

    def jump_to_start(self, media_idx: int, signal: bool = True) -> None:
        """Display the first frame of the specified video.

        Parameters
        ----------
        media_idx : int
            Index of the video to jump to the start.
        signal : bool, optional
            Whether to emit sigPositionChanged signal, by default True.
        """
        self.set_position(0, media_idx, signal=signal)

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

    # Overrides the empty implementation of parent class
    def set_sync_status(self, status: SyncStatus, media_idx: int) -> None:
        """Set the sync status for a specific video view.

        Parameters
        ----------
        status : SyncStatus
            The synchronization status to set.
        media_idx : int
            Index of the video view to update.
        """
        self._video_views[media_idx].set_sync_status(status)

    def start_playback(self, media_idx: int) -> None:
        """Start playing the specified video.

        Parameters
        ----------
        media_idx : int
            Index of the video to start playing.
        """
        # Make the specified video view the selected one (corresponds to user changing
        # the selected video).
        self._set_selected_video(media_idx)
        # Start playing the video.
        self._play_video()

    def pause_playback(self) -> None:
        """Pause playback of the currently playing video."""
        self._pause_video()

    def _play_video(self) -> None:
        """Play the selected video with its original frame rate."""
        if self._is_playing:
            logger.warning(
                "Received signal to play video even though video should be "
                "already playing. Skipping action."
            )
            return
        logger.debug("Playing video.")
        self._is_playing = True
        self._navigation_bar.set_playing()
        # Start the timer that controls automatic frame updates
        self._play_timer.start()

        self.sigPlaybackStateChanged.emit(self._selected_video_idx, True)

    def _pause_video(self) -> None:
        """Pause video playing and stop at current frame."""
        if not self._is_playing:
            logger.warning(
                "Received signal to pause video even though video should not "
                "be playing. Skipping action."
            )
            return
        logger.debug("Pausing video.")
        self._is_playing = False
        self._play_timer.stop()
        self._navigation_bar.set_paused()
        self._fps_label.setText("Playing FPS: -")
        # Reset the frame tracker to start fresh with the next play.
        self._frame_rate_tracker.reset()

        self.sigPlaybackStateChanged.emit(self._selected_video_idx, False)

    @Slot()
    def _toggle_play_pause(self) -> None:
        """Either play or pause the video based on the current state."""
        if self._is_playing:
            self._pause_video()
        else:
            self._play_video()

    @Slot()
    def _play_next_frame(self) -> None:
        """Play next frame of currently selected video when play timer timeouts."""
        frame_shown = self.display_next_frame_for_selected_video()
        if frame_shown:
            self._update_frame_rate()
        else:
            # Pause the video if we are in the end
            self._pause_video()

    def _update_frame_rate(self) -> None:
        """Update frame rate state and possibly also displayed fps."""
        # Tell frame rate tracker that a new frame was displayed.
        self._frame_rate_tracker.notify_new_frame()
        self._n_frames_since_last_fps_update += 1
        if self._n_frames_since_last_fps_update >= self._n_frames_between_fps_updates:
            # Update the displayed frame rate.
            self._fps_label.setText(
                "Playing FPS: "
                f"{round(self._frame_rate_tracker.get_current_frame_rate())}"
            )
            self._n_frames_since_last_fps_update = 0

    def _update_buttons_enabled(self) -> None:
        """Enable/disable navigation buttons based on the frame of selected video."""
        current_frame_idx = self._get_current_frame_index_of_selected_video()
        max_frame_idx = self._selected_video.frame_count - 1

        self._navigation_bar.set_prev_enabled(current_frame_idx > 0)
        self._navigation_bar.set_next_enabled(current_frame_idx < max_frame_idx)
        self._navigation_bar.set_play_pause_enabled(current_frame_idx < max_frame_idx)

    def _get_current_frame_index_of_selected_video(self) -> int:
        """Get the current index for the currently selected video."""
        return self._video_views[self._selected_video_idx].current_frame_idx

    @Slot(int)
    def _set_selected_video(self, new_index: int) -> None:
        """Handle user changing the selected video."""
        self._selected_video_idx = new_index
        self._selected_video = self._videos[new_index]

        self._frame_slider.set_max_value(
            self._selected_video.frame_count - 1, signal=False
        )
        self._frame_slider.set_value(
            self._get_current_frame_index_of_selected_video(), signal=False
        )
        self._update_time_label(new_max=self._selected_video.duration)
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

    def _update_time_label(self, new_max: float | None = None) -> None:
        """Update the time label to show the current time of the selected video.

        Optionally set a new maximum time for the label.
        """
        video_frame_idx = self._get_current_frame_index_of_selected_video()
        time_seconds = video_frame_idx / self._selected_video.fps
        self._time_label.set_current_time(time_seconds)
        if new_max is not None:
            self._time_label.set_max_time(new_max)


class VideoView(QWidget):
    """A widget for displaying video.

    Includes labels for current frame index and optionally for synchronization status.

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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._video = video
        self._current_frame_idx = 0

        self._layout = QVBoxLayout(self)

        # Add graphics view...
        graphics_widget = pg.GraphicsView(parent=self)
        self._layout.addWidget(graphics_widget)
        # that has viewbox...
        self._view_box = pg.ViewBox(lockAspect=True, invertY=True)
        graphics_widget.setCentralWidget(self._view_box)
        # that holds image item.
        self._image_view = pg.ImageItem()
        self._view_box.addItem(self._image_view)

        # Add a horizontal layout for extras like frame index label and center button.
        extras_layout = QHBoxLayout()
        self._layout.addLayout(extras_layout)

        # Add name of the video file as a label.
        video_name = os.path.basename(self._video.fname)
        video_label = QLabel(f"{video_name}")
        video_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        extras_layout.addWidget(video_label)

        # Add info icon that shows video stats when hovered over.
        info_icon = QLabel()
        info_pixmap = gui_utils.load_icon_pixmap("info.png")
        if info_pixmap is not None:
            info_icon.setPixmap(
                info_pixmap.scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            logger.warning("Info icon not found, using text-based icon")
            info_icon.setText("ℹ️")
        info_icon.setToolTip(
            f"File: {video.fname}\n"
            f"Duration: {video.duration:.2f} seconds\n"
            f"Frame count: {video.frame_count}\n"
            f"Resolution: {video.frame_width}x{video.frame_height}\n"
            f"FPS: {video.fps:.2f}"
        )
        extras_layout.addWidget(info_icon)
        # Add the same hover info to the video label.
        video_label.setToolTip(info_icon.toolTip())

        # Button to center the video view
        self._center_button = QPushButton("Center Video")
        self._center_button.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        self._center_button.clicked.connect(self.center_video)
        extras_layout.addWidget(self._center_button)

        extras_layout.addStretch()

        # Label to display the current frame index
        self._frame_label = QLabel()
        extras_layout.addWidget(self._frame_label)

        if show_sync_status:
            self._sync_status_label = QLabel()
            extras_layout.addWidget(self._sync_status_label)
        else:
            self._sync_status_label = None

        # Make sure that we can display the first frame of the video.
        first_frame = self._video.get_frame_at(0)
        if first_frame is None:
            raise ValueError("Could not read the first frame of the video.")
        # Display the first frame.
        self.display_frame_at(0)

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
            logger.info(f"Could not retrieve frame at index {frame_idx}. ")
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
            self._sync_status_label.setText("Synchronized")
            self._sync_status_label.setStyleSheet("color: green; font-weight: bold;")
        elif status == SyncStatus.NO_DATA_THERE:
            self._sync_status_label.setText("No primary data for this frame")
            self._sync_status_label.setStyleSheet("color: red; font-weight: bold;")
        elif status == SyncStatus.NO_DATA_HERE:
            self._sync_status_label.setText("No video frame for primary data.")
            self._sync_status_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            raise ValueError(f"Unknown sync status: {status}")

    def center_video(self) -> None:
        """Scale and pan the view around video such that the image fills the view."""
        self._view_box.autoRange()

    @property
    def current_frame_idx(self) -> int:
        """Get the index of the currently displayed frame."""
        return self._current_frame_idx

    def _update_frame_label(self) -> None:
        """Update the frame label to show the current frame number."""
        # Use one-based index for display
        self._frame_label.setText(
            f"Frame {self._current_frame_idx + 1}/{self._video.frame_count}"
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
