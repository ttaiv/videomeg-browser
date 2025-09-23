"""Contains AudioView and AudioBrowser Qt widgets for visualizing audio data."""

import logging
import os.path

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer, Signal, Slot  # type: ignore
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .. import audio_player
from ..media.audio import AudioFile
from . import gui_utils
from .syncable_browser import SyncableBrowserWidget
from .time_selector import TimeSelector

logger = logging.getLogger(__name__)


class AudioView(QWidget):
    """A widget for displaying audio waveform.

    Includes controls for channel selection and view options.

    Parameters
    ----------
    audio : AudioFile
        The audio file to be displayed.
    plotting_window_size : int | None, optional
        The size of the plotting window in samples. Plotting window size N means that
        N samples are aggregated into a single point in the plot. If None (default),
        the plotting window size is set to 1% of the audio sampling rate.
    default_view_len : float, optional
        The duration to show in the audio view with default zoom level,
        by default 10.0 seconds.
    time_selector_padding : float, optional
        Padding (in seconds) to apply when making sure that the user does not drag the
        selector too close to the edges of the view, by default 0.1 seconds.
    parent : QWidget | None, optional
        The parent widget for this view, by default None.
    """

    # Emits a signal with the sample index when the position changes
    sigSampleIndexChanged = Signal(int)  # sample index
    sigChannelSelectionChanged = Signal(object)  # channel index or None

    def __init__(
        self,
        audio: AudioFile,
        plotting_window_size: int | None = None,
        default_view_len: float = 10.0,
        time_selector_padding: float = 0.1,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent=parent)
        self._audio = audio
        if plotting_window_size is None:
            self._plotting_window_size = int(audio.sampling_rate / 100)
            logger.info(
                f"Using default plotting window size of {self._plotting_window_size} "
                f"for {audio.sampling_rate} Hz audio."
            )
        else:
            self._plotting_window_size = plotting_window_size

        self._time_selector_padding = time_selector_padding
        self._default_view_len = default_view_len
        # The index of the currently highlighted/selected sample
        self._current_sample = 0
        self._visible_duration_seconds = default_view_len  # this is changed by zooming
        self._channel_selection: int | None = None  # None shows mean of all channels

        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # Add the plot that visualizes the audio data
        self._setup_plot_widget()

        # Add a vertical line that indicates the current sample and is movable
        self._time_selector = TimeSelector(parent=self)
        self._time_selector.sigSelectedTimeChanged.connect(self._on_time_selector_moved)
        self._time_selector.add_to_plot(self._plot_widget)

        # Add controls and information about the audio.
        self._setup_toolbar()

        # Initial visualization
        self._plot_selected_channel()
        self._set_clamped_time_range(0, self._visible_duration_seconds)
        self.display_at_sample(0, signal=False)

    @property
    def current_sample(self) -> int:
        """Get the index of the current sample position."""
        return self._current_sample

    @property
    def current_time(self) -> float:
        """Get the current position in seconds."""
        return self._current_sample / self._audio.sampling_rate

    @property
    def channel_selection(self) -> int | None:
        """Get the currently selected channel index.

        Returns
        -------
        int | None
            The index of the selected channel, or None if mean of all the channels
            is shown.
        """
        return self._channel_selection

    def display_at_sample(self, sample_idx: int, signal: bool = True) -> bool:
        """Set the currently highlighted sample index and update the view if necessary.

        Parameters
        ----------
        sample_idx : int
            The sample index to highlight in the audio view.
        signal : bool, optional
            Whether to emit the position changed signal, by default True.

        Returns
        -------
        bool
            True if the visualization was updated, False if the index is out of bounds.
        """
        if sample_idx < 0 or sample_idx >= self._audio.n_samples:
            logger.warning(
                f"Cannot display audio at sample index {sample_idx} (out of bounds)"
            )
            return False

        self._current_sample = sample_idx
        # Current time gets updated based on the sample index.

        self._time_selector.set_selected_time_no_signal(self.current_time)
        self._move_view_to_current_time()
        self._time_label.set_current_time(self.current_time)

        if signal:
            self.sigSampleIndexChanged.emit(self._current_sample)

        return True

    def display_at_time(self, time_seconds: float, signal: bool = True) -> bool:
        """Set the currently highlighted time and update the view if necessary.

        This is just a wrapper around `display_at_sample` that converts time to
        sample index.

        Parameters
        ----------
        time_seconds : float
            The time in seconds to highlight in the audio view.
        signal : bool, optional
            Whether to emit the position changed signal, by default True.

        Returns
        -------
        bool
            True if the visualization was updated, False if the time is out of bounds.
        """
        sample_idx = int(time_seconds * self._audio.sampling_rate)
        return self.display_at_sample(sample_idx, signal=signal)

    def _setup_plot_widget(self) -> None:
        """Set up the plot widget for audio visualization."""
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.setLabel("bottom", "Time", "s")
        self._plot_widget.setLabel("left", "Amplitude")
        self._plot_widget.setMouseEnabled(x=True, y=False)
        self._layout.addWidget(self._plot_widget)

    def _setup_toolbar(self) -> None:
        """Set up toolbar that contains controls and information about the audio."""
        toolbar_layout = QHBoxLayout()
        self._layout.addLayout(toolbar_layout)

        # Add name of the audio file as a label
        audio_name = os.path.basename(self._audio.fname)
        audio_label = QLabel(f"{audio_name}")
        audio_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        toolbar_layout.addWidget(audio_label)

        # Add info label that shows audio stats when hovered over
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
            f"File: {self._audio.fname}\n"
            f"Sampling rate: {self._audio.sampling_rate} Hz\n"
            f"Channels: {self._audio.n_channels}\n"
            f"Bit depth: {self._audio.bit_depth} bits\n"
            f"Duration: {self._audio.duration:.2f} s\n"
            f"Samples: {self._audio.n_samples}"
        )
        toolbar_layout.addWidget(info_icon)
        # Add the same hover info to the audio label
        audio_label.setToolTip(info_icon.toolTip())

        # Add zoom controls
        zoom_label = QLabel("Zoom:")
        toolbar_layout.addWidget(zoom_label)

        self._zoom_in_button = QPushButton("+")
        self._zoom_in_button.clicked.connect(self._zoom_in)
        toolbar_layout.addWidget(self._zoom_in_button)

        self._zoom_out_button = QPushButton("-")
        self._zoom_out_button.clicked.connect(self._zoom_out)
        toolbar_layout.addWidget(self._zoom_out_button)

        self._zoom_reset_button = QPushButton("Reset")
        self._zoom_reset_button.clicked.connect(self._reset_zoom)
        toolbar_layout.addWidget(self._zoom_reset_button)

        toolbar_layout.addStretch()

        # Add channel selector
        channel_label = QLabel("Channel:")
        toolbar_layout.addWidget(channel_label)

        self._channel_selector = QComboBox()
        self._channel_selector.addItem("All (show mean)")
        for i in range(self._audio.n_channels):
            self._channel_selector.addItem(f"Channel {i + 1}")
        self._channel_selector.currentIndexChanged.connect(
            self._update_selected_channel
        )
        toolbar_layout.addWidget(self._channel_selector)

        # Add label that shows the current time and max time.
        self._time_label = gui_utils.ElapsedTimeLabel(
            current_time_seconds=self.current_time,
            max_time_seconds=self._audio.duration,
            parent=self,
        )
        toolbar_layout.addWidget(self._time_label)

    def _move_view_to_current_time(self) -> None:
        """Ensure that the view contains the currently highlighted/selected sample.

        If the current sample is outside the current view range, move the view
        as many window lengths as needed to bring the current sample into view.
        """
        current_time = self.current_time
        window_min, window_max = self._plot_widget.viewRange()[0]
        window_len = window_max - window_min

        if window_min <= current_time <= window_max:
            # The time selector is already in the view range, no need to change it.
            logger.debug(
                f"Selected time {current_time:.3f} s is already in the audio view "
                f"range [{window_min:.3f}, {window_max:.3f}] seconds. No change needed."
            )
            return

        if current_time < window_min:
            moves_needed = int(np.ceil((window_min - current_time) / window_len))
            new_window_min = window_min - moves_needed * window_len
            new_window_max = window_max - moves_needed * window_len
        else:  # selected_time > window_max
            moves_needed = int(np.ceil((current_time - window_max) / window_len))
            new_window_min = window_min + moves_needed * window_len
            new_window_max = window_max + moves_needed * window_len

        logger.debug(
            f"Moving audio view to include selected time {current_time:.3f} seconds."
        )
        self._set_clamped_time_range(new_window_min, new_window_max)

    def _set_clamped_time_range(self, new_min: float, new_max: float) -> None:
        """Set x-axis range of the plot ensuring it does not exceed audio duration."""
        min_time = 0.0
        max_time = self._audio.duration

        # NOTE: My IDE does not know the correct signature of setXRange, so I have
        # ignored the warnings.

        if new_min < min_time:
            logger.debug(
                "Setting audio time range to start: "
                f"[{min_time}, {self._visible_duration_seconds}] with visible duration "
                f"{self._visible_duration_seconds} seconds."
            )
            self._plot_widget.setXRange(
                min_time,
                self._visible_duration_seconds,
                padding=0,  # type: ignore
            )

        elif new_max > max_time:
            logger.debug(
                "Setting audio time range to end: "
                f"[{max_time - self._visible_duration_seconds}, {max_time}] with "
                f"visible duration {self._visible_duration_seconds} seconds."
            )
            self._plot_widget.setXRange(
                max_time - self._visible_duration_seconds,
                max_time,
                padding=0,  # type: ignore
            )
        else:
            logger.debug(
                f"Setting audio time range to: [{new_min}, {new_max}] seconds "
                f"with visible duration {self._visible_duration_seconds} seconds."
            )
            self._plot_widget.setXRange(new_min, new_max, padding=0)  # type: ignore

    def _plot_selected_channel(self) -> None:
        """Update the plot to show the selected channel or mean of all channels."""
        # Clear previous plots
        self._plot_widget.clear()
        # Re-add the time selector after clearing
        self._time_selector.add_to_plot(self._plot_widget)

        # Plot min and max of the audio in windows.
        times, audio_min, audio_max = self._audio.get_min_max_envelope(
            window_size=self._plotting_window_size,
            channel_idx=self._channel_selection,
            sample_range=None,  # Plot the whole audio
        )
        logger.debug(f"Plotting {len(audio_min)} samples of audio envelope.")
        # Plot a filled region between min and max
        upper_curve = self._plot_widget.plot(
            times, audio_max, pen=pg.mkPen("b", width=1)
        )
        lower_curve = self._plot_widget.plot(
            times, audio_min, pen=pg.mkPen("b", width=1)
        )
        fill = pg.FillBetweenItem(
            upper_curve,
            lower_curve,
            brush=(100, 100, 255, 80),
        )
        self._plot_widget.addItem(fill)

    @Slot()
    def _on_time_selector_moved(self) -> None:
        """Handle when the selector line is moved by the user.

        Updates the current sample based on the new position of the selector
        and emits a signal for the position change. Does not change the visible window.
        """
        # Clamp the new time both to the current view range to make it impossible to
        # move the selector outside the visible range and to audio duration.
        view_min, view_max = self._plot_widget.viewRange()[0]
        clamp_range = (max(0.0, view_min), min(self._audio.duration, view_max))
        self._time_selector.clamp_selected_time_to_range(
            clamp_range, padding=self._time_selector_padding
        )
        clamped_time = self._time_selector.selected_time
        new_sample = int(clamped_time * self._audio.sampling_rate)

        # Update currently selected sample and time label.
        self._current_sample = new_sample  # Updates self.current_time automatically
        self._time_label.set_current_time(self.current_time)

        # Emit signal for position change
        self.sigSampleIndexChanged.emit(new_sample)

    @Slot(int)
    def _update_selected_channel(self, index: int) -> None:
        """Handle when the user changes the selected channel."""
        if index == 0:
            # If "All (show mean)" is selected, set channel selection to None
            self._channel_selection = None
        else:
            self._channel_selection = index - 1  # Adjust for "All" being index 0

        self._plot_selected_channel()
        self.sigChannelSelectionChanged.emit(self._channel_selection)

    @Slot()
    def _zoom_in(self) -> None:
        """Zoom in on the waveform."""
        # Halve the visible duration
        self._visible_duration_seconds = max(1.0, self._visible_duration_seconds / 2)
        self._center_view_on_current_sample()

    @Slot()
    def _zoom_out(self) -> None:
        """Zoom out on the waveform."""
        # Double the visible duration
        self._visible_duration_seconds = min(
            self._audio.duration, self._visible_duration_seconds * 2
        )
        self._center_view_on_current_sample()

    @Slot()
    def _reset_zoom(self) -> None:
        """Reset the view to center around the current sample with default zoom."""
        # Reset to default zoom level
        self._visible_duration_seconds = self._default_view_len
        self._center_view_on_current_sample()

    def _center_view_on_current_sample(self) -> None:
        """Try to center the view around the current sample with correct window size.

        Centering wil not be perfect if the current sample is too close to the edges
        of the audio data.
        """
        half_window = self._visible_duration_seconds / 2
        self._set_clamped_time_range(
            self.current_time - half_window, self.current_time + half_window
        )


class AudioBrowser(SyncableBrowserWidget):
    """Qt widget for browsing audio with playback controls.

    This browser allows interactive visualization of audio data from AudioFile objects.

    Parameters
    ----------
    audio : AudioFile
        The audio file to visualize.
    playback_update_interval_ms : int, optional
        Determines how often the visualization is updated during audio playback.
        By default 50 ms, which corresponds to 20 updates per second.
    parent : QWidget | None, optional
        The parent widget, by default None.

    NOTE: This browser currently only supports a single audio file. Methods that receive
    `media_idx` parameter ignore it and signals emit `media_idx` as zero.
    """

    def __init__(
        self,
        audio: AudioFile,
        playback_update_interval_ms: int = 50,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the audio browser."""
        super().__init__(parent=parent)
        self._audio = audio
        # Get sample rate that is suitable for playing audio on the hardware we are
        # running on. Audio will be resampled if necessary.
        self._playing_rate = audio_player.find_sample_rate_for_playing(
            original_rate=audio.sampling_rate
        )

        # Create a timer that will be used to update the visualization during playback.
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(playback_update_interval_ms)
        self._playback_timer.timeout.connect(self._on_playback_timeout)

        self._is_playing = False
        # How many samples to advance in the visualization when the playback timer
        # timeouts (how many samples there is in the playback update interval).
        self._playback_elapsed_samples = int(
            playback_update_interval_ms / 1000 * audio.sampling_rate
        )

        self.setWindowTitle("Audio Browser")

        # Create the main layout
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # Create an audio view that handles the visualization.
        self._audio_view = AudioView(audio, parent=self)
        self._audio_view.sigSampleIndexChanged.connect(
            self._on_audio_view_sample_change
        )
        # Keep playback audio data in sync with the visualized data.
        self._audio_view.sigChannelSelectionChanged.connect(
            self._set_playback_audio_data
        )
        self._layout.addWidget(self._audio_view)

        # Create controls.

        self._slider = gui_utils.IndexSlider(
            min_value=0, max_value=audio.n_samples - 1, value=0, parent=self
        )
        self._slider.sigIndexChanged.connect(
            lambda idx: self.set_position(idx, 0, signal=True)
        )
        self._layout.addWidget(self._slider)

        self._navigation_bar = gui_utils.NavigationBar(
            prev_button_text="Backwards",
            next_button_text="Forward",
            parent=self,
        )
        self._layout.addWidget(self._navigation_bar)
        self._navigation_bar.sigPlayPauseClicked.connect(self._toggle_play_pause)
        self._navigation_bar.sigNextClicked.connect(self._jump_forward)
        self._navigation_bar.sigPreviousClicked.connect(self._jump_backwards)

        self._update_browser_to_current_sample()
        # Sets self._playback_audio_data
        self._set_playback_audio_data(channel_idx=self._audio_view.channel_selection)

    @property
    def current_sample(self) -> int:
        """Get the index of the current sample position."""
        return self._audio_view.current_sample

    @property
    def current_time(self) -> float:
        """Get the current position in seconds."""
        return self._audio_view.current_time

    @property
    def is_playing(self) -> bool:
        """Return whether the audio is currently playing."""
        return self._is_playing

    def set_position(
        self, position_idx: int, media_idx: int, signal: bool = True
    ) -> bool:
        """Set the current position to the given sample index.

        NOTE: Does not do anything with media_idx, as this browser currently only
        supports a single audio file.
        """
        success = self._audio_view.display_at_sample(position_idx, signal=False)
        if not success:
            logger.debug(
                f"Cannot set position to sample index {position_idx} (out of bounds). "
                "Keeping current position."
            )
            return False
        self._update_browser_to_current_sample()
        if signal:
            # Emit zero as media_idx.
            self.sigPositionChanged.emit(0, position_idx)
        return True

    def get_current_position(self, media_idx: int) -> int:
        """Get the index of the current sample position.

        Parameter media_idx is ignored as this browser only supports a single audio
        file.
        """
        if media_idx != 0:
            logger.warning(
                f"AudioBrowser only supports a single audio file, but was asked for "
                f"current position of media on index {media_idx}. Returning position "
                f"of the only audio file (index 0)."
            )
        return self.current_sample

    def jump_to_end(self, media_idx: int, signal: bool = True) -> None:
        """Display the last sample of the audio.

        NOTE: Does not do anything with media_idx, as this browser currently only
        supports a single audio file.
        """
        last_sample = self._audio.n_samples - 1
        self.set_position(last_sample, media_idx, signal=signal)

    def jump_to_start(self, media_idx: int, signal: bool = True) -> None:
        """Display the first sample of the audio.

        NOTE: Does not do anything with media_idx, as this browser currently only
        supports a single audio file.
        """
        self.set_position(0, media_idx, signal=signal)

    def start_playback(self, media_idx: int) -> None:
        """Start the audio playback from the current position."""
        # Audio browser currently only supports a single audio file,
        # so media_idx is not used.
        logger.debug("Starting audio playback.")

        visualized_time = self.current_time
        # Starting sample for playback depends on the sample rate used for playing.
        playback_start_sample = int(visualized_time * self._playing_rate)

        audio_player.play(
            self._playback_audio_data[playback_start_sample:],
            sampling_rate=self._playing_rate,
        )
        self._navigation_bar.set_playing()
        self._playback_timer.start()
        self._is_playing = True

        self.sigPlaybackStateChanged.emit(0, True)  # emit zero as media_idx

    def pause_playback(self) -> None:
        """Pause the audio playback."""
        logger.debug("Pausing audio playback.")
        audio_player.stop()
        self._navigation_bar.set_paused()
        self._playback_timer.stop()
        self._is_playing = False

        self.sigPlaybackStateChanged.emit(0, False)  # emit zero as media_idx

    def _update_browser_to_current_sample(self) -> None:
        """Update the audio browser UI to reflect the currently selected sample."""
        self._slider.set_value(self.current_sample, signal=False)
        self._update_buttons_enabled()

    @Slot(int)
    def _on_audio_view_sample_change(self, sample_idx: int) -> None:
        """Handle when the user dragged the time selector in the audio view."""
        # The updated sample index is fetched from the audio view using
        # self.current_sample.
        self._update_browser_to_current_sample()
        # Emit the position changed signal with zero as media index.
        self.sigPositionChanged.emit(0, sample_idx)

    @Slot(object)
    def _set_playback_audio_data(self, channel_idx: int | None) -> None:
        """Set the audio data to play when user changes selected channel."""
        if self._playing_rate == self._audio.sampling_rate:
            if channel_idx is None:
                self._playback_audio_data = self._audio.get_audio_mean()
            else:
                # Get data for the selected channel.
                self._playback_audio_data = self._audio.get_audio_all_channels()[
                    channel_idx, :
                ]
        else:
            logger.info(
                f"Resampling audio from {self._audio.sampling_rate} Hz to "
                f"{self._playing_rate} Hz for playback."
            )
            self._playback_audio_data = self._audio.resample_poly(
                self._playing_rate, channel_idx=channel_idx
            )

    @Slot()
    def _on_playback_timeout(self) -> None:
        """Update the visualization during playback when playback timer timeouts."""
        if not self._is_playing:
            logger.warning(
                "Playback timer timed out when self._is_playing is False. "
                "Skipping updating position."
            )
            return
        # Calculate at which sample index the playback is currently at.
        playback_sample_idx = self.current_sample + self._playback_elapsed_samples
        success = self.set_position(playback_sample_idx, media_idx=0, signal=True)
        if not success:
            # Pause playback if the sample index is out of bounds.
            self.pause_playback()

    @Slot()
    def _toggle_play_pause(self) -> None:
        """Start or stop the audio playback."""
        if self._is_playing:
            self.pause_playback()
        else:
            self.start_playback(media_idx=0)  # media_idx is not used in audio browser

    @Slot()
    def _jump_forward(self) -> None:
        """Advance one second in the audio."""
        samples_to_advance = int(self._audio.sampling_rate)
        new_sample = self.current_sample + samples_to_advance
        self.set_position(new_sample, 0, signal=True)

    @Slot()
    def _jump_backwards(self) -> None:
        """Go back one second in the audio."""
        samples_to_rewind = int(self._audio.sampling_rate)
        new_sample = self.current_sample - samples_to_rewind
        self.set_position(new_sample, 0, signal=True)

    def _update_buttons_enabled(self) -> None:
        """Enable or disable buttons based on the current position."""
        # Buttons advance or rewind one second, so we need to check
        # if that is possible.
        max_time = self._audio.duration - 1.0  # seconds
        min_time = 1.0

        self._navigation_bar.set_prev_enabled(self.current_time >= min_time)
        self._navigation_bar.set_next_enabled(self.current_time <= max_time)

        self._navigation_bar.set_play_pause_enabled(
            self.current_sample < self._audio.n_samples - 1
        )
