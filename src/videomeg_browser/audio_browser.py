"""Contains AudioBrowser Qt widget for displaying audio waveforms."""

import logging
import os.path
from typing import Literal

import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import QTimer, Signal, Slot  # type: ignore
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .audio import AudioFile

logger = logging.getLogger(__name__)


class AudioBrowser(QWidget):
    """Qt widget for visualizing audio waveforms.

    This browser allows visualization of audio data from AudioFile objects.
    The browser provides controls to navigate through the audio, play/pause the audio,
    and display audio channels individually or as a mean.

    Parameters
    ----------
    audio : AudioFile
        The audio file to visualize.
    display_mode : Literal["waveform", "spectrogram"], optional
        The initial display mode, either "waveform" or "spectrogram",
        by default "waveform".
    parent : QWidget | None, optional
        The parent widget, by default None.
    """

    sigPlaybackPositionChanged = Signal(float)  # position in seconds

    def __init__(
        self,
        audio: AudioFile,
        display_mode: Literal["waveform", "spectrogram"] = "waveform",
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the audio browser."""
        super().__init__(parent=parent)
        self._audio = audio
        self._display_mode = display_mode
        self._current_sample = 0
        self._is_playing = False
        self._visible_duration_seconds = 5.0  # Window size in seconds
        self._channel_selection = "mean"  # "mean" or channel index

        # Auto-call unpack_audio to ensure all data is loaded before visualization
        if hasattr(self._audio, "unpack_audio"):
            logger.info("Unpacking audio data...")
            getattr(self._audio, "unpack_audio")()

        # Set up timer for playing the audio
        self._play_timer = QTimer(parent=self)
        self._play_timer.timeout.connect(self._advance_playback)
        self._set_play_timer_interval()

        self.setWindowTitle("Audio Browser")

        # Create the main layout
        self._layout = QVBoxLayout()
        self.setLayout(self._layout)

        # Add plot for audio visualization
        self._setup_plot_widget()

        # Add playback controls
        self._setup_playback_controls()

        # Add info panel
        self._setup_info_panel()

        # Initial visualization
        self._update_plot()
        self._update_info_display()

    def _setup_plot_widget(self) -> None:
        """Set up the plot widget for audio visualization."""
        self._plot_widget = pg.PlotWidget()
        self._plot_widget.setBackground("w")
        self._plot_widget.setLabel("bottom", "Time", "s")
        self._plot_widget.setLabel("left", "Amplitude")
        self._plot_widget.showGrid(x=True, y=True, alpha=0.5)

        # Add a vertical line to indicate the current position
        self._position_line = pg.InfiniteLine(
            angle=90, movable=True, pen=pg.mkPen(color="r", width=2)
        )
        self._position_line.sigPositionChanged.connect(self._on_position_line_moved)
        self._plot_widget.addItem(self._position_line)

        # Add the plot to the layout
        self._layout.addWidget(self._plot_widget)

        # Add plots for each channel and the mean
        self._channel_plots = {}  # Store plot items for each channel
        self._mean_plot = None  # Plot for the mean of all channels

    def _setup_playback_controls(self) -> None:
        """Set up playback control buttons and timeline slider."""
        controls_layout = QHBoxLayout()

        # Navigation buttons
        self._prev_button = QPushButton("<<")
        self._prev_button.clicked.connect(self._jump_backward)
        controls_layout.addWidget(self._prev_button)

        self._play_pause_button = QPushButton("Play")
        self._play_pause_button.clicked.connect(self.toggle_play_pause)
        controls_layout.addWidget(self._play_pause_button)

        self._next_button = QPushButton(">>")
        self._next_button.clicked.connect(self._jump_forward)
        controls_layout.addWidget(self._next_button)

        # Time display
        self._time_label = QLabel("00:00 / 00:00")
        controls_layout.addWidget(self._time_label)

        # Channel selection
        controls_layout.addStretch()
        channel_label = QLabel("Channel:")
        controls_layout.addWidget(channel_label)

        self._channel_selector = QComboBox()
        self._channel_selector.addItem("Mean (all channels)")
        for i in range(self._audio.n_channels):
            self._channel_selector.addItem(f"Channel {i + 1}")
        self._channel_selector.currentIndexChanged.connect(self._on_channel_changed)
        controls_layout.addWidget(self._channel_selector)

        # Display mode selection
        display_label = QLabel("Display:")
        controls_layout.addWidget(display_label)

        self._display_mode_selector = QComboBox()
        self._display_mode_selector.addItem("Waveform")
        self._display_mode_selector.addItem("Spectrogram")
        self._display_mode_selector.currentIndexChanged.connect(
            self._on_display_mode_changed
        )
        controls_layout.addWidget(self._display_mode_selector)

        # Zoom controls
        zoom_label = QLabel("Zoom:")
        controls_layout.addWidget(zoom_label)

        self._zoom_in_button = QPushButton("+")
        self._zoom_in_button.clicked.connect(self._zoom_in)
        controls_layout.addWidget(self._zoom_in_button)

        self._zoom_out_button = QPushButton("-")
        self._zoom_out_button.clicked.connect(self._zoom_out)
        controls_layout.addWidget(self._zoom_out_button)

        self._layout.addLayout(controls_layout)

    def _setup_info_panel(self) -> None:
        """Set up the information panel showing audio metadata."""
        info_layout = QHBoxLayout()

        # File info
        file_info = QLabel(f"File: {os.path.basename(self._audio.fname)}")
        info_layout.addWidget(file_info)

        # Audio properties
        self._info_label = QLabel()
        info_layout.addWidget(self._info_label)

        self._layout.addLayout(info_layout)

    def _update_plot(self) -> None:
        """Update the audio plot based on current settings."""
        self._plot_widget.clear()

        # Add the position line back after clearing
        self._plot_widget.addItem(self._position_line)

        # Set the position line to the current sample
        current_time = self._current_sample / self._audio.sampling_rate
        self._position_line.setValue(current_time)

        # Calculate visible window
        half_window = self._visible_duration_seconds / 2
        start_time = max(0, current_time - half_window)
        end_time = min(self._audio.duration, current_time + half_window)

        # Convert times to samples
        start_sample = int(start_time * self._audio.sampling_rate)
        end_sample = int(end_time * self._audio.sampling_rate)

        # Update x-axis limits
        self._plot_widget.setXRange(start_time, end_time)

        if self._display_mode == "waveform":
            self._plot_waveform(start_sample, end_sample)
        else:  # spectrogram
            self._plot_spectrogram(start_sample, end_sample)

        # Update time label
        self._update_time_label()

    def _plot_waveform(self, start_sample: int, end_sample: int) -> None:
        """Plot the audio waveform."""
        # Create time vector for x-axis
        times = np.arange(start_sample, end_sample) / self._audio.sampling_rate

        if self._channel_selection == "mean":
            # Plot the mean of all channels
            audio_data = self._audio.get_audio_mean(
                sample_range=(start_sample, end_sample)
            )
            self._mean_plot = self._plot_widget.plot(
                times, audio_data, pen=pg.mkPen(color="b", width=1)
            )
        else:
            # Plot the selected channel
            channel_idx = int(self._channel_selection)
            audio_data = self._audio.get_audio_all_channels(
                sample_range=(start_sample, end_sample)
            )
            self._channel_plots[channel_idx] = self._plot_widget.plot(
                times, audio_data[channel_idx], pen=pg.mkPen(color="g", width=1)
            )

    def _plot_spectrogram(self, start_sample: int, end_sample: int) -> None:
        """Plot the audio spectrogram."""
        try:
            import scipy.signal as signal

            if self._channel_selection == "mean":
                audio_data = self._audio.get_audio_mean(
                    sample_range=(start_sample, end_sample)
                )
            else:
                channel_idx = int(self._channel_selection)
                audio_data = self._audio.get_audio_all_channels(
                    sample_range=(start_sample, end_sample)
                )
                audio_data = audio_data[channel_idx]

            # Calculate spectrogram
            f, t, Sxx = signal.spectrogram(
                audio_data,
                fs=self._audio.sampling_rate,
                nperseg=min(1024, len(audio_data) // 8),
                noverlap=min(512, len(audio_data) // 16),
            )

            # Convert to dB scale for better visualization
            Sxx = 10 * np.log10(Sxx + 1e-10)

            # Create the image item
            img = pg.ImageItem()
            img.setImage(Sxx)

            # Position the image correctly
            img.scale(
                (end_sample - start_sample) / self._audio.sampling_rate / Sxx.shape[1],
                f[-1] / Sxx.shape[0],
            )
            img.translate(start_sample / self._audio.sampling_rate, 0)

            # Add a color map
            cmap = pg.colormap.get("viridis")
            img.setColorMap(cmap)

            # Add the image to the plot
            self._plot_widget.addItem(img)

            # Update axes
            self._plot_widget.setLabel("left", "Frequency", "Hz")
            self._plot_widget.setLabel("bottom", "Time", "s")

        except ImportError:
            logger.warning("scipy not available. Cannot display spectrogram.")
            # Fall back to waveform
            self._plot_waveform(start_sample, end_sample)

    def _update_info_display(self) -> None:
        """Update the information display."""
        info_text = (
            f"Sampling rate: {self._audio.sampling_rate} Hz | "
            f"Channels: {self._audio.n_channels} | "
            f"Bit depth: {self._audio.bit_depth} bits | "
            f"Duration: {self._audio.duration:.2f} s"
        )
        self._info_label.setText(info_text)

    def _update_time_label(self) -> None:
        """Update the time label to show current position and total duration."""
        current_time = self._current_sample / self._audio.sampling_rate
        total_time = self._audio.duration

        # Format as MM:SS
        current_min, current_sec = divmod(current_time, 60)
        total_min, total_sec = divmod(total_time, 60)

        time_text = f"{int(current_min):02}:{int(current_sec):02} / {int(total_min):02}:{int(total_sec):02}"
        self._time_label.setText(time_text)

    def _set_play_timer_interval(self) -> None:
        """Set the play timer interval based on current settings."""
        # Update every 50ms for smooth playback
        self._play_timer.setInterval(50)

    def _advance_playback(self) -> None:
        """Advance the playback position."""
        # Calculate how many samples to advance based on timer interval
        interval_ms = self._play_timer.interval()
        samples_to_advance = int(self._audio.sampling_rate * interval_ms / 1000)

        # Update current sample
        self._current_sample += samples_to_advance

        # Check if we've reached the end
        if self._current_sample >= self._audio.n_samples:
            self._current_sample = self._audio.n_samples - 1
            self.pause()

        # Update the visualization
        self._update_plot()

        # Emit signal for current position
        self.sigPlaybackPositionChanged.emit(
            self._current_sample / self._audio.sampling_rate
        )

    def _on_position_line_moved(self) -> None:
        """Handle when the position line is moved by the user."""
        new_time = self._position_line.value()
        new_sample = int(new_time * self._audio.sampling_rate)

        # Clamp to valid range
        new_sample = max(0, min(new_sample, self._audio.n_samples - 1))

        # Update current position
        self._current_sample = new_sample

        # Update the display
        self._update_time_label()

        # Emit signal for position change
        self.sigPlaybackPositionChanged.emit(new_time)

    def _on_channel_changed(self, index: int) -> None:
        """Handle when the user changes the selected channel."""
        if index == 0:
            self._channel_selection = "mean"
        else:
            self._channel_selection = str(index - 1)  # Convert to 0-based index

        # Update the visualization
        self._update_plot()

    def _on_display_mode_changed(self, index: int) -> None:
        """Handle when the user changes the display mode."""
        if index == 0:
            self._display_mode = "waveform"
        else:
            self._display_mode = "spectrogram"

        # Update the visualization
        self._update_plot()

    def _jump_backward(self) -> None:
        """Jump backward by a set amount."""
        # Jump back 1 second
        jump_samples = int(self._audio.sampling_rate * 1.0)
        self._current_sample = max(0, self._current_sample - jump_samples)

        # Update the visualization
        self._update_plot()

    def _jump_forward(self) -> None:
        """Jump forward by a set amount."""
        # Jump forward 1 second
        jump_samples = int(self._audio.sampling_rate * 1.0)
        self._current_sample = min(
            self._audio.n_samples - 1, self._current_sample + jump_samples
        )

        # Update the visualization
        self._update_plot()

    def _zoom_in(self) -> None:
        """Zoom in on the waveform."""
        # Halve the visible duration
        self._visible_duration_seconds = max(1.0, self._visible_duration_seconds / 2)

        # Update the visualization
        self._update_plot()

    def _zoom_out(self) -> None:
        """Zoom out on the waveform."""
        # Double the visible duration
        self._visible_duration_seconds = min(
            self._audio.duration, self._visible_duration_seconds * 2
        )

        # Update the visualization
        self._update_plot()

    @Slot()
    def play(self) -> None:
        """Start playback of the audio."""
        if not self._is_playing:
            self._is_playing = True
            self._play_timer.start()
            self._play_pause_button.setText("Pause")

    @Slot()
    def pause(self) -> None:
        """Pause playback of the audio."""
        if self._is_playing:
            self._is_playing = False
            self._play_timer.stop()
            self._play_pause_button.setText("Play")

    @Slot()
    def toggle_play_pause(self) -> None:
        """Toggle between play and pause states."""
        if self._is_playing:
            self.pause()
        else:
            self.play()

    @Slot(int)
    def set_position(self, sample: int) -> None:
        """Set the current position to the given sample index."""
        # Clamp to valid range
        self._current_sample = max(0, min(sample, self._audio.n_samples - 1))

        # Update the visualization
        self._update_plot()

        # Emit signal for position change
        self.sigPlaybackPositionChanged.emit(
            self._current_sample / self._audio.sampling_rate
        )

    @Slot(float)
    def set_position_time(self, time_seconds: float) -> None:
        """Set the current position to the given time in seconds."""
        sample = int(time_seconds * self._audio.sampling_rate)
        self.set_position(sample)

    def get_current_position(self) -> float:
        """Get the current playback position in seconds."""
        return self._current_sample / self._audio.sampling_rate

    @property
    def audio(self) -> AudioFile:
        """Get the audio file being visualized."""
        return self._audio
