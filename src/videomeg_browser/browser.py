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
import sys

import pyqtgraph as pg
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from .video import VideoFile, VideoFileCV2, VideoFileHelsinkiVideoMEG

logger = logging.getLogger(__name__)

pg.setConfigOptions(imageAxisOrder="row-major")


class VideoBrowser(QWidget):
    """A browser for viewing video frames.

    Parameters
    ----------
    video : VideoFile
        The video file to be displayed.
    """

    def __init__(self, video: VideoFile):
        super().__init__()
        self.video = video
        self.current_frame_idx = 0

        self.setWindowTitle("Video Browser Prototype")

        layout = QVBoxLayout(self)

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

    def display_next_frame(self):
        """Display the next frame in the video."""
        frame = self.video.get_frame_at(self.current_frame_idx + 1)
        if frame is None:
            logger.debug(
                "Skipping updating to the next frame, already at the last frame."
            )
            return

        self.current_frame_idx += 1
        self.im_view.setImage(frame)
        self.update_frame_label()
        self.update_slider()

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

    def slider_frame_changed(self, value: int):
        """Update view to display the frame corresponding to the slider's position."""
        self.current_frame_idx = value
        frame = self.video.get_frame_at(self.current_frame_idx)
        if frame is None:
            raise ValueError(f"Invalid frame index {value} selected with the slider.")

        self.im_view.setImage(frame)
        self.update_frame_label()

    def update_frame_label(self):
        """Update the frame label to show the current frame number."""
        # Use one-based index for display
        self.frame_label.setText(
            f"Current Frame: {self.current_frame_idx + 1}/{self.video.frame_count}"
        )

    def update_slider(self):
        """Update the slider to reflect the current frame index."""
        # Use one-based index for display
        self.frame_slider.setValue(self.current_frame_idx)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    app = QApplication([])

    video_videomeg = VideoFileHelsinkiVideoMEG(
        "/u/69/taivait1/unix/video_meg_testing/Subject_2_Luna/Video_MEG/animal_meg_subject_2_240614.video.dat"
    )
    video_exported = VideoFileCV2(
        "/u/69/taivait1/unix/video_meg_testing/Subject_2_Luna/export_video/animal_meg_subject_2_240614.avi"
    )

    window = VideoBrowser(video_videomeg)
    window.resize(1000, 800)
    window.show()
    sys.exit(app.exec_())
