try:
    from qtpy.QtCore import Qt
except Exception as exc:
    if exc.__class__.__name__ == "QtBindingsNotFoundError":
        raise ImportError(
            "No Qt binding found, please install PyQt6, PyQt5, PySide6, or PySide2"
        ) from None
    else:
        raise

import sys

import cv2
import pyqtgraph as pg
from cv2.typing import MatLike
from qtpy.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget

pg.setConfigOptions(imageAxisOrder="row-major")


class VideoFile:
    """
    Data object that holds a video file and provides methods to read frames from it.
    """

    def __init__(self, fname: str) -> None:
        self.fname = fname
        # Capture the video file for processing
        self.cap = cv2.VideoCapture(fname)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {fname}")

        # Store video properties
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Matches to cv2.CAP_PROP_POS_FRAMES and tells the index of the next frame to be read
        self.next_frame_idx = 0

    def read_next_frame(self) -> MatLike | None:
        """
        Read the next frame from the video file.
        """
        if not self.cap.isOpened():
            raise ValueError("Trying to read from a closed video file.")

        ret, frame = self.cap.read()
        if not ret:
            # End of video?
            return None

        self.next_frame_idx += 1
        # Convert the frame from BGR to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    def read_previous_frame(self) -> MatLike | None:
        """
        Read the frame before the last read frame from the video file.
        """
        if not self.cap.isOpened():
            raise ValueError("Trying to read from a closed video file.")

        if self.next_frame_idx < 2:
            print("Already at the first frame.")
            return None

        # Last read frame is one step back, the frame before that is two steps back
        self._set_next_frame(self.next_frame_idx - 2)
        return self.read_next_frame()

    def _set_next_frame(self, frame_idx: int) -> None:
        """
        Set the next frame to be read from the video file.
        """
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise ValueError(f"Frame index out of bounds: {frame_idx}")

        self.next_frame_idx = frame_idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


class VideoBrowser(QMainWindow):
    def __init__(self, video: VideoFile):
        super().__init__()
        self.video = video

        self.setWindowTitle("Video Browser Prototype")

        # Create container to hold the plot and button
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout()
        container.setLayout(layout)

        # Create an ImageView widget to display the video frames
        self.im_view = pg.ImageView()
        self.im_view.setImage(self.video.read_next_frame())
        layout.addWidget(self.im_view)

        # Create a button for navigating to the next frame
        self.button = QPushButton("Next Frame")
        self.button.clicked.connect(self.display_next_frame)
        layout.addWidget(self.button)

        # Create a button for navigating to the previous frame
        self.prev_button = QPushButton("Previous Frame")
        self.prev_button.clicked.connect(self.display_previous_frame)
        layout.addWidget(self.prev_button)

    def display_next_frame(self):
        """
        Display the next frame in the video.
        """
        frame = self.video.read_next_frame()
        if frame is None:
            print("End of video reached.")
            return

        self.im_view.setImage(frame)

    def display_previous_frame(self):
        """
        Display the previous frame in the video.
        """
        frame = self.video.read_previous_frame()
        if frame is None:
            print("Already at the first frame.")
            return

        self.im_view.setImage(frame)


if __name__ == "__main__":
    app = QApplication([])

    video = VideoFile(
        "/u/69/taivait1/unix/video_meg_testing/Subject_2_Luna/export_video/animal_meg_subject_2_240614_cropped.avi"
    )

    window = VideoBrowser(video)
    window.resize(800, 600)
    window.show()

    sys.exit(app.exec_())
