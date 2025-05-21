try:
    from qtpy.QtCore import Qt
except Exception as exc:
    if exc.__class__.__name__ == "QtBindingsNotFoundError":
        raise ImportError(
            "No Qt binding found, please install PyQt6, PyQt5, PySide6, or PySide2"
        ) from None
    else:
        raise

from cv2.typing import NumPyArrayNumeric
from numpy import dtype, floating, integer
from qtpy.QtWidgets import QMainWindow, QApplication, QVBoxLayout, QWidget, QPushButton
import pyqtgraph as pg
import sys
import cv2
import numpy as np

from cv2.typing import MatLike

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

        self.frame_index = -1
        # Read the first frame (increments frame_index)
        self.current_frame = self.read_next_frame()

    def read_next_frame(self) -> MatLike | None:
        """
        Read the next frame from the video file and set is as the current frame.
        """
        if not self.cap.isOpened():
            raise ValueError("Trying to read from a closed video file.")

        ret, frame = self.cap.read()
        if not ret:
            # End of video?
            return None
        
        self.frame_index += 1
        # Convert the frame from BGR to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.current_frame = frame
        return frame

    '''
    def previous_frame(self):
        """
        Seek to the previous frame.
        """
        if self.frame_index > 0:
            self.frame_index -= 1
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        return self.next_frame()

    def jump_to_frame(self, frame_number: int):
        """
        Seek to an arbitrary frame number.
        """
        if 0 <= frame_number < self.frame_count:
            self.frame_index = frame_number
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        return self.next_frame()
    '''


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
        self.im_view.setImage(self.video.current_frame)
        layout.addWidget(self.im_view)

        # Create a button
        self.button = QPushButton("Next Frame")
        self.button.clicked.connect(self.display_next_frame)
        layout.addWidget(self.button)

    
    def display_next_frame(self):
        """
        Display the next frame in the video.
        """
        frame = self.video.read_next_frame()
        if frame is None:
            print("End of video reached.")
            return
        
        self.im_view.setImage(frame)

if __name__ == "__main__":
    app = QApplication([])

    video = VideoFile("/u/69/taivait1/unix/video_meg_testing/Subject_2_Luna/export_video/animal_meg_subject_2_240614_with_audio.avi")
    window = VideoBrowser(video)
    window.resize(800, 600)
    window.show()

    sys.exit(app.exec_())


