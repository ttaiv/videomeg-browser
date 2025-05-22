import os.path as op

import mne
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QDockWidget,
)

from .browser import VideoBrowser, VideoFile


def run_together_with_raw_data_browser():
    """Run the video browser together with the raw data browser."""
    base_path = "/u/69/taivait1/unix/video_meg_testing/Subject_2_Luna"

    # Create a video file object
    video_file = VideoFile(
        op.join(base_path, "export_video", "animal_meg_subject_2_240614.avi")
    )

    # Create a raw data object
    raw = mne.io.read_raw_fif(
        op.join(base_path, "Raw", "animal_meg_subject_2_240614.fif"), preload=True
    )

    # Set up Qt application
    app = QApplication([])
    # Instantiate the MNE Qt Browser
    raw_browser = raw.plot(block=False)

    # Set up the video browser
    video_browser = VideoBrowser(video_file)

    # Dock the video browser to the raw data browser with Qt magic
    dock = QDockWidget("Video Browser", raw_browser)
    dock.setWidget(video_browser)
    dock.setFloating(True)
    raw_browser.addDockWidget(Qt.RightDockWidgetArea, dock)

    # Profit
    raw_browser.show()
    app.exec_()


if __name__ == "__main__":
    run_together_with_raw_data_browser()
