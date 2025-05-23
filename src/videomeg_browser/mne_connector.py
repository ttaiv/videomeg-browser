import os.path as op

import mne
from mne_qt_browser._pg_figure import TimeScrollBar
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QDockWidget,
    QSlider,
)

from .browser import VideoBrowser, VideoFile


class ScrollBarSynchronizer:
    """Synchronize the scroll bar of the raw data browser with the video browser."""

    def __init__(self, raw_scroll_bar: TimeScrollBar, vid_slider: QSlider):
        self.raw_scroll_bar = raw_scroll_bar
        self.vid_slider = vid_slider
        self._syncing = False

        self.raw_scroll_bar.valueChanged.connect(self._sync_video_to_raw)
        self.vid_slider.valueChanged.connect(self._sync_raw_to_video)

    def _sync_video_to_raw(self, value):
        """Update the video position based on the raw data browser's scroll bar."""
        if not self._syncing:
            self._syncing = True
            self.vid_slider.setValue(value)
            self._syncing = False

    def _sync_raw_to_video(self, value):
        """Update the raw data browser's scroll bar based on the video slider."""
        if not self._syncing:
            self._syncing = True
            self.raw_scroll_bar.setValue(value)
            self._syncing = False


def run_together_with_raw_data_browser():
    """Run the video browser together with the raw data browser."""
    base_path = "/u/69/taivait1/unix/video_meg_testing/Subject_2_Luna"

    # Create a video file object
    video_file = VideoFile(op.join(base_path, "Video_MEG", "kapsu.mp4"))

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

    # Sync the scroll bars
    sync = ScrollBarSynchronizer(raw_browser.mne.ax_hscroll, video_browser.frame_slider)

    # Profit
    raw_browser.show()
    app.exec_()


if __name__ == "__main__":
    run_together_with_raw_data_browser()
