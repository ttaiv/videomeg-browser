import logging
import sys

from videomeg_browser.video import VideoFileHelsinkiVideoMEG
from videomeg_browser.video_browser import VideoBrowser

try:
    from qtpy.QtCore import Qt
except Exception as exc:
    if exc.__class__.__name__ == "QtBindingsNotFoundError":
        raise ImportError(
            "No Qt binding found, please install PyQt6, PyQt5, PySide6, or PySide2"
        ) from None
    else:
        raise

from qtpy.QtWidgets import QApplication

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
app = QApplication([])

video = VideoFileHelsinkiVideoMEG(
    "/u/69/taivait1/unix/video_meg_testing/Subject_2_Luna/Video_MEG/animal_meg_subject_2_240614.video.dat",
    magic_str="ELEKTA_VIDEO_FILE"
)

window = VideoBrowser([video])
window.resize(1000, 800)
window.show()
sys.exit(app.exec_())
