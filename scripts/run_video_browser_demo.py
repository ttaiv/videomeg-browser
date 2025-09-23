"""Browse a video file with video browser.

Running this requires some video file. It can be either a video file recorded
with Helsinki videoMEG project software or a standard video file format like .mp4.
Adjust the file path and video type as needed.
"""

import logging
import sys

from qtpy.QtWidgets import QApplication

from videomeg_browser import VideoFileCV2, VideoFileHelsinkiVideoMEG  # noqa: F401
from videomeg_browser.browsers import VideoBrowser


def main() -> None:
    """Run the video browser demo."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    video_path = (
        "/u/69/taivait1/unix/video_meg_testing/Subject_2_Luna/Video_MEG/"
        "animal_meg_subject_2_240614.video.dat"
    )

    # Choose the type of video file by uncommenting/commenting.

    video = VideoFileHelsinkiVideoMEG(video_path, magic_str="ELEKTA_VIDEO_FILE")
    # video = VideoFileCV2(video_path)  # suitable for .mp4, .avi, etc.

    # Print video stats
    video.print_stats()

    app = QApplication([])
    window = VideoBrowser([video])
    window.resize(1000, 800)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
