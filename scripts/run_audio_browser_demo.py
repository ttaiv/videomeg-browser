"""Demo script for the AudioBrowser.

Running this requires an audio file in Helsinki videoMEG project format.
Adjust the file path as needed.
"""

import logging
import sys

from qtpy.QtWidgets import QApplication

from videomeg_browser import AudioFileHelsinkiVideoMEG
from videomeg_browser.browsers import AudioBrowser


def main() -> None:
    """Run the audio browser demo."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Replace this with the path to your audio file
    audio_path = (
        "/u/69/taivait1/unix/video_meg_testing/2025-07-11_MEG2MEG_test/"
        "2025-07-11--18-18-41_audio_00.aud"
    )

    audio = AudioFileHelsinkiVideoMEG(audio_path)
    audio.unpack_audio()

    audio.print_stats()

    app = QApplication([])
    window = AudioBrowser(audio)
    window.resize(1000, 600)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
