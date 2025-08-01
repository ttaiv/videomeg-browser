"""Demo script for the AudioBrowser.

Running this requires an audio file in Helsinki videoMEG project format.
Adjust the file path as needed.
"""

import logging
import sys

from qtpy.QtWidgets import QApplication

from videomeg_browser.audio import AudioFileHelsinkiVideoMEG
from videomeg_browser.audio_browser import AudioBrowser


def main() -> None:
    """Run the audio browser demo."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Replace this with the path to your audio file
    audio_path = "/u/69/taivait1/unix/video_meg_testing/2025-07-11_MEG2MEG_test/2025-07-11--18-18-41_audio_00.aud"

    try:
        # Check if the file exists
        import os

        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found at {audio_path}")
            print("Please adjust the file path in the script.")
            sys.exit(1)

        # Try different magic strings that might be used in the file
        magic_strings = [
            "HELSINKI_VIDEO_MEG_PROJECT_AUDIO_FILE",  # Default
            "ELEKTA_AUDIO_FILE",  # Alternative that might be used
        ]

        audio = None
        for magic_str in magic_strings:
            try:
                print(f"Trying with magic string: {magic_str}")
                audio = AudioFileHelsinkiVideoMEG(audio_path, magic_str=magic_str)
                break
            except ValueError as e:
                print(f"Failed with magic string '{magic_str}': {str(e)}")
                continue

        if audio is None:
            print("Could not open the audio file with any known magic string.")
            print("Please check the file format or provide the correct magic string.")
            sys.exit(1)

        # Print audio stats
        audio.print_stats()

        # Create and run the application
        app = QApplication([])
        window = AudioBrowser(audio)
        window.resize(1000, 600)
        window.show()
        sys.exit(app.exec_())
    except FileNotFoundError:
        print(f"Error: Audio file not found at {audio_path}")
        print("Please adjust the file path in the script.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
