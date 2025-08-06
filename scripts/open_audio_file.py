"""Example script to open an audio file and inspect it."""

import logging

import matplotlib.pyplot as plt

from videomeg_browser.audio import AudioFileHelsinkiVideoMEG


def main() -> None:
    """Open an audio file and inspect its contents."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    audio_path = (
        "/u/69/taivait1/unix/video_meg_testing/2025-07-11_MEG2MEG_test/"
        "2025-07-11--18-18-41_audio_00.aud"
    )

    audio = AudioFileHelsinkiVideoMEG(audio_path)
    audio.unpack_audio()

    audio.print_stats()

    audio_ts = audio.get_audio_timestamps()
    print(f"Got audio timestamps with shape: {audio_ts.shape}")

    audio_samples = audio.get_audio_all_channels()
    print(f"Got audio samples with shape: {audio_samples.shape}")

    # Example: Extract audio from the first channel (microphone)
    audio_from_microphone = audio_samples[0, :]

    plt.figure(figsize=(12, 4))
    plt.xlabel("Timestamps (ms)")
    plt.ylabel("Amplitude")
    plt.plot(audio_ts, audio_from_microphone)
    plt.title("Audio Signal from Microphone")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
