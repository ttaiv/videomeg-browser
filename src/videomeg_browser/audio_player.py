"""Contains functions for playing audio using sounddevice."""

import logging

import numpy as np
import numpy.typing as npt
import sounddevice as sd

logger = logging.getLogger(__name__)


def play(audio: npt.NDArray[np.float32], sampling_rate: int) -> None:
    """Start playing audio.

    Parameters
    ----------
    audio : npt.NDArray[np.float32]
       1D array of audio samples to be played.
    sampling_rate : int
        The sampling rate of the audio data.
    """
    if audio.ndim != 1:
        raise ValueError(
            "Audio data must be a 1D array of samples, multiple channels are not "
            "currently supported."
        )
    sd.play(audio, samplerate=sampling_rate)


def stop() -> None:
    """Stop current audio playback."""
    sd.stop()


def find_sample_rate_for_playing(original_rate: int) -> int:
    """Find the best supported sample rate for the default output device.

    Parameters
    ----------
    original_rate : int
        The original sample rate of the audio file to be played.

    Returns
    -------
    int
        The best supported sample rate for the default output device.
    """
    # Common sample rates to try (in order of preference)
    preferred_rates = [48000, 44100, 96000, 88200, 32000, 22050, 16000, 8000]

    # Move or add original rate to the front of the list to try it first.
    if original_rate in preferred_rates:
        preferred_rates.remove(original_rate)
    preferred_rates.insert(0, original_rate)

    # Get the default output device info
    device_info = sd.query_devices(kind="output")
    assert isinstance(device_info, dict), "Device info for one device should be a dict."
    logger.info(
        f"Default output device: {device_info['name']} has default sample rate "
        f"{device_info['default_samplerate']} Hz"
    )

    for rate in preferred_rates:
        try:
            sd.check_output_settings(samplerate=rate)
            logger.info(f"Selected output sample rate to be {rate} Hz.")
            return rate
        except sd.PortAudioError:
            continue

    # Fallback: use device's default sample rate
    default_rate = int(device_info["default_samplerate"])
    logger.info(
        f"Using default output sample rate {default_rate} Hz as default output "
        "device does not support preferred rates."
    )
    return default_rate
