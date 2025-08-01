"""Contains AudioFile interface and its implemmentations for reading audio files."""

# License: BSD-3-Clause
# Copyright (c) 2014 BioMag Laboratory, Helsinki University Central Hospital
# Copyright (c) 2025 Aalto University

import logging
import struct
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from .helsinki_videomeg_file_utils import UnknownVersionError, read_attrib

logger = logging.getLogger(__name__)

_REGR_SEGM_LENGTH = 20  # seconds, should be integer


class AudioFile(ABC):
    """Handles reading audio files."""

    def __init__(self, fname: str) -> None:
        """Initialize the audio file reader with the given file name."""
        self._fname = fname

    @abstractmethod
    def get_audio_all_channels(
        self, sample_range: tuple[int, int] | None = None
    ) -> npt.NDArray[np.float32]:
        """Get audio data for all channels in the specified sample range.

        Parameters
        ----------
        sample_range : tuple[int, int] | None
            A tuple specifying the start and end (exclusive) sample indices to include
            in the output. If None (default), all the samples are included.

        Returns
        -------
        npt.NDArray[np.float32]
            A 2D array of shape (n_channes, n_samples) containing the audio data.
        """
        pass

    @abstractmethod
    def get_audio_mean(
        self, sample_range: tuple[int, int] | None = None
    ) -> npt.NDArray[np.float32]:
        """Get mean audio data across channels in the specified sample range.

        Parameters
        ----------
        sample_range : tuple[int, int] | None
            A tuple specifying the start and end (exclusive) sample indices to include
            in the output. If None (default), all the samples are included.

        Returns
        -------
        npt.NDArray[np.float32]
            A 1D array containing the mean audio data for the specified sample range.
        """
        pass

    @property
    def fname(self) -> str:
        """Return full path to the audio file that is being read."""
        return self._fname

    @property
    @abstractmethod
    def sampling_rate(self) -> int:
        """Return the nominal sampling rate of the audio."""
        pass

    @property
    @abstractmethod
    def n_channels(self) -> int:
        """Return the number of channels in the audio."""
        pass

    @property
    @abstractmethod
    def bit_depth(self) -> int:
        """Return the bit depth of the audio."""
        pass

    @property
    @abstractmethod
    def duration(self) -> float:
        """Return the duration of the audio in seconds."""
        pass

    @property
    @abstractmethod
    def n_samples(self) -> int:
        """Return the number of samples (per channel) in the audio."""
        pass

    def print_stats(self) -> None:
        """Print basic statistics about the audio file."""
        print(f"Stats for audio: {self.fname}")
        print(f"  - Number of channels: {self.n_channels}")
        print(f"  - Sampling rate: {self.sampling_rate} Hz")
        print(f"  - Bit depth: {self.bit_depth} bits")
        print(f"  - Duration: {self.duration:.2f} seconds")
        print(f"  - Number of samples per channel: {self.n_samples}")


class AudioFileHelsinkiVideoMEG(AudioFile):
    """Read an audio file in the Helsinki videoMEG project format.

    In addition to the properties of AudioFile interface, the following
    attributes are available:
        buffer_timestamps_ms  - buffers' timestamps (unix time in milliseconds)
        raw_audio             - raw audio data
        format_string         - format string for the audio data
        buffer_size           - buffer size (bytes)

    To access the unpacked audio data ((n_channels, n_samples) numpy array) and its
    timestamps, call `unpack_audio()` method and then use the getter methods.
    Timestamps that can be used for synchronization are available via
    `get_audio_timestamps()` method.

    Parameters
    ----------
    fname : str
        Full path to the audio file.
    magic_str : str, optional
        Magic string that should be at the beginning of video file.
        Default is "HELSINKI_VIDEO_MEG_PROJECT_AUDIO_FILE".
    """

    def __init__(
        self, fname: str, magic_str: str = "HELSINKI_VIDEO_MEG_PROJECT_AUDIO_FILE"
    ) -> None:
        super().__init__(fname)
        # Open the file to parse metadata and read the audio bytes into memory.
        with open(fname, "rb") as data_file:
            # Check the magic string
            if not data_file.read(len(magic_str)) == magic_str.encode("utf8"):
                raise ValueError(
                    f"File {fname} does not start with the expected "
                    f"magic string: {magic_str}."
                )
            self.ver = struct.unpack("I", data_file.read(4))[0]
            if self.ver != 0:
                # Can only read version 0 for the time being
                raise UnknownVersionError()

            self._sampling_rate, self._n_channels = struct.unpack(
                "II", data_file.read(8)
            )
            self.format_string = data_file.read(2).decode("ascii")
            self._bit_depth = self._get_bit_depth(self.format_string)

            # Get the size of the data part in the file.
            begin_data = data_file.tell()
            data_file.seek(0, 2)
            end_data = data_file.tell()
            data_file.seek(begin_data, 0)

            ts, self.buffer_size, total_sz = read_attrib(data_file, self.ver)
            data_file.seek(begin_data, 0)

            assert (end_data - begin_data) % total_sz == 0

            n_chunks = (end_data - begin_data) // total_sz
            self.raw_audio = bytearray(n_chunks * self.buffer_size)
            self.buffer_timestamps_ms = np.zeros(n_chunks)

            for i in range(n_chunks):
                ts, sz, cur_total_sz = read_attrib(data_file, self.ver)
                assert cur_total_sz == total_sz
                self.raw_audio[self.buffer_size * i : self.buffer_size * (i + 1)] = (
                    data_file.read(sz)
                )
                self.buffer_timestamps_ms[i] = ts
        # close the file

        # Initialize the attributes for unpacked audio data as None.
        # If user tries to access these without explicitly calling unpack_audio(),
        # it will be done automatically by the getter methods.

        # (n_channels, n_samples)
        self._unpacked_audio: npt.NDArray[np.float32] | None = None
        self._unpacked_mean_audio: npt.NDArray[np.float32] | None = None  # (n_samples,)
        self._audio_timestamps_ms: npt.NDArray[np.float64] | None = None  # (n_samples,)

    def get_audio_all_channels(
        self, sample_range: tuple[int, int] | None = None
    ) -> npt.NDArray[np.float32]:
        """Get audio data for all channels in the specified sample range.

        Triggers unpacking of audio data if it has not been done yet.

        Parameters
        ----------
        sample_range : tuple[int, int] | None
            A tuple specifying the start and end (exclusive) sample indices to include
            in the output. If None (default), all the samples are included.

        Returns
        -------
        npt.NDArray[np.float32]
            A 2D array of shape (n_channes, n_samples) containing the audio data.
        """
        self._ensure_unpacked_audio()
        assert self._unpacked_audio is not None, (
            "Audio data should be unpacked after calling _ensure_unpacked_audio."
        )
        if sample_range is None:
            return self._unpacked_audio[:, :]

        start_sample, end_sample = sample_range
        if start_sample < 0 or end_sample > self._unpacked_audio.shape[1]:
            raise ValueError("Sample range is out of bounds.")

        return self._unpacked_audio[:, start_sample:end_sample]

    def get_audio_mean(
        self, sample_range: tuple[int, int] | None = None
    ) -> npt.NDArray[np.float32]:
        """Get mean audio data across channels in the specified sample range.

        Triggers unpacking of audio data if it has not been done yet.

        Parameters
        ----------
        sample_range : tuple[int, int] | None
            A tuple specifying the start and end (exclusive) sample indices to include
            in the output. If None (default), all the samples are included.

        Returns
        -------
        npt.NDArray[np.float32]
            A 1D array containing the mean audio data for the specified sample range.
        """
        self._ensure_unpacked_audio()
        assert self._unpacked_mean_audio is not None, (
            "Mean audio data should be available after calling _ensure_unpacked_audio."
        )
        if sample_range is None:
            return self._unpacked_mean_audio[:]

        start_sample, end_sample = sample_range
        if start_sample < 0 or end_sample > self._unpacked_mean_audio.shape[0]:
            raise ValueError("Sample range is out of bounds.")

        return self._unpacked_mean_audio[start_sample:end_sample]

    def get_audio_timestamps(self) -> npt.NDArray[np.float64]:
        """Get timestamps for all audio samples.

        Triggers unpacking of audio data if it has not been done yet.

        Returns
        -------
        npt.NDArray[np.float64]
            A 1D array containing timestamps for all audio samples.
        """
        self._ensure_unpacked_audio()
        assert self._audio_timestamps_ms is not None, (
            "Audio timestamps should be available after calling _ensure_unpacked_audio."
        )
        return self._audio_timestamps_ms

    def unpack_audio(self) -> None:
        """Unpack the raw byte audio data and compute timestamps for all samples.

        This method is automatically called by the getter methods that require unpacked
        audio data. However, as this method is heavy on memory and time consumption,
        it is recommended to once call it manually before using the getters to avoid
        surprisingly heavy get operations.

        NOTE: this method consumes a lot of memory!
        """
        # ------------------------------------------------------------------
        # Compute timestamps for all the audio samples
        #
        logger.info("Unpacking audio data, this may take a while...")
        bytes_per_sample = struct.calcsize(self.format_string)
        n_chunks = len(self.buffer_timestamps_ms)
        samp_per_buf = self.buffer_size // (self._n_channels * bytes_per_sample)
        nsamp = samp_per_buf * n_chunks
        samps = np.arange(samp_per_buf - 1, nsamp, samp_per_buf)

        errs = -np.ones(n_chunks)
        audio_ts = -np.ones(nsamp)

        # split the data into segments for piecewise linear regression
        split_indx = list(range(0, nsamp, _REGR_SEGM_LENGTH * self._sampling_rate))
        split_indx[-1] = (
            nsamp  # the last segment might be up to twice as long as the others
        )

        for i in range(len(split_indx) - 1):
            sel_indx = np.where(
                (samps >= split_indx[i]) & (samps < split_indx[i + 1])
            )  # select one segment
            p = np.polyfit(
                samps[sel_indx], self.buffer_timestamps_ms[sel_indx], 1
            )  # compute the regression coefficients
            errs[sel_indx] = np.abs(
                np.polyval(p, samps[sel_indx]) - self.buffer_timestamps_ms[sel_indx]
            )  # compute the regression error
            audio_ts[split_indx[i] : split_indx[i + 1]] = np.polyval(
                p, np.arange(split_indx[i], split_indx[i + 1])
            )  # compute the timestamps with regression

        assert audio_ts.min() >= 0  # make sure audio_ts was completely filled
        assert errs.min() >= 0  # make sure errs was completely filled
        logger.info(
            f"Audio regression fit errors (abs): mean {errs.mean():.3f}, median "
            f"{np.median(errs):.3f}, max {errs.max():.3f}"
        )

        # ------------------------------------------------------------------
        # Parse the raw audio data
        #
        audio = np.zeros((self._n_channels, nsamp), dtype=np.float32)

        # NOTE: assuming the raw audio is interleaved
        for i in range(0, nsamp * self._n_channels):
            (samp_val,) = struct.unpack(
                self.format_string,
                self.raw_audio[i * bytes_per_sample : (i + 1) * bytes_per_sample],
            )
            audio[i % self._n_channels, i // self._n_channels] = samp_val

        self._unpacked_audio = audio
        self._unpacked_mean_audio = audio.mean(axis=0)
        self._audio_timestamps_ms = audio_ts

    def print_stats(self) -> None:
        """Print basic statistics about the audio file."""
        # Overrides the base class method to ensure audio is unpacked first.
        self._ensure_unpacked_audio()
        return super().print_stats()

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    @property
    def n_channels(self) -> int:
        return self._n_channels

    @property
    def bit_depth(self) -> int:
        return self._bit_depth

    @property
    def n_samples(self) -> int:
        """Return the number of samples (per channel) in the unpacked audio.

        Triggers unpacking of audio data if it has not been done yet.
        """
        self._ensure_unpacked_audio()
        assert self._unpacked_audio is not None, (
            "Audio data should be unpacked after calling _ensure_unpacked_audio."
        )
        return self._unpacked_audio.shape[1]

    @property
    def duration(self) -> float:
        """Return the duration of the audio in seconds.

        Duration is calculated as the number of samples divided by the sampling rate.
        Triggers unpacking of audio data if it has not been done yet.
        """
        return self.n_samples / self.sampling_rate

    def _get_bit_depth(self, format_string: str) -> int:
        """Get the bit depth from the format string."""
        # Dictionary mapping format characters to bit depths
        bit_depth_map = {
            "b": 8,  # signed char
            "B": 8,  # unsigned char
            "h": 16,  # short
            "H": 16,  # unsigned short
            "i": 32,  # int
            "I": 32,  # unsigned int
            "l": 32,  # long
            "L": 32,  # unsigned long
            "q": 64,  # long long
            "Q": 64,  # unsigned long long
            "f": 32,  # float
            "d": 64,  # double
        }
        # Extract the format character, ignoring endianness indicators
        bit_depth_char = format_string[-1]

        if bit_depth_char not in bit_depth_map:
            raise ValueError(
                f"Unsupported bit depth character: {bit_depth_char} in format "
                f"string {format_string}"
            )
        return bit_depth_map[bit_depth_char]

    def _ensure_unpacked_audio(self) -> None:
        """Ensure that the audio data is unpacked."""
        if self._unpacked_audio is None:
            logger.warning(
                "Unpacked audio data is not available. "
                "Calling unpack_audio() to unpack the audio data. "
                "Consider calling unpack_audio() manually before using the getters."
            )
            self.unpack_audio()
