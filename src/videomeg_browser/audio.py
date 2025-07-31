"""Contains AudioFile interface and its implemmentations for reading audio files."""

# License: BSD-3-Clause
# Copyright (c) 2014 BioMag Laboratory, Helsinki University Central Hospital
# Copyright (c) 2025 Aalto University

import logging
import struct
from abc import ABC, abstractmethod

import numpy as np

from .helsinki_videomeg_file_utils import UnknownVersionError, read_attrib

logger = logging.getLogger(__name__)

_REGR_SEGM_LENGTH = 20  # seconds, should be integer


class AudioFile(ABC):
    """Handles reading audio files."""

    # TODO: Choose methods needed for this interface.


class AudioFileHelsinkiVideoMEG(AudioFile):
    """
    To read an audio file initialize AudioData object with file name and get
    the data from the object's variables:
        srate           - nominal sampling rate
        nchan           - number of channels
        ts              - buffers' timestamps
        raw_audio       - raw audio data
        format_string   - format string for the audio data
        buf_sz          - buffer size (bytes)
    """

    def __init__(self, file_name):
        data_file = open(file_name, "rb")
        assert (
            data_file.read(len("HELSINKI_VIDEO_MEG_PROJECT_AUDIO_FILE"))
            == b"HELSINKI_VIDEO_MEG_PROJECT_AUDIO_FILE"
        )  # make sure the magic string is OK
        self.ver = struct.unpack("I", data_file.read(4))[0]

        if self.ver != 0:
            # Can only read version 0 for the time being
            raise UnknownVersionError()

        self.srate, self.nchan = struct.unpack("II", data_file.read(8))
        self.format_string = data_file.read(2).decode("ascii")

        # get the size of the data part of the file
        begin_data = data_file.tell()
        data_file.seek(0, 2)
        end_data = data_file.tell()
        data_file.seek(begin_data, 0)

        ts, self.buf_sz, total_sz = read_attrib(data_file, self.ver)
        data_file.seek(begin_data, 0)

        assert (end_data - begin_data) % total_sz == 0

        n_chunks = (end_data - begin_data) // total_sz
        self.raw_audio = bytearray(n_chunks * self.buf_sz)
        self.ts = np.zeros(n_chunks)

        for i in range(n_chunks):
            ts, sz, cur_total_sz = read_attrib(data_file, self.ver)
            assert cur_total_sz == total_sz
            self.raw_audio[self.buf_sz * i : self.buf_sz * (i + 1)] = data_file.read(sz)
            self.ts[i] = ts

        data_file.close()

    def format_audio(self):
        """Return the formatted version or self.raw_audio.
        Return:
            audio     - nchan-by-nsamp matrix of the audio data
            audio_ts  - timestamps for all the audio samples

        Caution: this function consumes a lot of memory!
        """
        # ------------------------------------------------------------------
        # Compute timestamps for all the audio samples
        #
        bytes_per_sample = struct.calcsize(self.format_string)
        n_chunks = len(self.ts)
        samp_per_buf = self.buf_sz // (self.nchan * bytes_per_sample)
        nsamp = samp_per_buf * n_chunks
        samps = np.arange(samp_per_buf - 1, nsamp, samp_per_buf)

        errs = -np.ones(n_chunks)
        audio_ts = -np.ones(nsamp)

        # split the data into segments for piecewise linear regression
        split_indx = list(range(0, nsamp, _REGR_SEGM_LENGTH * self.srate))
        split_indx[-1] = (
            nsamp  # the last segment might be up to twice as long as the others
        )

        for i in range(len(split_indx) - 1):
            sel_indx = np.where(
                (samps >= split_indx[i]) & (samps < split_indx[i + 1])
            )  # select one segment
            p = np.polyfit(
                samps[sel_indx], self.ts[sel_indx], 1
            )  # compute the regression coefficients
            errs[sel_indx] = np.abs(
                np.polyval(p, samps[sel_indx]) - self.ts[sel_indx]
            )  # compute the regression error
            audio_ts[split_indx[i] : split_indx[i + 1]] = np.polyval(
                p, np.arange(split_indx[i], split_indx[i + 1])
            )  # compute the timestamps with regression

        assert audio_ts.min() >= 0  # make sure audio_ts was completely filled
        assert errs.min() >= 0  # make sure errs was completely filled
        print(
            "AudioData: regression fit errors (abs): mean %f, median %f, max %f"
            % (errs.mean(), np.median(errs), errs.max())
        )

        # ------------------------------------------------------------------
        # Parse the raw audio data
        #
        audio = np.zeros((self.nchan, nsamp))

        # NOTE: assuming the raw audio is interleaved
        for i in range(0, nsamp * self.nchan):
            (samp_val,) = struct.unpack(
                self.format_string,
                self.raw_audio[i * bytes_per_sample : (i + 1) * bytes_per_sample],
            )
            audio[i % self.nchan, i // self.nchan] = samp_val

        return audio, audio_ts
