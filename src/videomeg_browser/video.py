import struct
from abc import ABC, abstractmethod

import cv2
import numpy as np
from cv2.typing import MatLike


class VideoFile(ABC):
    """Container that holds a video file and provides method to read frames from it."""

    @abstractmethod
    def get_frame_at(self, frame_idx: int) -> MatLike | None:
        """Read a specific frame from the video file."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release the video file."""
        pass

    @abstractmethod
    def __del__(self) -> None:
        """Ensure the video file is released when the object is deleted."""
        pass

    @abstractmethod
    def __enter__(self) -> "VideoFile":
        """Enter the runtime context for the video file."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the runtime context and release the video file."""
        pass

    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Return the number of frames in the video file."""
        pass

    @property
    @abstractmethod
    def fps(self) -> int:
        """Return the frames per second of the video file."""
        pass

    @property
    @abstractmethod
    def frame_width(self) -> int:
        """Return the width of the video frames."""
        pass

    @property
    @abstractmethod
    def frame_height(self) -> int:
        """Return the height of the video frames."""
        pass


class VideoFileCV2(VideoFile):
    """Container that holds a video file and provides methods to read frames from it."""

    def __init__(self, fname: str) -> None:
        self.fname = fname
        # Capture the video file for processing
        self._cap = cv2.VideoCapture(fname)
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video file: {fname}")

        # Matches to cv2.CAP_PROP_POS_FRAMES and tells the index of the next
        # frame to be read
        self._next_frame_idx = 0

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FPS))

    @property
    def frame_width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _read_next_frame(self) -> MatLike | None:
        """Read the next frame from the video file."""
        if not self._cap.isOpened():
            raise ValueError("Trying to read from a closed video file.")

        ret, frame = self._cap.read()
        if not ret:
            # End of video?
            return None

        self._next_frame_idx += 1
        # Convert the frame from BGR to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    def _read_previous_frame(self) -> MatLike | None:
        """Read the frame before the last read frame from the video file."""
        if not self._cap.isOpened():
            raise ValueError("Trying to read from a closed video file.")

        if self._next_frame_idx < 2:
            print("Already at the first frame.")
            return None

        # Last read frame is one step back, the frame before that is two steps back
        self._set_next_frame(self._next_frame_idx - 2)
        return self._read_next_frame()

    def get_frame_at(self, frame_idx: int) -> MatLike | None:
        """Read a specific frame from the video file."""
        if not self._cap.isOpened():
            raise ValueError("Trying to read from a closed video file.")

        if frame_idx < 0 or frame_idx >= self.frame_count:
            print(f"Frame index out of bounds: {frame_idx}, returning None.")
            return None

        self._set_next_frame(frame_idx)
        return self._read_next_frame()

    def _set_next_frame(self, frame_idx: int) -> None:
        """Set the next frame to be read from the video file."""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise ValueError(f"Frame index out of bounds: {frame_idx}")

        self._next_frame_idx = frame_idx
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def close(self) -> None:
        """Release the video capture object."""
        if self._cap.isOpened():
            self._cap.release()

    def __del__(self) -> None:
        """Ensure the video capture object is released when the object is deleted."""
        self.close()

    def __enter__(self) -> "VideoFileCV2":
        """Enter the runtime context for the video file."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the runtime context and release the video file."""
        self.close()


# --- Code below is adapted from PyVideoMEG project ---


class UnknownVersionError(Exception):
    """Error due to unknown file version."""

    pass


def _read_attrib(data_file, ver):
    """
    Read data block attributes. If cannot read the attributes (EOF?), return
    -1 in ts
    """
    if ver == 1:
        attrib = data_file.read(12)
        if len(attrib) == 12:
            ts, sz = struct.unpack("QI", attrib)
        else:
            ts = -1
            sz = -1
        block_id = 0
        total_sz = sz + 12

    elif ver == 2 or ver == 3:
        attrib = data_file.read(20)
        if len(attrib) == 20:
            ts, block_id, sz = struct.unpack("QQI", attrib)
        else:
            ts = -1
            block_id = 0
            sz = -1
        total_sz = sz + 20

    else:
        raise UnknownVersionError()

    return ts, block_id, sz, total_sz


class VideoData:
    """
    To read a video file initialize VideoData object with file name. You can
    then get the frame times from the object's ts variable. To get individual
    frames use get_frame function.
    """

    def __init__(self, file_name):
        self._file = open(file_name, "rb")
        assert (
            self._file.read(len("ELEKTA_VIDEO_FILE")) == b"ELEKTA_VIDEO_FILE"
        )  # make sure the magic string is OK
        self.ver = struct.unpack("I", self._file.read(4))[0]

        if self.ver == 1 or self.ver == 2:
            self.site_id = -1
            self.is_sender = -1

        elif self.ver == 3:
            self.site_id, self.is_sender = struct.unpack("BB", self._file.read(2))

        else:
            raise UnknownVersionError()

        # get the file size
        begin_data = self._file.tell()
        self._file.seek(0, 2)
        end_data = self._file.tell()
        self._file.seek(begin_data, 0)

        self.ts = np.array([])
        self._frame_ptrs = []

        while self._file.tell() < end_data:  # we did not reach end of file
            ts, block_id, sz, total_sz = _read_attrib(self._file, self.ver)
            assert ts != -1
            self.ts = np.append(self.ts, ts)
            self._frame_ptrs.append((self._file.tell(), sz))
            assert self._file.tell() + sz <= end_data
            self._file.seek(sz, 1)

        self.nframes = self.ts.size

    def __del__(self):
        self._file.close()

    def get_frame(self, indx):
        """
        Return indx-th frame a jpg image in the memory.
        """
        offset, sz = self._frame_ptrs[indx]
        self._file.seek(offset)
        return self._file.read(sz)
