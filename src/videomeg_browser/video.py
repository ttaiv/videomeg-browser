"""Contains VideoFile interface and its implementations for reading video files."""

# License: BSD-3-Clause
# Copyright (c) 2014 BioMag Laboratory, Helsinki University Central Hospital
# Copyright (c) 2025 Aalto University

import logging
import struct
from abc import ABC, abstractmethod

import cv2
import imageio.v3 as iio
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class VideoFile(ABC):
    """Container that holds a video file and provides method to read frames from it."""

    @abstractmethod
    def __del__(self) -> None:
        """Ensure the video file is released when the object is deleted."""
        pass

    @abstractmethod
    def __enter__(self) -> "VideoFile":
        """Enter the runtime context with opened video file."""
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the runtime context and release the video file."""
        pass

    @abstractmethod
    def get_frame_at(self, frame_idx: int) -> npt.NDArray[np.uint8] | None:
        """Read a specific frame from the video file.

        Parameters
        ----------
        frame_idx : int
            Index of the frame to read.

        Returns
        -------
        npt.NDArray[np.uint8] | None
            The frame as a NumPy array of shape (height, width, 3) or None if the frame
            cannot be read. The color format is RGB and the frame is in row-major order.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release the video file."""
        pass

    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Return the number of frames in the video file."""
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        """Return the frames per second of the video file."""
        pass

    @property
    @abstractmethod
    def frame_width(self) -> int:
        """Return the width of the video frames in pixels."""
        pass

    @property
    @abstractmethod
    def frame_height(self) -> int:
        """Return the height of the video frames in pixels."""
        pass

    @property
    @abstractmethod
    def fname(self) -> str:
        """Return the full path to the video file."""
        pass


class VideoFileCV2(VideoFile):
    """Container that holds a video file and provides methods to read frames from it."""

    def __init__(self, fname: str) -> None:
        self._fname = fname
        # Capture the video file for processing
        self._cap = cv2.VideoCapture(fname)
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video file: {fname}")

        # Matches to cv2.CAP_PROP_POS_FRAMES and tells the index of the next
        # frame to be read
        self._next_frame_idx = 0

    def __del__(self) -> None:
        """Ensure the video capture object is released when the object is deleted."""
        self.close()

    def __enter__(self) -> "VideoFileCV2":
        """Enter the runtime context for the video file."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the runtime context and release the video file."""
        self.close()

    def close(self) -> None:
        """Release the video capture object."""
        if self._cap.isOpened():
            self._cap.release()
        else:
            logger.debug(
                "Trying to release an already released video capture, ignoring."
            )

    def get_frame_at(self, frame_idx: int):
        """Read a specific frame from the video file.

        Parameters
        ----------
        frame_idx : int
            Index of the frame to read.

        Returns
        -------
        npt.NDArray[np.uint8] | None
            The frame as a NumPy array of shape (height, width, 3) or None if the frame
            cannot be read. The color format is RGB and the frame is in row-major order.
        """
        if not self._cap.isOpened():
            raise ValueError("Trying to read from a closed video file.")

        if frame_idx < 0 or frame_idx >= self.frame_count:
            logger.debug(f"Frame index out of bounds: {frame_idx}, returning None.")
            return None

        # Only alter the next frame to be read if necessary.
        # This increases performance when reading frames sequentially.
        if frame_idx != self._next_frame_idx:
            self._set_next_frame(frame_idx)

        return self._read_next_frame()

    @property
    def frame_count(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def frame_width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def frame_height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fname(self) -> str:
        return self._fname

    def _read_next_frame(self) -> npt.NDArray[np.uint8] | None:
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

        return frame.astype(np.uint8, copy=False)

    def _set_next_frame(self, frame_idx: int) -> None:
        """Set the next frame to be read from the video file."""
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise ValueError(f"Frame index out of bounds: {frame_idx}")

        self._next_frame_idx = frame_idx
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)


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


class VideoFileHelsinkiVideoMEG(VideoFile):
    """Video file reader for video files in Helsinki VideoMEG project format.

    Frame timestamps in milliseconds are stored in the `timestamps_ms` attribute,
    individual frames can be accessed using the `get_frame_at` method.

    Parameters
    ----------
    fname : str
        Full path to the video file to be read.
    """

    def __init__(self, fname) -> None:
        self._file_name = fname
        self._file = open(fname, "rb")
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

        # Get the file size.
        begin_data = self._file.tell()
        self._file.seek(0, 2)
        end_data = self._file.tell()
        self._file.seek(begin_data, 0)

        # For each frame, store timestamp and pointer to the frame data on disk.
        timestamps_list = []
        self._frame_ptrs = []  # List of tuples (offset, size) for each frame

        while self._file.tell() < end_data:  # we did not reach end of file
            ts, block_id, sz, total_sz = _read_attrib(self._file, self.ver)
            assert ts != -1
            timestamps_list.append(ts)
            self._frame_ptrs.append((self._file.tell(), sz))
            assert self._file.tell() + sz <= end_data
            self._file.seek(sz, 1)

        # Convert timestamps to numpy array
        self.timestamps_ms = np.array(timestamps_list, dtype=np.float64)
        self._nframes = len(timestamps_list)

        # Use first frame to determine width and height
        first_frame = self.get_frame_at(0)
        if first_frame is None:
            raise ValueError("Could not read the first frame of the video.")
        self._frame_width = first_frame.shape[1]
        self._frame_height = first_frame.shape[0]
        self._fps = self._estimate_fps(estimate_with="mean")

    def __del__(self) -> None:
        """Ensure the video file is closed when the object is deleted."""
        self.close()

    def __enter__(self) -> "VideoFileHelsinkiVideoMEG":
        """Enter the runtime context with opened video file."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the runtime context and close the video file."""
        self.close()

    def close(self) -> None:
        """Close the video file."""
        if not hasattr(self, "_file"):
            # The file opening probably failed during initialization of the object.
            logger.debug(
                "Trying to close a video file that was never opened, ignoring."
            )
        elif self._file.closed:
            logger.debug("Trying to close an already closed video file, ignoring.")
        else:
            self._file.close()

    def get_frame_at(self, frame_idx: int) -> npt.NDArray[np.uint8] | None:
        """Read a specific frame from the video file.

        Parameters
        ----------
        frame_idx : int
            Index of the frame to read.

        Returns
        -------
        npt.NDArray[np.uint8] | None
            The frame as a NumPy array of shape (height, width, 3) or None if the frame
            cannot be read. The color format is RGB and the frame is in row-major order.
        """
        if self._file.closed:
            raise ValueError("Trying to read from a closed video file.")
        if frame_idx < 0 or frame_idx >= self._nframes:
            logger.debug(f"Frame index out of bounds: {frame_idx}, returning None.")
            return None

        offset, sz = self._frame_ptrs[frame_idx]
        self._file.seek(offset)
        frame_bytes = self._file.read(sz)

        return iio.imread(frame_bytes)

    @property
    def frame_count(self) -> int:
        return self._nframes

    @property
    def fps(self) -> float:
        """Return the ESTIMATED frames per second of the video file."""
        return self._fps

    @property
    def frame_width(self) -> int:
        return self._frame_width

    @property
    def frame_height(self) -> int:
        return self._frame_height

    @property
    def fname(self) -> str:
        return self._file_name

    def _estimate_fps(self, estimate_with: str = "mean") -> float:
        """Estimate frames per second (FPS) based on timestamps."""
        if self._nframes < 2:
            return 0

        ts_in_seconds = self.timestamps_ms / 1000
        time_diff = np.diff(ts_in_seconds)

        if estimate_with == "mean":
            avg_time_diff = np.mean(time_diff)
        elif estimate_with == "median":
            avg_time_diff = np.median(time_diff)
        else:
            raise ValueError(f"Unknown estimation method: {estimate_with}")

        if avg_time_diff <= 0:
            raise ValueError(
                f"Average time difference is non-positive: {avg_time_diff}. "
                "Cannot estimate FPS."
            )

        return float(1 / avg_time_diff)


class VideoFileImageIO(VideoFile):
    """Reads standard video files (at least .mp4) using imageio.v3. with pyav plugin.

    NOTE: This seems to be slower than the implementation using OpenCV (cv2).
    Especially random access to frames is slower. This problem might be mitigated
    by reading part of the video file into memory at once. Furthermore, for some reason,
    reading the last frame of the video causes StopIteration exception.

    Parameters
    ----------
    fname : str
        Full path to the video file to be read.
    """

    def __init__(self, fname: str) -> None:
        self._fname = fname
        self._file = iio.imopen(fname, "r", plugin="pyav")

        video_properties = self._file.properties()
        n_frames = video_properties.n_images
        if n_frames is None:
            raise ValueError(
                "Could not determine the number of frames in the video file."
            )
        self._n_frames = n_frames
        assert self._n_frames == video_properties.shape[0], (
            "n_images in the properties should match the shape[0]."
        )
        self._frame_height = video_properties.shape[1]
        self._frame_width = video_properties.shape[2]

        metadata = self._file.metadata()
        if "fps" in metadata:
            self._fps = metadata["fps"]
        else:
            raise ValueError(
                "Could not determine the frames per second (FPS) of the video file."
            )

    def __del__(self) -> None:
        """Ensure the video file is closed when the object is deleted."""
        self.close()

    def __enter__(self) -> "VideoFileImageIO":
        """Enter the runtime context with opened video file."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the runtime context and close the video file."""
        self.close()

    def close(self) -> None:
        """Close the video file."""
        if not hasattr(self, "_file"):
            # The file opening probably failed during initialization of the object.
            logger.debug(
                "Trying to close a video file that was never opened, ignoring."
            )
        else:
            self._file.close()

    def get_frame_at(self, frame_idx: int) -> npt.NDArray[np.uint8] | None:
        """Read a specific frame from the video file.

        Parameters
        ----------
        frame_idx : int
            Index of the frame to read.

        Returns
        -------
        npt.NDArray[np.uint8] | None
            The frame as a NumPy array of shape (height, width, 3) or None if the frame
            cannot be read. The color format is RGB and the frame is in row-major order.
        """
        if frame_idx < 0 or frame_idx >= self._n_frames:
            logger.debug(f"Frame index out of bounds: {frame_idx}, returning None.")
            return None

        return self._file.read(index=frame_idx)

    @property
    def frame_count(self) -> int:
        return self._n_frames

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_width(self) -> int:
        return self._frame_width

    @property
    def frame_height(self) -> int:
        return self._frame_height

    @property
    def fname(self) -> str:
        return self._fname
