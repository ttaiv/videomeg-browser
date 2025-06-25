"""Fake implementations of classes for testing purposes."""

import numpy as np
import numpy.typing as npt

from videomeg_browser.video import VideoFile


class FakeVideoFile(VideoFile):
    """Video file created by providing a numpy array of frames.

    Parameters
    ----------
    frames : npt.NDArray[np.uint8]
        A numpy array of shape (n_frames, height, width, channels) containing
        the video frames.
    fps : float
        The frames per second of the video.
    fname : str, optional
        The name of the video file, by default "fake_video.dat"
    """

    def __init__(
        self, frames: npt.NDArray[np.uint8], fps: float, fname: str = "fake_video.dat"
    ) -> None:
        self._fname = fname
        self._frames = frames
        self._fps = fps
        self._closed = False

    def __del__(self) -> None:
        """Delete the video file."""
        # No actual file to delete, so nothing to do here
        self.close()

    def __enter__(self) -> "FakeVideoFile":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Exit the context manager."""
        self.close()

    def get_frame_at(self, frame_idx: int) -> npt.NDArray[np.uint8] | None:
        """Get the frame at the specified index."""
        if self._closed:
            raise ValueError("Video file is closed.")
        if frame_idx < 0 or frame_idx >= self._frames.shape[0]:
            return None
        return self._frames[frame_idx]

    def frame_idx_to_ms(self, frame_idx: int) -> float:
        """Convert a frame index to milliseconds."""
        return (frame_idx / self._fps) * 1000

    def close(self) -> None:
        """Close the video file."""
        self._closed = True

    @property
    def frame_count(self) -> int:
        """Get the number of frames in the video."""
        return self._frames.shape[0]

    @property
    def fps(self) -> float:
        """Get the frames per second of the video."""
        return self._fps

    @property
    def frame_width(self) -> int:
        """Get the width of the video frames."""
        return self._frames.shape[2]

    @property
    def frame_height(self) -> int:
        """Get the height of the video frames."""
        return self._frames.shape[1]

    @property
    def fname(self) -> str:
        """Get the name of the video file."""
        return self._fname
