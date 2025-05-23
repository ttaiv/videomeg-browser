import cv2
from cv2.typing import MatLike


class VideoFile:
    """Container that holds a video file and provides methods to read frames from it."""

    def __init__(self, fname: str) -> None:
        self.fname = fname
        # Capture the video file for processing
        self._cap = cv2.VideoCapture(fname)
        if not self._cap.isOpened():
            raise ValueError(f"Could not open video file: {fname}")

        # Store video properties
        self.frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Matches to cv2.CAP_PROP_POS_FRAMES and tells the index of the next
        # frame to be read
        self._next_frame_idx = 0

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
