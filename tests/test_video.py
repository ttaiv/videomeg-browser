"""Tests for video file handling."""

import os

import numpy as np
import pytest

from videomeg_browser.video import VideoFileHelsinkiVideoMEG

# Tests with a specific video file in Helsinki videoMEG format.
VIDEO_PATH = (
    "/u/69/taivait1/unix/video_meg_testing/Subject_2_Luna/Video_MEG/"
    "animal_meg_subject_2_240614.video.dat"
)
VIDEO_FPS = 30.0  # Frames per second for the test video


@pytest.mark.skipif(
    not os.path.exists(VIDEO_PATH), reason="Test .video.dat file not available."
)
class TestVideoFileHelsinkiVideoMEG:
    """Tests for Helsinki videoMEG project video file handling."""

    def test_open_close(self):
        """Test opening and closing the video file."""
        video = VideoFileHelsinkiVideoMEG(VIDEO_PATH)
        # Check that we can read frames after opening
        assert video.frame_count > 0
        assert video.get_frame_at(video.frame_count // 2) is not None

        # Close the video file and check that frames cannot be read anymore
        video.close()
        assert video.frame_count > 0  # Frame count should still be valid
        with pytest.raises(ValueError):
            video.get_frame_at(video.frame_count // 2)

        video.close()  # Ensure closing does not raise an error if already closed

    def test_open_close_context_manager(self):
        """Test opening and closing the video file using context manager."""
        with VideoFileHelsinkiVideoMEG(VIDEO_PATH) as video:
            # Check that we can read frames while the context is open
            assert video.frame_count > 0
            assert video.get_frame_at(video.frame_count // 2) is not None

        # After exiting the context, the file should be closed
        assert video.frame_count > 0  # Frame count should still be valid
        with pytest.raises(ValueError):
            video.get_frame_at(video.frame_count // 2)

        video.close()  # Ensure closing does not raise an error if already closed

    def test_properties(self):
        """Test video file properties after initialization."""
        with VideoFileHelsinkiVideoMEG(VIDEO_PATH) as video:
            assert video.fname == VIDEO_PATH
            # Using approx as Helsinki videoMEG video files do not have exact FPS
            # and it is approximated using time between frames.
            assert video.fps == pytest.approx(VIDEO_FPS, rel=1e-3)  # within 0.1%
            assert video.frame_count > 0
            assert video.frame_height > 0
            assert video.frame_width > 0
            assert len(video.timestamps_ms) == video.frame_count

    def test_get_frame_at(self):
        """Test getting frames at specific indices."""
        with VideoFileHelsinkiVideoMEG(VIDEO_PATH) as video:
            # Test that all frames can be accessed and have correct dimensions
            test_frame_indices = np.linspace(
                0, video.frame_count - 1, num=100, dtype=int
            )
            for frame_idx in test_frame_indices:
                test_frame = video.get_frame_at(frame_idx)
                assert test_frame is not None
                assert test_frame.shape[0] == video.frame_height
                assert test_frame.shape[1] == video.frame_width
                assert test_frame.shape[2] == 3  # RGB frame
            # Test that out-of-bounds indices return None
            too_large_frame = video.get_frame_at(video.frame_count)
            assert too_large_frame is None
            too_small_frame = video.get_frame_at(-1)
            assert too_small_frame is None

    def test_invalid_file_path(self, tmp_path):
        """Test that providing an invalid file path raises an error."""
        invalid_path = tmp_path / "invalid.video.dat"
        invalid_path.write_bytes(b"NOT_A_VIDEO_FILE")
        with pytest.raises(FileNotFoundError):
            _ = VideoFileHelsinkiVideoMEG("invalid_path.video.dat")
