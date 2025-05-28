"""Tests for video module. GENERATED WITH GITHUB COPILOT (AI)."""

import os

import cv2
import numpy as np
import pytest

from videomeg_browser.video import VideoFileCV2, VideoFileHelsinkiVideoMEG

VIDEO_MEG_PATH = (
    "/u/69/taivait1/unix/video_meg_testing/Subject_2_Luna/Video_MEG/"
    "animal_meg_subject_2_240614.video.dat"
)


@pytest.fixture
def sample_video(tmp_path):
    """Create a simple video file using OpenCV for testing."""
    video_path = tmp_path / "test.avi"
    fourcc = cv2.VideoWriter.fourcc("X", "V", "I", "D")
    out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (64, 48))
    for i in range(5):
        frame = np.full((48, 64, 3), i * 50, dtype=np.uint8)
        out.write(frame)
    out.release()
    return str(video_path)


def test_videofilecv2_properties_and_frame(sample_video):
    """Test VideoFileCV2 properties and frame reading."""
    with VideoFileCV2(sample_video) as vf:
        assert vf.frame_count == 5
        assert vf.frame_width == 64
        assert vf.frame_height == 48
        assert vf.fps > 0
        frame = vf.get_frame_at(0)
        assert frame is not None
        assert frame.shape == (48, 64, 3)
        # Test out-of-bounds
        assert vf.get_frame_at(100) is None
        ms = vf.frame_idx_to_ms(0)
        assert ms == 0.0
        ms2 = vf.frame_idx_to_ms(1)
        assert ms2 > 0


def test_videofilecv2_context_manager(sample_video):
    """Test VideoFileCV2 context manager and closed state."""
    vf = VideoFileCV2(sample_video)
    with vf as v:
        assert v.frame_count == 5
    # After context, should be closed
    with pytest.raises(ValueError):
        vf.get_frame_at(0)


def test_videofilecv2_invalid_file():
    """Test that VideoFileCV2 raises ValueError for invalid file."""
    with pytest.raises(ValueError):
        VideoFileCV2("nonexistent_file.avi")


def test_videofilecv2_frame_idx_to_ms_out_of_bounds(sample_video):
    """Test frame_idx_to_ms raises ValueError for out-of-bounds index."""
    with VideoFileCV2(sample_video) as vf:
        with pytest.raises(ValueError):
            vf.frame_idx_to_ms(-1)
        with pytest.raises(ValueError):
            vf.frame_idx_to_ms(100)


def test_videofilecv2_get_frame_at_closed(sample_video):
    """Test get_frame_at raises ValueError if file is closed."""
    vf = VideoFileCV2(sample_video)
    vf.close()
    with pytest.raises(ValueError):
        vf.get_frame_at(0)


def test_videofilecv2_multiple_frames(sample_video):
    """Test reading multiple frames in sequence."""
    with VideoFileCV2(sample_video) as vf:
        frames = [vf.get_frame_at(i) for i in range(vf.frame_count)]
        # Ensure all frames are not None before checking shape
        non_none_frames = [frame for frame in frames if frame is not None]
        assert len(non_none_frames) == vf.frame_count
        assert all(frame.shape == (48, 64, 3) for frame in non_none_frames)
        # Frames should differ in pixel values
        if len(non_none_frames) > 1:
            f0, f1 = non_none_frames[0], non_none_frames[1]
            assert not np.array_equal(f0, f1)


def test_videofilecv2_set_next_frame_and_read(sample_video):
    """Test _set_next_frame and _read_next_frame for correct frame access."""
    with VideoFileCV2(sample_video) as vf:
        vf._set_next_frame(2)
        frame = vf._read_next_frame()
        assert frame is not None
        assert frame.shape == (48, 64, 3)
        # After reading, _next_frame_idx should increment
        assert vf._next_frame_idx == 3


def test_videofilecv2_set_next_frame_out_of_bounds(sample_video):
    """Test _set_next_frame raises ValueError for out-of-bounds index."""
    with VideoFileCV2(sample_video) as vf:
        with pytest.raises(ValueError):
            vf._set_next_frame(-1)
        with pytest.raises(ValueError):
            vf._set_next_frame(100)


def test_videofilecv2_read_next_frame_closed(sample_video):
    """Test _read_next_frame raises ValueError if file is closed."""
    vf = VideoFileCV2(sample_video)
    vf.close()
    with pytest.raises(ValueError):
        vf._read_next_frame()


def test_videofilecv2_del_closes_file(sample_video):
    """Test __del__ releases the video file (no error on double close)."""
    vf = VideoFileCV2(sample_video)
    cap = vf._cap
    del vf
    # The cap should be released (isOpened returns False)
    assert not cap.isOpened()


def test_videofilecv2_context_manager_double_close(sample_video):
    """Test that double closing does not raise error."""
    vf = VideoFileCV2(sample_video)
    with vf:
        pass
    # Second close should not raise
    vf.close()


@pytest.mark.skipif(
    not os.path.exists(VIDEO_MEG_PATH), reason="Test .video.dat file not available."
)
def test_videofilehelsinkivideomeg_properties():
    """Test VideoFileHelsinkiVideoMEG properties and first frame."""
    with VideoFileHelsinkiVideoMEG(VIDEO_MEG_PATH) as vf:
        assert vf.frame_count > 0
        assert vf.frame_width > 0
        assert vf.frame_height > 0
        assert vf.fps > 0
        frame = vf.get_frame_at(0)
        assert frame is not None
        assert frame.shape[0] == vf.frame_height
        assert frame.shape[1] == vf.frame_width
        assert frame.shape[2] == 3
        # Test out-of-bounds
        assert vf.get_frame_at(vf.frame_count) is None
        ms = vf.frame_idx_to_ms(0)
        assert ms == 0.0
        if vf.frame_count > 1:
            ms2 = vf.frame_idx_to_ms(1)
            assert ms2 > 0


@pytest.mark.skipif(
    not os.path.exists(VIDEO_MEG_PATH), reason="Test .video.dat file not available."
)
def test_videofilehelsinkivideomeg_context_manager():
    """Test context manager and closed state for VideoFileHelsinkiVideoMEG."""
    vf = VideoFileHelsinkiVideoMEG(VIDEO_MEG_PATH)
    with vf as v:
        assert v.frame_count > 0
    # After context, should be closed
    with pytest.raises(ValueError):
        vf.get_frame_at(0)


@pytest.mark.skipif(
    not os.path.exists(VIDEO_MEG_PATH), reason="Test .video.dat file not available."
)
def test_videofilehelsinkivideomeg_frame_idx_to_ms_out_of_bounds():
    """Test frame_idx_to_ms raises ValueError for out-of-bounds index."""
    with VideoFileHelsinkiVideoMEG(VIDEO_MEG_PATH) as vf:
        with pytest.raises(ValueError):
            vf.frame_idx_to_ms(-1)
        with pytest.raises(ValueError):
            vf.frame_idx_to_ms(vf.frame_count)


@pytest.mark.skipif(
    not os.path.exists(VIDEO_MEG_PATH), reason="Test .video.dat file not available."
)
def test_videofilehelsinkivideomeg_get_frame_at_closed():
    """Test get_frame_at raises ValueError if file is closed."""
    vf = VideoFileHelsinkiVideoMEG(VIDEO_MEG_PATH)
    vf.close()
    with pytest.raises(ValueError):
        vf.get_frame_at(0)


@pytest.mark.skipif(
    not os.path.exists(VIDEO_MEG_PATH), reason="Test .video.dat file not available."
)
def test_videofilehelsinkivideomeg_multiple_frames():
    """Test reading multiple frames in sequence from VideoFileHelsinkiVideoMEG."""
    with VideoFileHelsinkiVideoMEG(VIDEO_MEG_PATH) as vf:
        frames = [vf.get_frame_at(i) for i in range(min(5, vf.frame_count))]
        non_none_frames = [frame for frame in frames if frame is not None]
        assert len(non_none_frames) == len(frames)
        assert all(
            frame.shape[0] == vf.frame_height and frame.shape[1] == vf.frame_width
            for frame in non_none_frames
        )
        if len(non_none_frames) > 1:
            f0, f1 = non_none_frames[0], non_none_frames[1]
            assert not np.array_equal(f0, f1)


@pytest.mark.skipif(
    not os.path.exists(VIDEO_MEG_PATH), reason="Test .video.dat file not available."
)
def test_videofilehelsinkivideomeg_del_closes_file():
    """Test __del__ releases the file (no error on double close)."""
    vf = VideoFileHelsinkiVideoMEG(VIDEO_MEG_PATH)
    f = vf._file
    del vf
    assert f.closed


@pytest.mark.skipif(
    not os.path.exists(VIDEO_MEG_PATH), reason="Test .video.dat file not available."
)
def test_videofilehelsinkivideomeg_context_manager_double_close():
    """Test that double closing does not raise error for VideoFileHelsinkiVideoMEG."""
    vf = VideoFileHelsinkiVideoMEG(VIDEO_MEG_PATH)
    with vf:
        pass
    vf.close()
