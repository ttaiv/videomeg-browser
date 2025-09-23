import unittest.mock as mock

import numpy as np
import numpy.typing as npt
import pytest

from videomeg_browser.browsers.video_browser import FrameRateTracker


class TestFrameRateTracker:
    """Test the frame rate tracking helper class."""

    def test_first_two_frames(self) -> None:
        """Test that the tracker functions correctly with frames."""
        tracker = FrameRateTracker(max_intervals_to_average=3)
        self._test_new_tracker_with_two_frames(tracker)

    def test_reset(self) -> None:
        """Test resetting the tracker."""
        tracker = FrameRateTracker(max_intervals_to_average=5)
        # Store frame intervals to the tracker.
        self._test_new_tracker_with_two_frames(tracker)
        # Test that the tracker has stored frame intervals.
        assert tracker.get_current_frame_rate() > 0, (
            "Frame rate should be positive after two frame notifications"
        )
        tracker.reset()
        # Now the tests for new tracker should pass again.
        self._test_new_tracker_with_two_frames(tracker)

    @pytest.mark.parametrize(
        "simulated_fps, n_simulated_frames", [(30, 60), (60, 60), (15, 120), (1, 5)]
    )
    def test_with_constant_interval(
        self, simulated_fps: float, n_simulated_frames: int
    ) -> None:
        """Test with perfectly constant interval between frame notifications."""
        # Create timestamps for simulated frames
        simulated_frame_interval = 1 / simulated_fps
        frame_times = np.arange(n_simulated_frames) * simulated_frame_interval

        # Run the test with different max averaging windows.
        # The result should not change as the interval between frames is constant.
        for n_intervals in range(1, 55, 5):
            tracker = FrameRateTracker(max_intervals_to_average=n_intervals)
            self._notify_tracker_with_mock_frame_times(tracker, frame_times)

            tracker_fps = tracker.get_current_frame_rate()
            assert tracker_fps == pytest.approx(simulated_fps)

    @pytest.mark.parametrize(
        "simulated_fps, n_simulated_frames, noise_scale",
        [(30, 60, 3), (60, 20, 5), (10, 100, 1.5)],
    )
    def test_with_variable_interval(
        self, simulated_fps: float, n_simulated_frames: int, noise_scale: float
    ) -> None:
        """Test with variable interval between frame notifications."""
        simulated_frame_interval = 1 / simulated_fps
        # Create frame times as with constant intervals but add some noise.
        frame_times = np.arange(n_simulated_frames) * simulated_frame_interval + (
            np.random.default_rng(seed=42).normal(
                scale=noise_scale, size=n_simulated_frames
            )
        )
        # Sort to make sure that frames times are ascending.
        frame_times = np.sort(frame_times)
        # Get frame intervals to manually calculate the correct fps.
        frame_intervals = np.diff(frame_times)

        # Run the test with different averaging windows.
        for n_intervals in range(1, n_simulated_frames, 5):
            # Manually calculate the correct fps based on averaging window.
            correct_fps = 1 / np.mean(frame_intervals[-n_intervals:])

            tracker = FrameRateTracker(max_intervals_to_average=n_intervals)
            self._notify_tracker_with_mock_frame_times(tracker, frame_times)

            tracker_fps = tracker.get_current_frame_rate()
            assert tracker_fps == pytest.approx(correct_fps)

    def test_with_invalid_input(self) -> None:
        """Test that the tracker raises ValueError with invalid input."""
        with pytest.raises(ValueError):
            _ = FrameRateTracker(max_intervals_to_average=0)
        with pytest.raises(ValueError):
            _ = FrameRateTracker(max_intervals_to_average=-1)

    def _test_new_tracker_with_two_frames(self, tracker: FrameRateTracker) -> None:
        """Test that new/resetted tracker functions correctly."""
        assert tracker.get_current_frame_rate() == 0.0, (
            "Frame rate should be zero when tracker has not been notified."
        )
        tracker.notify_new_frame()
        assert tracker.get_current_frame_rate() == 0.0, (
            "Frame rate should be zero when tracker is only notified once. "
            "(cannot calculate frame interval with just one frame)"
        )
        tracker.notify_new_frame()
        assert tracker.get_current_frame_rate() > 0, (
            "Frame rate should be positive after two frame notification."
        )
        assert tracker.get_current_frame_rate() > 0, (
            "Frame rate should be positive after two frame notification."
        )

    def _notify_tracker_with_mock_frame_times(
        self, tracker: FrameRateTracker, frame_times: npt.NDArray[np.floating]
    ) -> None:
        # Mock the time.perf_counter used to acquire times when frame arrives.
        with mock.patch("time.perf_counter", side_effect=frame_times):
            for _ in range(len(frame_times)):
                tracker.notify_new_frame()
