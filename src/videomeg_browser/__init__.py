"""MNE-Python extension for synchronized viewing of MEG/EEG, video, and audio."""

from .comp_tstamps import comp_tstamps
from .media.audio import AudioFileHelsinkiVideoMEG
from .media.video import VideoFileCV2, VideoFileHelsinkiVideoMEG
from .synced_raw_media_browser import (
    browse_raw_with_audio,
    browse_raw_with_video,
    browse_raw_with_video_and_audio,
)
from .timestamp_aligner import TimestampAligner

__all__ = [
    "browse_raw_with_video",
    "browse_raw_with_audio",
    "browse_raw_with_video_and_audio",
    "TimestampAligner",
    "comp_tstamps",
    "VideoFileHelsinkiVideoMEG",
    "VideoFileCV2",
    "AudioFileHelsinkiVideoMEG",
]
