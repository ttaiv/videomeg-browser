import logging
import os.path as op

import mne

from videomeg_browser.raw_video_sync import SyncedRawVideoBrowser, TimeIndexMapper
from videomeg_browser.video import VideoFileHelsinkiVideoMEG

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d %(message)s",
)
base_path = "/u/69/taivait1/unix/video_meg_testing/Subject_2_Luna"
# Create a video file object
video_file = VideoFileHelsinkiVideoMEG(
    op.join(base_path, "Video_MEG", "animal_meg_subject_2_240614.video.dat")
)

# Create a raw data object
raw = mne.io.read_raw_fif(
    op.join(base_path, "Raw", "animal_meg_subject_2_240614.fif"), preload=True
)

# Set up mapping between time points of raw data and video frame indices
# This is tailored for the Helsinki Video MEG data format
time_mapper = TimeIndexMapper(raw, raw_timing_ch="STI016", video=video_file)

browser = SyncedRawVideoBrowser(raw, video_file, time_mapper)
browser.show()
