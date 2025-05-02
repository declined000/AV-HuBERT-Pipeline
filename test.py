from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.video.VideoClip import ColorClip
import numpy as np
import os

DURATION = 10  # seconds
DUMMY_DIR = "C:/github/rw/AV-HuBERT-S2S/dummy_files"
os.makedirs(DUMMY_DIR, exist_ok=True)

dummy_audio_path = os.path.join(DUMMY_DIR, "dummy_audio_10s.wav")
dummy_video_path = os.path.join(DUMMY_DIR, "dummy_video_10s.mp4")

# 🎵 Create silent audio (mono = 1 channel)
sample_rate = 16000
audio_array = np.zeros((DURATION * sample_rate, 1))  # shape: (samples, channels)
audio_clip = AudioArrayClip(audio_array, fps=sample_rate)
audio_clip.write_audiofile(dummy_audio_path, fps=sample_rate)

# 🎬 Create black video
w, h = 160, 120
black_clip = ColorClip(size=(w, h), color=(0, 0, 0), duration=DURATION)
black_clip.write_videofile(dummy_video_path, fps=25, codec="libx264", audio=False)
