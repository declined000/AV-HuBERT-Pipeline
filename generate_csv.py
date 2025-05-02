import os
import csv
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import ColorClip

# Paths
LIPREAD_ROOT = "C:/github/rw/AV-HuBERT-S2S/GLips/lipread_files"
DUMMY_DIR = "C:/github/rw/AV-HuBERT-S2S/dummy_files"
OUTPUT_CSV = "C:/github/rw/AV-HuBERT-S2S/glips_filelist.csv"

VIDEO_EXT = ".mp4"
AUDIO_EXT = ".wav"

# Prepare directories
os.makedirs(DUMMY_DIR, exist_ok=True)

# Always create fresh CSV
if os.path.exists(OUTPUT_CSV):
    os.remove(OUTPUT_CSV)

def get_duration(path):
    try:
        clip = AudioFileClip(path)
        duration = clip.duration
        clip.close()
        return duration
    except Exception as e:
        print(f"❌ Could not get duration of {path}: {e}")
        return None

def create_dummy_audio(out_path, duration):
    os.system(
        f'ffmpeg -y -loglevel error -f lavfi -i anullsrc=r=16000:cl=mono -t {duration} -q:a 9 -acodec pcm_s16le "{out_path}"'
    )


import contextlib
import sys

def create_dummy_video(out_path, duration):
    try:
        w, h = 160, 120
        color_clip = ColorClip(size=(w, h), color=(0, 0, 0), duration=duration)

        with open(os.devnull, "w") as fnull:
            with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
                color_clip.write_videofile(out_path, fps=25, codec="libx264", audio=False)
    except Exception as e:
        print(f"❌ Failed to create dummy video at {out_path}: {e}")



with open(OUTPUT_CSV, mode="w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "video", "audio", "label", "dummy_video", "dummy_audio"])

    for label_folder in os.listdir(LIPREAD_ROOT):
        print(label_folder)
        test_path = os.path.join(LIPREAD_ROOT, label_folder, "test")
        if not os.path.isdir(test_path):
            continue

        for f in os.listdir(test_path):
            if not f.endswith(VIDEO_EXT):
                continue
            if "_lip_movement" in f:
                continue  # 🚫 Skip reprocessed files

            base = f.replace(VIDEO_EXT, "")
            video_path = os.path.join(test_path, base + VIDEO_EXT)
            audio_path = os.path.join(test_path, base + AUDIO_EXT)

            if not os.path.exists(video_path) or not os.path.exists(audio_path):
                print(f"⚠️ Skipping {base}: missing video or audio.")
                continue

            duration = get_duration(audio_path)
            if not duration:
                continue

            dummy_audio = os.path.join(DUMMY_DIR, base + "_dummy_audio.wav")
            dummy_video = os.path.join(DUMMY_DIR, base + "_dummy_video.mp4")

            create_dummy_audio(dummy_audio, duration)
            create_dummy_video(dummy_video, duration)

            writer.writerow([
                base,
                video_path.replace("\\", "/"),
                audio_path.replace("\\", "/"),
                label_folder,
                dummy_video.replace("\\", "/"),
                dummy_audio.replace("\\", "/")
            ])
            # Success messages suppressed

print(f"\n✅ CSV complete: {OUTPUT_CSV}")
