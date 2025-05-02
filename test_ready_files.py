import os
import shutil
from pathlib import Path

# Paths
PROCESSED_DIR = Path("C:/github/rw/AV-HuBERT-S2S/video_processed")
GLIPS_ROOT = Path("C:/github/rw/AV-HuBERT-S2S/GLips/lipread_files")
DEST_DIR = Path("C:/github/rw/AV-HuBERT-S2S/test_ready")
DEST_DIR.mkdir(exist_ok=True)

# Step 1: Collect all *_lip_movement.mp4 files
lip_files = list(PROCESSED_DIR.glob("*_lip_movement.mp4"))
print(f"📹 Found {len(lip_files)} lip movement videos.")

copied = 0
missing_audio = 0

for lip_path in lip_files:
    stem = lip_path.stem.replace("_lip_movement", "")  # e.g. "aber_0140-0056"
    label = stem.split("_")[0]  # e.g. "aber"
    wav_path = GLIPS_ROOT / label / "test" / f"{stem}.wav"

    if wav_path.exists():
        # Copy both lip video and audio
        shutil.copy2(lip_path, DEST_DIR / lip_path.name)
        shutil.copy2(wav_path, DEST_DIR / wav_path.name)
        print(f"✅ Copied: {lip_path.name}, {wav_path.name}")
        copied += 1
    else:
        print(f"⚠️ Missing audio for: {stem}")
        missing_audio += 1

print(f"\n🏁 Done. {copied} pairs copied. {missing_audio} missing audio files.")
