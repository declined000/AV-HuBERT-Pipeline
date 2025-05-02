import os
import shutil
from itertools import islice
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.dataset.video_to_audio_lips import process_raw_data_for_avsr

# Configuration
LIPREAD_ROOT = "C:/github/rw/AV-HuBERT-S2S/GLips/lipread_files"
TEMP_RAW_DIR = "C:/github/rw/AV-HuBERT-S2S/raw_face_videos"
OUTPUT_DIR = "C:/github/rw/AV-HuBERT-S2S/video_processed"
MAX_WORKERS = 10  # Adjust based on your CPU


def prepare_directories():
    os.makedirs(TEMP_RAW_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def copy_test_videos(max_files_per_label=10):
    print("\n📦 Copying test videos...")
    for label in os.listdir(LIPREAD_ROOT):
        test_path = os.path.join(LIPREAD_ROOT, label, "test")
        if not os.path.isdir(test_path):
            continue

        print(f"📁 Processing label: {label}")
        copied = 0

        for file in os.listdir(test_path):
            if file.endswith(".mp4"):
                src_path = os.path.join(test_path, file)
                dst_path = os.path.join(TEMP_RAW_DIR, file)
                shutil.copy2(src_path, dst_path)
                copied += 1

                if copied >= max_files_per_label:
                    break

        print(f"  ✅ Copied {copied} video(s) from '{label}'")


def should_skip(file):
    return (
        not file.endswith(".mp4") or
        "_lip_movement" in file or
        "_normalized_video" in file or
        "_video_" in file
    )


def process_file(file):
    video_path = os.path.join(TEMP_RAW_DIR, file)
    try:
        print(f"▶️ Processing: {file}")
        result = process_raw_data_for_avsr(
            input_file_path=video_path,
            output_dir=OUTPUT_DIR
        )
        print(f"✅ Done: {file}")
        return file, "success", result
    except Exception as e:
        print(f"❌ Failed: {file} — {e}")
        return file, "error", str(e)


def run_preprocessing():
    print("\n⚙️ Running preprocessing (lip-only extraction per video)...")
    files = [f for f in os.listdir(TEMP_RAW_DIR) if not should_skip(f)]
    print(f"📁 Found {len(files)} valid files to process in {TEMP_RAW_DIR}")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_file, file): file for file in files}

        for i, future in enumerate(as_completed(futures), start=1):
            file = futures[future]
            try:
                filename, status, result = future.result()
                print(f"📝 {i}. {filename}: {status}")
            except Exception as exc:
                print(f"⚠️ Exception occurred in {file}: {exc}")

    print("\n🏁 Finished processing all files.")


if __name__ == "__main__":
    prepare_directories()
    # copy_test_videos(max_files_per_label=10)
    run_preprocessing()
