import os
import csv
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import Speech2TextTokenizer
from src.model.avhubert2text import AV2TextForConditionalGeneration
from src.dataset.load_data import load_feature

# Paths
PROCESSED_DIR = "C:/github/rw/AV-HuBERT-S2S/video_processed"
AUDIO_SOURCE_DIR = "C:/github/rw/AV-HuBERT-S2S/GLips/lipread_files"
CSV_OUTPUT_PATH = "C:/github/rw/AV-HuBERT-S2S/inference_results.csv"
LANGUAGE = "de"
MAX_WORKERS = 16    # Keep at 1 unless you know your GPU can handle more

# Load model
print("📦 Loading model...")
model_name = f"nguyenvulebinh/AV-HuBERT-MuAViC-{LANGUAGE}"
model = AV2TextForConditionalGeneration.from_pretrained(model_name, cache_dir="./model-bin")
tokenizer = Speech2TextTokenizer.from_pretrained(model_name, cache_dir="./model-bin")
model = model.cuda().eval()

def run_inference(file):
    filename_base = file.replace("_lip_movement.mp4", "")
    video_path = os.path.join(PROCESSED_DIR, file)

    # Find audio file
    audio_path = None
    for root, _, files in os.walk(AUDIO_SOURCE_DIR):
        for f in files:
            if f.startswith(filename_base) and f.endswith(".wav"):
                audio_path = os.path.join(root, f)
                break
        if audio_path:
            break

    if not audio_path or not os.path.exists(audio_path):
        print(f"❌ Skipping {filename_base}: audio file not found.")
        return None

    try:
        sample = load_feature(video_path, audio_path)
        audio_feats = sample["audio_source"].cuda()
        video_feats = sample["video_source"].cuda()
        attention_mask = torch.BoolTensor(audio_feats.size(0), audio_feats.size(-1)).fill_(False).cuda()

        output_audio = model.generate(audio_feats, attention_mask=attention_mask, video=torch.zeros_like(video_feats))
        output_video = model.generate(torch.zeros_like(audio_feats), attention_mask=None, video=video_feats)
        output_both = model.generate(audio_feats, attention_mask=attention_mask, video=video_feats)

        decoded_audio = tokenizer.batch_decode(output_audio, skip_special_tokens=True)[0]
        decoded_video = tokenizer.batch_decode(output_video, skip_special_tokens=True)[0]
        decoded_both = tokenizer.batch_decode(output_both, skip_special_tokens=True)[0]

        print(f"✅ {filename_base}")
        return {
            "filename": filename_base,
            "audio_only_prediction": decoded_audio,
            "video_only_prediction": decoded_video,
            "audio_video_prediction": decoded_both
        }
    except Exception as e:
        print(f"❌ Error processing {filename_base}: {e}")
        return None

if __name__ == "__main__":
    print("⚙️ Starting inference...")
    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith("_lip_movement.mp4")]
    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_inference, file) for file in files]
        for future in as_completed(futures):
            result = future.result()
            if result:
                results.append(result)

    print(f"\n💾 Saving results to {CSV_OUTPUT_PATH}")
    with open(CSV_OUTPUT_PATH, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["filename", "audio_only_prediction", "video_only_prediction", "audio_video_prediction"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print("✅ All done.")
