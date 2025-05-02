import os
import csv
import pandas as pd
import torch
from transformers import Speech2TextTokenizer
from src.model.avhubert2text import AV2TextForConditionalGeneration
from src.dataset.load_data import load_feature

# CONFIG
CSV_INPUT = "c:/github/rw/AV-HuBERT-S2S/glips_filelist.csv"
CSV_OUTPUT = "glips_inference_results.csv"
MODEL_NAME = "nguyenvulebinh/AV-HuBERT-MuAViC-de"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
print("📦 Loading model and tokenizer...")
model = AV2TextForConditionalGeneration.from_pretrained(MODEL_NAME, cache_dir="./model-bin").to(DEVICE).eval()
tokenizer = Speech2TextTokenizer.from_pretrained(MODEL_NAME, cache_dir="./model-bin")
print("✅ Model and tokenizer loaded.")

# Load file info from CSV
print(f"📄 Reading input file: {CSV_INPUT}")
df = pd.read_csv(CSV_INPUT)

# Build a mapping of file groups
file_map = {}
for _, row in df.iterrows():
    basename = row["filename"].replace(".mp4", "").replace(".wav", "")
    if basename not in file_map:
        file_map[basename] = {"label": row["label"]}
    if row["filetype"] == "audio":
        file_map[basename]["audio"] = row["filepath"]
    elif row["filetype"] == "video":
        file_map[basename]["video"] = row["filepath"]

# Start inference and save output
print(f"📝 Creating output CSV: {CSV_OUTPUT}")
with open(CSV_OUTPUT, mode="w", encoding="utf-8", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["filename", "input_type", "ground_truth", "predicted_text"])

    for basename, entry in file_map.items():
        audio_path = entry.get("audio")
        video_path = entry.get("video")
        label = entry.get("label", "")

        if not audio_path or not video_path:
            print(f"⚠️ Skipping {basename}: missing {'audio' if not audio_path else 'video'}.")
            continue

        print(f"\n🔍 Processing sample: {basename}")
        print(f"   ▶ Ground truth label: {label}")
        print(f"   ▶ Audio path: {audio_path}")
        print(f"   ▶ Video path: {video_path}")

        try:
            sample = load_feature(video_path, audio_path)
            audio_feats = sample["audio_source"].to(DEVICE)
            video_feats = sample["video_source"].to(DEVICE)
            attention_mask = torch.BoolTensor(audio_feats.shape[0], audio_feats.shape[-1]).fill_(False).to(DEVICE)

            # Audio-only inference
            with torch.no_grad():
                print("   🔈 Running audio-only inference...")
                audio_output = model.generate(
                    audio_feats,
                    attention_mask=attention_mask,
                    video=None,
                    max_length=1024,
                )
            audio_text = tokenizer.batch_decode(audio_output, skip_special_tokens=True)[0]
            print(f"   ✅ Audio-only prediction: {audio_text}")
            writer.writerow([basename, "audio-only", label, audio_text])

            # Video-only inference
            with torch.no_grad():
                print("   📹 Running video-only inference...")
                video_output = model.generate(
                    audio=None,
                    attention_mask=None,
                    video=video_feats,
                    max_length=1024,
                )
            video_text = tokenizer.batch_decode(video_output, skip_special_tokens=True)[0]
            print(f"   ✅ Video-only prediction: {video_text}")
            writer.writerow([basename, "video-only", label, video_text])

        except Exception as e:
            print(f"❌ Error processing {basename}: {e}")
