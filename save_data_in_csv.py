import os
import csv
import json
from math import floor
import torch

class TextTransform:
    def __init__(self):
        self.vocab = ['<blank>', '<sos>', '<eos>'] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        self.char2index = {c: i for i, c in enumerate(self.vocab)}
        self.index2char = {i: c for i, c in enumerate(self.vocab)}

    def tokenize(self, text):
        text = text.upper()
        return torch.tensor([self.char2index[c] for c in text if c in self.char2index])

    def post_process(self, token_tensor):
        return "".join([self.index2char[int(t)] for t in token_tensor if int(t) in self.index2char])

def extract_duration(txt_path):
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Duration of utterance in seconds:"):
                    return float(line.strip().split(":")[1].strip())
    except Exception as e:
        print(f"❌ Could not read duration from {txt_path}: {e}")
    return 0.0

def create_token_ids(root_dir):
    words = sorted([w for w in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, w))])
    return {word: idx for idx, word in enumerate(words)}

def create_split_label_csv(root_dir, split_name, output_csv, text_transform, limit_ratio=None):
    data = []
    multiplier = 100

    for word in os.listdir(root_dir):
        word_path = os.path.join(root_dir, word)
        if not os.path.isdir(word_path):
            continue

        split_path = os.path.join(word_path, split_name)
        if not os.path.isdir(split_path):
            continue

        wav_files = [f for f in os.listdir(split_path) if f.endswith(".wav")]

        # Subset only a fraction if specified (for train split only)
        if limit_ratio is not None and split_name == "train":
            wav_files = wav_files[:floor(len(wav_files) * limit_ratio)]

        for fname in wav_files:
            base_name = fname[:-4]
            abs_path = os.path.abspath(os.path.join(split_path, fname))
            txt_path = os.path.join(split_path, base_name + ".txt")
            duration = extract_duration(txt_path)

            # Tokenize the word (label) using TextTransform, then format as space-separated token IDs.
            token_ids = text_transform.tokenize(word)
            token_id_str = " ".join(map(str, token_ids.tolist()))
            
            data.append(["GLips", abs_path.replace("\\", "/"), int(duration * multiplier), token_id_str])

        print(f"✅ Processed word: {word} (split: {split_name}, items: {len(wav_files)})")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print(f"📁 Saved {split_name} CSV to: {output_csv} ({len(data)} rows)")

def save_word2id(word2id, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(word2id, f, indent=2, ensure_ascii=False)
    print(f"🧠 Saved word2id map to: {save_path}")

# === Usage ===
base_dir = r"C:\github\research workshop\GLips"
root_dir = os.path.join(base_dir, "lipread_files")
label_dir = os.path.join(base_dir, "labels")
os.makedirs(label_dir, exist_ok=True)

# Create an instance of TextTransform
text_transform = TextTransform()

# Optionally, still create a token mapping (if needed for training)
word2id = create_token_ids(root_dir)

# Save splits using the TextTransform to tokenize the word (folder name)
create_split_label_csv(root_dir, "train", os.path.join(label_dir, "train_labels_subset.csv"), text_transform, limit_ratio=1/20)
create_split_label_csv(root_dir, "val", os.path.join(label_dir, "val_labels.csv"), text_transform, limit_ratio=1/20)
create_split_label_csv(root_dir, "test", os.path.join(label_dir, "test_labels.csv"), text_transform)

# Save token map (if needed for training)
save_word2id(word2id, os.path.join(label_dir, "word2id.json"))
