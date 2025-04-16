import os
import csv
import json
from math import floor
import torch

# A custom TextTransform that uses a vocabulary of whole words.
class TextTransform:
    def __init__(self, vocab):
        # Our vocabulary includes special tokens at the beginning.
        self.token_list = ["<blank>", "<sos>", "<eos>"] + vocab
        self.word2id = {word: idx for idx, word in enumerate(self.token_list)}
        self.ignore_id = -1

    def tokenize(self, text):
        # Convert the word to lower case.
        text = text.lower()
        # Look up the id in our vocabulary; if not found, use the id for <blank>
        token_id = self.word2id.get(text, self.word2id["<blank>"])
        return torch.tensor([token_id])

    def post_process(self, token_tensor):
        # For a single token tensor, return the corresponding word.
        token_id = int(token_tensor[0])
        return self.token_list[token_id]

def extract_duration(txt_path):
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("Duration of utterance in seconds:"):
                    return float(line.strip().split(":")[1].strip())
    except Exception as e:
        print(f"❌ Could not read duration from {txt_path}: {e}")
    return 0.0

def create_vocab(root_dir):
    # Collect folder names (i.e. words) from root_dir as vocabulary items.
    vocab = []
    for word in os.listdir(root_dir):
        word_path = os.path.join(root_dir, word)
        if os.path.isdir(word_path):
            vocab.append(word.lower())
    return sorted(vocab)

def create_split_label_csv(root_dir, split_name, output_csv, text_transform, limit_ratio=None, word2token_map=None):
    data = []
    multiplier = 100  # To scale the duration

    # Process each folder (word) in the root directory.
    for word in os.listdir(root_dir):
        word_path = os.path.join(root_dir, word)
        if not os.path.isdir(word_path):
            continue

        split_path = os.path.join(word_path, split_name)
        if not os.path.isdir(split_path):
            continue

        wav_files = [f for f in os.listdir(split_path) if f.endswith(".wav")]

        # Subset only a fraction if specified (only for the train split).
        if limit_ratio is not None and split_name == "train":
            wav_files = wav_files[:floor(len(wav_files) * limit_ratio)]

        # Tokenize the whole word using our custom TextTransform.
        token_ids_tensor = text_transform.tokenize(word)
        token_ids_list = token_ids_tensor.tolist()
        token_id_str = " ".join(map(str, token_ids_list))

        # Save the mapping from the word to its token IDs.
        if word2token_map is not None:
            word2token_map[word.lower()] = token_ids_list

        for fname in wav_files:
            base_name = fname[:-4]
            abs_path = os.path.abspath(os.path.join(split_path, fname))
            txt_path = os.path.join(split_path, base_name + ".txt")
            duration = extract_duration(txt_path)
            
            # print(f"{word} => {token_ids_tensor} => {token_id_str}")
            data.append(["GLips", abs_path.replace("\\", "/"), int(duration * multiplier), token_id_str])

        print(f"✅ Processed word: {word} (split: {split_name}, items: {len(wav_files)})")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(data)

    print(f"📁 Saved {split_name} CSV to: {output_csv} ({len(data)} rows)")

def save_word2token(word2token, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(word2token, f, indent=2, ensure_ascii=False)
    print(f"🧠 Saved word-to-token map to: {save_path}")

if __name__ == "__main__":
    # Directories
    base_dir = r"C:\github\research workshop\GLips"
    root_dir = os.path.join(base_dir, "lipread_files")
    label_dir = os.path.join(base_dir, "labels")
    os.makedirs(label_dir, exist_ok=True)

    # Create vocabulary from folder names in root_dir
    vocab = create_vocab(root_dir)
    print("Vocabulary:", vocab)

    # Create a TextTransform instance with our own vocabulary.
    text_transform = TextTransform(vocab)

    # Shared mapping dictionary from word to token IDs.
    word2token_map = {}

    # Generate CSV files for each split and update the word-to-token mapping.
    create_split_label_csv(root_dir, "train", os.path.join(label_dir, "train_labels_subset.csv"),
                           text_transform, limit_ratio=1/20, word2token_map=word2token_map)
    create_split_label_csv(root_dir, "val", os.path.join(label_dir, "val_labels.csv"),
                           text_transform, limit_ratio=1/20, word2token_map=word2token_map)
    create_split_label_csv(root_dir, "test", os.path.join(label_dir, "test_labels.csv"),
                           text_transform, word2token_map=word2token_map)

    # Save the full word-to-token mapping.
    save_word2token(word2token_map, os.path.join(label_dir, "word2token.json"))
