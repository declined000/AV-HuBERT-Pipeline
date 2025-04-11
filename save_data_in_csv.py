import os
import csv

def create_label_csvs(word_root_dir, output_dir=".", relative_to=None):
    splits = {
        "train": [],
        "val": [],
        "test": []
    }

    # If not set, use current working directory (e.g., "research workshop")
    if relative_to is None:
        relative_to = os.getcwd()

    for word in os.listdir(word_root_dir):
        word_path = os.path.join(word_root_dir, word)
        if not os.path.isdir(word_path):
            continue

        for split in splits.keys():
            split_path = os.path.join(word_path, split)
            if not os.path.isdir(split_path):
                continue

            for filename in os.listdir(split_path):
                if filename.endswith(".wav"):
                    abs_path = os.path.join(split_path, filename)
                    rel_path = os.path.relpath(abs_path, start=relative_to)
                    splits[split].append((rel_path.replace("\\", "/"), word))

    for split, data in splits.items():
        csv_filename = os.path.join(output_dir, f"{split}_labels.csv")
        with open(csv_filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["relative_path", "label"])
            writer.writerows(data)
        print(f"✅ {csv_filename} written with {len(data)} samples.")

# ✅ Usage based on your setup:
word_folder_root = r"C:\github\research workshop\GLips\lipread_files"
output_csv_path = r"C:\github\research workshop"
relative_base = r"C:\github\research workshop"  # <-- THIS is the key

create_label_csvs(word_folder_root, output_csv_path, relative_to=relative_base)
