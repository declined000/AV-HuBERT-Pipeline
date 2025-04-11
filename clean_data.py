import os
import shutil


def delete_m4a_files(directory):
    deleted = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".m4a"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    # print(f"Deleted: {file_path}")
                    deleted += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
    print(f"\n✅ Done. {deleted} .m4a files deleted.")

# Replace this with your actual folder path
folder_path = r"C:\github\research workshop\GLips"

# delete_m4a_files(folder_path)



def remove_wildcard_folders(base_path):
    removed_folders = []
    
    for root, dirs, _ in os.walk(base_path):
        for folder in dirs:
            print(folder)
            if "+" in folder:
                folder_path = os.path.join(root, folder)
                try:
                    shutil.rmtree(folder_path)
                    removed_folders.append(folder_path)
                    print(f"🗑️ Removed: {folder_path}")
                except Exception as e:
                    print(f"⚠️ Failed to remove {folder_path}: {e}")
        # Only process top-level dirs in each loop
        break

    print(f"\n✅ Done. Total folders removed: {len(removed_folders)}")
    print("\n🧾 Examples of removed folders:")
    for example in removed_folders[:5]:
        print(f" - {example}")

# Replace with the path to your folder containing word-named subfolders
dataset_path = r"C:\github\research workshop\GLips\lipread_files"

# remove_wildcard_folders(dataset_path)

#  the output of above function

# ✅ Done. Total folders removed: 64

# 🧾 Examples of removed folders:
#  - C:\github\research workshop\GLips\lipread_files\++ber
#  - C:\github\research workshop\GLips\lipread_files\++berhaupt
#  - C:\github\research workshop\GLips\lipread_files\++brigens
#  - C:\github\research workshop\GLips\lipread_files\+Âffentlichen
#  - C:\github\research workshop\GLips\lipread_files\+ñnderung