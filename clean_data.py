import os

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

delete_m4a_files(folder_path)
