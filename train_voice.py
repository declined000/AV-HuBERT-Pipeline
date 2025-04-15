import subprocess

subprocess.run([
    "c:\\github\\research workshop\\env311\\Scripts\\python.exe",
    "auto_avsr\\train.py",
    "--exp-dir", "GLips\\exp",
    "--exp-name", "glips_audio",
    "--modality", "audio",
    "--root-dir", "GLips",
    "--train-file", "train_labels_subset.csv",
    "--val-file", "val_labels.csv",     # <-- explicitly added
    "--test-file", "test_labels.csv",   # <-- explicitly added
    "--num-nodes", "1",
    "--gpus", "1",
    "--max-frames", "512",
    "--max-epochs", "1"
])
