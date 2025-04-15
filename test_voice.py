import subprocess

subprocess.run([
    "c:\\github\\research workshop\\env311\\Scripts\\python.exe",
    "auto_avsr\\eval.py",
    "--modality", "audio",
    "--root-dir", "GLips",
    "--test-file", "test_labels.csv",
    "--pretrained-model-path", "GLips\\exp\\glips_audio\\last-v3.ckpt"
])
