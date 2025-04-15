import cv2
import torch
import os

# === CONFIGURATION ===
video_path = r"C:\github\research workshop\GLips\lipread_files\aber\train\aber_0137-0015.mp4"

# === MODEL EXPECTATIONS ===
expected_fps = 25
expected_frame_size = (88, 88)  # Height, Width
expected_channels = 1  # Grayscale

# === VALIDATION FUNCTION ===
def validate_video_format(video_path):
    print(f"🔍 Validating video: {video_path}")

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"❌ File not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Cannot open video: {video_path}")

    # FPS check
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"🎥 FPS: {fps}")
    assert abs(fps - expected_fps) < 1, f"❌ Expected ~{expected_fps} FPS, got {fps}"

    # Frame count
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📦 Total frames: {frame_count}")

    frames = []
    for _ in range(min(5, frame_count)):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, expected_frame_size)
        tensor_frame = torch.tensor(resized).unsqueeze(0).float() / 255.0
        frames.append(tensor_frame)

    cap.release()

    if not frames:
        raise ValueError("❌ No valid frames found.")

    video_tensor = torch.stack(frames)  # T x 1 x H x W
    print(f"📐 Tensor shape: {video_tensor.shape}")
    assert video_tensor.ndim == 4, "❌ Tensor should be 4D (T x C x H x W)"
    assert video_tensor.shape[1] == expected_channels, f"❌ Expected {expected_channels} channel(s)"
    assert video_tensor.shape[2:] == expected_frame_size, f"❌ Frame size mismatch: {video_tensor.shape[2:]}"
    assert video_tensor.dtype == torch.float32, "❌ Expected dtype float32"

    print("✅ Video is valid and ready for the model.")

# === RUN ===
validate_video_format(video_path)
