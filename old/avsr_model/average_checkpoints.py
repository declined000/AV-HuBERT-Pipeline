import os
import torch

def average_checkpoints(last):
    print("⚠️ Skipping checkpoint averaging: no averaging needed for this run.")
    return None  # Return None since we are not actually averaging

def ensemble(args):
    print("⚠️ Ensemble step skipped — not averaging any checkpoints.")
    return "model_avg_10.pth"

