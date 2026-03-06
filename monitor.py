#!/usr/bin/env python3
import subprocess
import time
import os

def check_training():
    """Check if training is running"""
    try:
        result = subprocess.run(['pgrep', '-f', 'train_open_vocab_v2.py'], 
                              capture_output=True, text=True)
        return bool(result.stdout.strip())
    except:
        return False

def gpu_status():
    """Show GPU memory usage"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "GPU info unavailable"

def check_checkpoints():
    """List recent checkpoints"""
    ckpt_dir = "runs"
    if os.path.exists(ckpt_dir):
        files = []
        for root, dirs, filenames in os.walk(ckpt_dir):
            for f in filenames:
                if f.endswith('.pth'):
                    files.append(os.path.join(root, f))
        files.sort(key=os.path.getmtime, reverse=True)
        return files[:5]  # Latest 5
    return []

if __name__ == "__main__":
    print("=== Training Monitor ===")
    print(f"Training running: {check_training()}")
    print(f"GPU: {gpu_status()}")
    print("Recent checkpoints:")
    for ckpt in check_checkpoints():
        print(f"  {ckpt}")