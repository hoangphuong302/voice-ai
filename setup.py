"""
Voice AI - Setup Script
========================
Downloads F5-TTS Vietnamese model weights from Hugging Face.
Run once: py -3.10 setup.py
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
REPO_ID = "hynt/F5-TTS-Vietnamese-ViVoice"
FILES = ["model_last.pt", "vocab.txt"]


def main():
    from huggingface_hub import hf_hub_download

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    for fname in FILES:
        target = os.path.join(WEIGHTS_DIR, fname)
        if os.path.exists(target):
            size_mb = os.path.getsize(target) / 1e6
            print(f"[OK] {fname} already exists ({size_mb:.1f} MB), skipping.")
            continue

        print(f"[DOWNLOAD] {fname} from {REPO_ID}...")
        try:
            path = hf_hub_download(
                repo_id=REPO_ID,
                filename=fname,
                local_dir=WEIGHTS_DIR,
            )
            size_mb = os.path.getsize(path) / 1e6
            print(f"[OK] Downloaded {fname} ({size_mb:.1f} MB) -> {path}")
        except Exception as e:
            print(f"[ERROR] Failed to download {fname}: {e}")
            sys.exit(1)

    print("\n=== Setup complete! ===")
    print(f"Weights directory: {WEIGHTS_DIR}")
    print("Start the server: py -3.10 f5_api.py")


if __name__ == "__main__":
    main()
