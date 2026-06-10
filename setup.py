"""
Voice AI - Setup Script
========================
Downloads F5-TTS Vietnamese model weights from Hugging Face.
Run once: py -3.10 setup.py
"""

import os
import sys
import shutil

sys.stdout.reconfigure(encoding="utf-8")

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
REPO_ID = "hynt/F5-TTS-Vietnamese-ViVoice"


def main():
    from huggingface_hub import hf_hub_download

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # 1. Download model_last.pt (~5.4GB)
    model_target = os.path.join(WEIGHTS_DIR, "model_last.pt")
    if os.path.exists(model_target):
        size_gb = os.path.getsize(model_target) / 1e9
        print(f"[OK] model_last.pt already exists ({size_gb:.2f} GB), skipping.")
    else:
        print(f"[DOWNLOAD] model_last.pt from {REPO_ID} (~5.4GB, please wait)...")
        try:
            path = hf_hub_download(repo_id=REPO_ID, filename="model_last.pt", local_dir=WEIGHTS_DIR)
            size_gb = os.path.getsize(path) / 1e9
            print(f"[OK] Downloaded model_last.pt ({size_gb:.2f} GB)")
        except Exception as e:
            print(f"[ERROR] Failed: {e}")
            sys.exit(1)

    # 2. Download vocab (stored as config.json on HF, but is actually the vocab file)
    vocab_target = os.path.join(WEIGHTS_DIR, "vocab.txt")
    if os.path.exists(vocab_target):
        lines = sum(1 for _ in open(vocab_target, encoding="utf-8"))
        print(f"[OK] vocab.txt already exists ({lines} tokens), skipping.")
    else:
        print(f"[DOWNLOAD] vocab (config.json -> vocab.txt) from {REPO_ID}...")
        try:
            path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
            shutil.copy2(path, vocab_target)
            lines = sum(1 for _ in open(vocab_target, encoding="utf-8"))
            print(f"[OK] Downloaded vocab.txt ({lines} tokens)")
        except Exception as e:
            print(f"[ERROR] Failed: {e}")
            sys.exit(1)

    print("\n=== Setup complete! ===")
    print(f"Weights directory: {WEIGHTS_DIR}")
    print("Start the server: py -3.10 f5_api.py")


if __name__ == "__main__":
    main()
