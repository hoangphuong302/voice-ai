"""
Pre-compute voice embeddings for all voices in the library.
This script loads the Viterbox model ONCE, then processes all voices,
saving their conditioning tensors as .pt files for instant loading during inference.

Run this once or whenever new voices are added.
Usage: py -3.10 precompute_embeddings.py
"""
import json
import os
import sys
import time
import torch
from pathlib import Path

# Fix Windows Unicode encoding
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, "D:/voice-ai/viterbox-tts")

ROOT = Path("D:/voice-ai")
VOICES_FILE = ROOT / "voices.json"

def main():
    print("=" * 60)
    print("  Voice AI - Pre-compute Embeddings for All Voices")
    print("=" * 60)
    
    # Load voices database
    with open(VOICES_FILE, "r", encoding="utf-8") as f:
        voices = json.load(f)
    
    # Filter voices that have ref_audio and aren't minimax
    eligible = []
    for vid, v in voices.items():
        ref_audio = v.get("ref_audio", "")
        model = v.get("model", "")
        if model.startswith("minimax/"):
            continue  # Skip cloud voices
        if not ref_audio:
            continue
        ref_path = ROOT / ref_audio
        if not ref_path.exists():
            print(f"  SKIP {vid}: ref_audio not found at {ref_path}")
            continue
        eligible.append((vid, v, ref_path))
    
    print(f"\nFound {len(eligible)} voices with ref audio (out of {len(voices)} total)")
    
    # Check which already have cached embeddings
    need_compute = []
    for vid, v, ref_path in eligible:
        cache_path = ROOT / "voices" / vid / "embeddings.pt"
        if cache_path.exists():
            # Check if cache is newer than ref audio
            ref_mtime = os.path.getmtime(ref_path)
            cache_mtime = os.path.getmtime(cache_path)
            if cache_mtime > ref_mtime:
                print(f"  CACHED {vid} ({v.get('name', vid)})")
                continue
        need_compute.append((vid, v, ref_path))
    
    if not need_compute:
        print("\nAll voices already have cached embeddings!")
        return
    
    print(f"\nNeed to compute embeddings for {len(need_compute)} voices")
    print("Loading Viterbox model...")
    
    # Load model once
    from viterbox import Viterbox
    tts = Viterbox.from_pretrained("cuda")
    print("Model loaded!\n")
    
    # Process each voice
    success = 0
    failed = 0
    for i, (vid, v, ref_path) in enumerate(need_compute):
        name = v.get("name", vid)
        print(f"[{i+1}/{len(need_compute)}] Processing: {name} ({vid})")
        print(f"  ref_audio: {ref_path}")
        
        try:
            t_start = time.time()
            
            # Prepare conditionals with default exaggeration
            # (emotion_adv will be overridden at inference time)
            tts.prepare_conditionals(str(ref_path), exaggeration=0.5)
            
            # Save embeddings
            cache_dir = ROOT / "voices" / vid
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / "embeddings.pt"
            
            # Save the full conditioning (t3 + s3)
            tts.conds.save(str(cache_path))
            
            # Also save ref_wav separately for completeness
            ref_wav_path = cache_dir / "ref_wav_cached.pt"
            if tts.conds.ref_wav is not None:
                torch.save(tts.conds.ref_wav.cpu(), str(ref_wav_path))
            
            dt = time.time() - t_start
            size_kb = os.path.getsize(cache_path) / 1024
            print(f"  ✅ Saved embeddings.pt ({size_kb:.0f} KB) in {dt:.1f}s")
            success += 1
            
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            failed += 1
        
        # Clear CUDA cache periodically
        if (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    print(f"\n{'=' * 60}")
    print(f"  DONE: {success} succeeded, {failed} failed")
    print(f"  Total cached voices: {success + sum(1 for vid, v, ref in eligible if (ROOT / 'voices' / vid / 'embeddings.pt').exists())}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
