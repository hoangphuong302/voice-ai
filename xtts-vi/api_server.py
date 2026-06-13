"""
XTTS-v2 Vietnamese API Server
Fine-tuned on 941h PhoAudiobook (professional audiobook recordings)
Model: thivux/XTTS-v2-vietnamse
Runs on port 8082
"""
import io
import os
import re
import sys
import time
import json
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from scipy import signal as scipy_signal

app = Flask(__name__)

MODEL = None
XTTS_DIR = Path(__file__).parent
SAMPLE_RATE = 24000

# Vietnamese text preprocessing
def preprocess_vietnamese(text):
    text = re.sub(r'\s+', ' ', text).strip()
    if text and text[-1] not in '.!?':
        text += '.'
    text = text.replace('"', ' ').replace('"', ' ').replace('"', ' ')
    text = re.sub(r'(\d+)%', r'\1 phan tram', text)
    return text


def split_into_chunks(text):
    """Split at every punctuation for short, accurate chunks."""
    parts = re.split(r'([.!?,;:]+)', text)
    chunks = []
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        if not chunk:
            continue
        punct = parts[i + 1].strip() if i + 1 < len(parts) else ''
        if punct:
            chunk = chunk + punct
        if any(p in punct for p in '.!?'):
            pause_ms = 500
        elif any(p in punct for p in ',;'):
            pause_ms = 280
        else:
            pause_ms = 150
        if len(chunk) > 60:
            words = chunk.split()
            mid = len(words) // 2
            if mid > 0:
                chunks.append((' '.join(words[:mid]), 200))
                chunks.append((' '.join(words[mid:]), pause_ms))
            else:
                chunks.append((chunk, pause_ms))
        else:
            chunks.append((chunk, pause_ms))
    return chunks if chunks else [(text, 300)]


def normalize_audio(audio, target_peak=0.85):
    if len(audio) == 0:
        return audio
    audio = audio - np.mean(audio)
    try:
        sos = scipy_signal.butter(2, 50, btype='highpass', fs=SAMPLE_RATE, output='sos')
        audio = scipy_signal.sosfilt(sos, audio).astype(np.float32)
    except Exception:
        pass
    peak = np.abs(audio).max()
    if peak > 0.001:
        audio = audio * (target_peak / peak)
    return audio.astype(np.float32)


def join_with_pauses(segments, pauses, sr):
    if not segments:
        return np.array([], dtype=np.float32)
    if len(segments) == 1:
        return segments[0]
    result = segments[0].copy()
    for i, seg in enumerate(segments[1:]):
        if len(seg) == 0:
            continue
        pause_ms = pauses[i] if i < len(pauses) else 300
        pause = np.zeros(int(sr * pause_ms / 1000), dtype=np.float32)
        fade_len = min(int(sr * 0.01), len(result), len(seg))
        if fade_len > 0:
            result[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            seg = seg.copy()
            seg[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        result = np.concatenate([result, pause, seg])
    return result


def load_model():
    global MODEL, SAMPLE_RATE
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    model_dir = XTTS_DIR / "model"
    config_path = model_dir / "config.json"
    
    if not config_path.exists():
        # Download from HuggingFace
        print("[XTTS-Vi] Downloading model from thivux/XTTS-v2-vietnamse ...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id="thivux/XTTS-v2-vietnamse",
            local_dir=str(model_dir),
        )
        print("[XTTS-Vi] Download complete!")

    print("[XTTS-Vi] Loading model ...")
    config = XttsConfig()
    config.load_json(str(config_path))
    
    MODEL = Xtts.init_from_config(config)
    MODEL.load_checkpoint(
        config,
        checkpoint_dir=str(model_dir),
        checkpoint_path=str(model_dir / "best_model.pth"),
        vocab_path=str(model_dir / "vocab.json"),
        use_deepspeed=False,
    )
    MODEL.cuda()
    print("[XTTS-Vi] Model loaded on CUDA!")
    
    # Patch tokenizer to support Vietnamese
    if hasattr(MODEL, 'tokenizer') and hasattr(MODEL.tokenizer, 'char_limits'):
        MODEL.tokenizer.char_limits['vi'] = 250
        # Monkey-patch preprocess_text to support Vietnamese
        _orig_preprocess = MODEL.tokenizer.preprocess_text
        def _patched_preprocess(txt, lang):
            if lang == 'vi':
                # Vietnamese: just clean whitespace, no special preprocessing needed
                import re as _re
                txt = _re.sub(r'\s+', ' ', txt).strip()
                return txt
            return _orig_preprocess(txt, lang)
        MODEL.tokenizer.preprocess_text = _patched_preprocess
        print("[XTTS-Vi] Patched tokenizer: added 'vi' language support")
    
    # Get sample rate from config
    SAMPLE_RATE = config.audio.sample_rate if hasattr(config, 'audio') else 24000
    print(f"[XTTS-Vi] Sample rate: {SAMPLE_RATE}Hz")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "XTTS-v2-Vietnamese", "engine": "PhoAudiobook-941h"})


@app.route("/tts", methods=["POST"])
def tts():
    global MODEL, SAMPLE_RATE
    if MODEL is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.json or {}
    text = data.get("text", "")
    ref_audio_path = data.get("ref_audio", "")
    ref_text = data.get("ref_text", "")

    if not text:
        return jsonify({"error": "Missing text"}), 400

    if not ref_audio_path or not os.path.exists(ref_audio_path):
        return jsonify({"error": f"Reference audio not found: {ref_audio_path}"}), 400

    try:
        t0 = time.time()
        
        # Extract dynamic generation parameters with natural defaults
        temperature = float(data.get("temperature", 0.7))
        repetition_penalty = float(data.get("repetition_penalty", 2.0))
        top_k = int(data.get("top_k", 50))
        top_p = float(data.get("top_p", 0.85))

        # Compute speaker embedding from reference audio
        gpt_cond_latent, speaker_embedding = MODEL.get_conditioning_latents(
            audio_path=[ref_audio_path]
        )
        
        # Preprocess and split text
        text = preprocess_vietnamese(text)
        chunks = split_into_chunks(text)
        print(f"[XTTS-Vi] Split into {len(chunks)} chunks")

        audio_segments = []
        pause_durations = []

        for i, (chunk_text, pause_ms) in enumerate(chunks):
            if not chunk_text.strip():
                continue
            
            print(f"[XTTS-Vi]   [{i+1}/{len(chunks)}] {chunk_text[:50]}...")
            
            out = MODEL.inference(
                text=chunk_text,
                language="vi",
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                top_k=top_k,
                top_p=top_p,
            )
            
            audio = np.array(out["wav"], dtype=np.float32)
            audio = normalize_audio(audio)
            
            # Trim silence
            threshold = 0.01
            nonzero = np.where(np.abs(audio) > threshold)[0]
            if len(nonzero) > 0:
                start = max(0, nonzero[0] - int(SAMPLE_RATE * 0.015))
                end = min(len(audio), nonzero[-1] + int(SAMPLE_RATE * 0.015))
                audio = audio[start:end]
            
            audio_segments.append(audio)
            pause_durations.append(pause_ms)

        # Join with punctuation-aware pauses
        if audio_segments:
            final_audio = join_with_pauses(audio_segments, pause_durations, SAMPLE_RATE)
        else:
            final_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)

        final_audio = normalize_audio(final_audio, target_peak=0.9)

        dt = time.time() - t0
        duration = len(final_audio) / SAMPLE_RATE
        print(f"[XTTS-Vi] Done in {dt:.1f}s, {len(chunks)} chunks, {duration:.1f}s audio")

        buf = io.BytesIO()
        sf.write(buf, final_audio, SAMPLE_RATE, format="WAV")
        buf.seek(0)

        return send_file(buf, mimetype="audio/wav", as_attachment=True, download_name="output.wav")

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("  XTTS-v2 Vietnamese API Server v1.2")
    print("  Port: 8082 | Model: thivux/XTTS-v2-vietnamse")
    print("  Trained on: 941h PhoAudiobook (professional)")
    print("=" * 60)
    load_model()
    app.run(host="127.0.0.1", port=8082, debug=False)
