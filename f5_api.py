"""
Voice AI - F5-TTS Local API Server
===================================
High-quality Vietnamese voice cloning powered by F5-TTS.
Runs on port 9880, integrates with OpenClaw tab on web-mcbooks-export.

Modes:
  - preset: Use saved voices from voices.json
  - fast:   Zero-shot clone with 3-10s reference audio
  - train:  Fine-tune on longer recordings (1-10 min)

Data persistence:
  - voices.json + voices/ folder → git repo on D: + GitHub
  - weights/ folder → local only (gitignored)
"""

import os
import sys
import json
import time
import shutil
import hashlib
import subprocess
import threading
import tempfile
from pathlib import Path
from datetime import datetime, timezone

# Force UTF-8
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
VOICES_JSON = ROOT / "voices.json"
VOICES_DIR = ROOT / "voices"
WEIGHTS_DIR = ROOT / "weights"
OUTPUT_DIR = ROOT / "output"
UPLOADS_DIR = ROOT / "uploads"

for d in [VOICES_DIR, WEIGHTS_DIR, OUTPUT_DIR, UPLOADS_DIR]:
    d.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# Flask App
# ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ──────────────────────────────────────────────
# F5-TTS Model (lazy loaded)
# ──────────────────────────────────────────────
_model = None
_vocoder = None
_model_lock = threading.Lock()

# Training state
_train_status = {
    "running": False,
    "progress": "",
    "error": None,
    "started_at": None,
    "finished_at": None,
}
_train_lock = threading.Lock()


def _get_model():
    """Lazy-load F5-TTS model + vocoder on first request."""
    global _model, _vocoder
    if _model is not None:
        return _model, _vocoder

    with _model_lock:
        if _model is not None:
            return _model, _vocoder

        print("[Voice AI] Loading F5-TTS model...")
        import torch
        from f5_tts.model import DiT
        from f5_tts.infer.utils_infer import load_model, load_vocoder

        ckpt_path = str(WEIGHTS_DIR / "model_last.pt")
        vocab_file = str(WEIGHTS_DIR / "vocab.txt")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Model weights not found at {ckpt_path}. "
                "Run: py -3.10 setup.py to download weights."
            )
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(
                f"Vocab file not found at {vocab_file}. "
                "Run: py -3.10 setup.py to download weights."
            )

        # F5-TTS Vietnamese model config (DiT architecture)
        model_cfg = dict(
            dim=1024, depth=22, heads=16,
            ff_mult=2, text_dim=512, conv_layers=4
        )

        _model = load_model(
            model_cls=DiT,
            model_cfg=model_cfg,
            ckpt_path=ckpt_path,
            mel_spec_type="vocos",
            vocab_file=vocab_file,
        )

        _vocoder = load_vocoder(vocoder_name="vocos")
        print("[Voice AI] Model loaded successfully!")
        return _model, _vocoder


# ──────────────────────────────────────────────
# Voices Database
# ──────────────────────────────────────────────
def _load_voices():
    if not VOICES_JSON.exists():
        return {}
    try:
        return json.loads(VOICES_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_voices(data):
    VOICES_JSON.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


# ──────────────────────────────────────────────
# Git Sync (background)
# ──────────────────────────────────────────────
def _git_sync(message="Auto-sync voices"):
    """Commit and push voices.json + voices/ to GitHub."""
    try:
        cwd = str(ROOT)
        subprocess.run(["git", "add", "voices.json", "voices/"], cwd=cwd,
                        capture_output=True, timeout=10)
        subprocess.run(["git", "commit", "-m", message], cwd=cwd,
                        capture_output=True, timeout=10)
        result = subprocess.run(["git", "push", "origin", "main"], cwd=cwd,
                                 capture_output=True, timeout=30)
        if result.returncode == 0:
            print(f"[Voice AI] Git sync OK: {message}")
        else:
            # Try 'master' branch if 'main' fails
            subprocess.run(["git", "push", "origin", "master"], cwd=cwd,
                            capture_output=True, timeout=30)
            print(f"[Voice AI] Git sync OK (master): {message}")
    except Exception as e:
        print(f"[Voice AI] Git sync failed: {e}")


def _git_sync_async(message="Auto-sync voices"):
    threading.Thread(target=_git_sync, args=(message,), daemon=True).start()


# ──────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "Voice AI",
        "engine": "F5-TTS",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": _model is not None,
    })


@app.route("/voices", methods=["GET"])
def list_voices():
    """Return all saved voices."""
    voices = _load_voices()
    result = []
    for vid, v in voices.items():
        result.append({
            "id": vid,
            "name": v.get("name", vid),
            "desc": v.get("desc", ""),
            "type": v.get("type", "preset"),
            "created_at": v.get("created_at", ""),
        })
    return jsonify(result)


@app.route("/voices", methods=["POST"])
def save_voice():
    """
    Save a new voice to the library.
    Expects multipart form with:
      - file: reference audio (wav)
      - name: display name
      - desc: description
      - ref_text: transcript of the audio
      - type: 'cloned' or 'trained'
    """
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No audio file"}), 400

    file = request.files["file"]
    name = request.form.get("name", "Custom Voice")
    desc = request.form.get("desc", "")
    ref_text = request.form.get("ref_text", "")
    vtype = request.form.get("type", "cloned")

    # Generate unique voice ID
    vid = f"voice_{int(time.time())}_{hashlib.md5(name.encode()).hexdigest()[:6]}"

    # Save reference audio
    voice_dir = VOICES_DIR / vid
    voice_dir.mkdir(exist_ok=True)
    ref_path = voice_dir / "ref.wav"
    file.save(str(ref_path))

    # Update voices database
    voices = _load_voices()
    voices[vid] = {
        "name": name,
        "desc": desc,
        "ref_audio": f"voices/{vid}/ref.wav",
        "ref_text": ref_text,
        "type": vtype,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_voices(voices)

    # Sync to GitHub in background
    _git_sync_async(f"Add voice: {name}")

    return jsonify({
        "success": True,
        "voice_id": vid,
        "message": f"Voice '{name}' saved and syncing to GitHub.",
    })


@app.route("/voices/<voice_id>", methods=["DELETE"])
def delete_voice(voice_id):
    """Delete a voice from the library."""
    voices = _load_voices()
    if voice_id not in voices:
        return jsonify({"success": False, "error": "Voice not found"}), 404

    name = voices[voice_id].get("name", voice_id)
    del voices[voice_id]
    _save_voices(voices)

    # Remove voice directory
    voice_dir = VOICES_DIR / voice_id
    if voice_dir.exists():
        shutil.rmtree(voice_dir, ignore_errors=True)

    _git_sync_async(f"Delete voice: {name}")
    return jsonify({"success": True, "message": f"Deleted '{name}'"})


@app.route("/tts", methods=["POST"])
def text_to_speech():
    """
    Generate speech from text.

    JSON body:
      - text: Vietnamese text to synthesize
      - voice: voice ID (for preset mode)
      - mode: 'preset' or 'fast'
      - speed: speed factor (0.5-2.0, default 1.0)
      - ref_audio_path: path to uploaded ref audio (for fast mode)
      - ref_text: transcript of ref audio (for fast mode)
      - outputPath: optional path to save output WAV
    """
    import torch
    import soundfile as sf
    from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    mode = data.get("mode", "preset")
    voice_id = data.get("voice", "nu6")
    speed = float(data.get("speed", data.get("speed_factor", 1.0)))
    output_path = data.get("outputPath", "")

    if not text:
        return jsonify({"success": False, "message": "Thiếu văn bản cần đọc"}), 400

    try:
        model, vocoder = _get_model()
    except Exception as e:
        return jsonify({"success": False, "message": f"Lỗi tải model: {e}"}), 500

    # Determine reference audio and text
    if mode == "fast":
        ref_audio_path = data.get("ref_audio_path", "")
        ref_text = data.get("prompt_text", data.get("ref_text", "")).strip()
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            return jsonify({"success": False, "message": "Không tìm thấy file âm thanh mẫu"}), 400
    else:
        # Preset mode - load from voices.json
        voices = _load_voices()
        if voice_id not in voices:
            return jsonify({"success": False, "message": f"Voice '{voice_id}' không tồn tại"}), 404

        voice_cfg = voices[voice_id]
        ref_audio_rel = voice_cfg.get("ref_audio", "")
        ref_audio_path = str(ROOT / ref_audio_rel)
        ref_text = voice_cfg.get("ref_text", "")

        if not os.path.exists(ref_audio_path):
            return jsonify({"success": False, "message": f"File audio mẫu không tồn tại: {ref_audio_path}"}), 404

    print(f"[Voice AI] TTS request: mode={mode}, voice={voice_id}, text={text[:60]}...")

    try:
        # Preprocess reference audio
        ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
            ref_audio_path, ref_text
        )

        # Run inference
        wav_data, sample_rate, spectrogram = infer_process(
            ref_audio=ref_audio_processed,
            ref_text=ref_text_processed,
            gen_text=text,
            model_obj=model,
            vocoder=vocoder,
            speed=speed,
        )

        if wav_data is None:
            return jsonify({"success": False, "message": "Không tạo được audio (text quá ngắn?)"}), 500

        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(OUTPUT_DIR))
        sf.write(tmp.name, wav_data, sample_rate)
        tmp.close()

        # Also save to outputPath if specified
        if output_path and output_path.strip():
            try:
                os.makedirs(output_path.strip(), exist_ok=True)
                final_name = os.path.join(
                    output_path.strip(),
                    f"voice_ai_{voice_id}_{int(time.time())}.wav"
                )
                shutil.copy2(tmp.name, final_name)
                print(f"[Voice AI] Output saved to: {final_name}")
            except Exception as e:
                print(f"[Voice AI] Could not save to outputPath: {e}")

        print(f"[Voice AI] TTS done, {len(wav_data)} samples at {sample_rate}Hz")
        return send_file(
            tmp.name,
            mimetype="audio/wav",
            as_attachment=False,
            download_name="voice_ai_output.wav"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Lỗi tổng hợp: {str(e)}"}), 500


@app.route("/upload", methods=["POST"])
def upload_audio():
    """Upload a reference audio file for cloning. Returns the saved path."""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file"}), 400

    file = request.files["file"]
    fname = f"{int(time.time())}_{file.filename}"
    fpath = UPLOADS_DIR / fname
    file.save(str(fpath))

    return jsonify({
        "success": True,
        "filename": fname,
        "path": str(fpath.resolve()),
    })


@app.route("/train", methods=["POST"])
def start_training():
    """
    Start fine-tuning F5-TTS on a dataset.

    JSON body:
      - dataset_path: path to folder with wav files + metadata.csv
      - epochs: number of training epochs (default 100)
      - batch_size: batch size (default 2)
      - voice_name: name for the resulting voice
    """
    with _train_lock:
        if _train_status["running"]:
            return jsonify({
                "success": False,
                "message": "Training is already in progress",
                "status": _train_status,
            }), 409

    data = request.get_json(force=True, silent=True) or {}
    dataset_path = data.get("dataset_path", "").strip()
    epochs = int(data.get("epochs", 100))
    batch_size = int(data.get("batch_size", 2))
    voice_name = data.get("voice_name", f"trained_{int(time.time())}")

    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({
            "success": False,
            "message": f"Dataset path không tồn tại: {dataset_path}"
        }), 400

    def _run_training():
        with _train_lock:
            _train_status.update({
                "running": True,
                "progress": "Initializing...",
                "error": None,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
            })

        try:
            # Use f5-tts finetune CLI
            cmd = [
                sys.executable, "-m", "f5_tts.train.finetune_cli",
                "--dataset", dataset_path,
                "--epochs", str(epochs),
                "--batch_size", str(batch_size),
                "--output", str(WEIGHTS_DIR / voice_name),
                "--pretrained", str(WEIGHTS_DIR / "model_last.pt"),
            ]

            with _train_lock:
                _train_status["progress"] = f"Training {voice_name} ({epochs} epochs)..."

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200  # 2h max
            )

            if result.returncode == 0:
                # Register trained voice
                voices = _load_voices()
                vid = f"trained_{int(time.time())}"
                voices[vid] = {
                    "name": voice_name,
                    "desc": f"Fine-tuned {epochs} epochs on {dataset_path}",
                    "ref_audio": f"voices/{vid}/ref.wav",
                    "ref_text": "",
                    "type": "trained",
                    "checkpoint": f"weights/{voice_name}",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                _save_voices(voices)
                _git_sync_async(f"Add trained voice: {voice_name}")

                with _train_lock:
                    _train_status.update({
                        "running": False,
                        "progress": f"Completed: {voice_name}",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    })
            else:
                with _train_lock:
                    _train_status.update({
                        "running": False,
                        "error": result.stderr[:500] if result.stderr else "Unknown error",
                        "progress": "Failed",
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    })

        except Exception as e:
            with _train_lock:
                _train_status.update({
                    "running": False,
                    "error": str(e),
                    "progress": "Failed",
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                })

    threading.Thread(target=_run_training, daemon=True).start()
    return jsonify({
        "success": True,
        "message": f"Training started for '{voice_name}'",
        "taskId": voice_name,
    })


@app.route("/train/status", methods=["GET"])
def train_status():
    """Return current training status."""
    return jsonify(_train_status)


@app.route("/sync", methods=["POST"])
def sync_to_github():
    """Manually trigger a git sync to GitHub."""
    message = (request.get_json(silent=True) or {}).get("message", "Manual sync")
    _git_sync_async(message)
    return jsonify({"success": True, "message": "Sync started"})


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Voice AI - F5-TTS Local Server")
    print("  Port: 9880 | Engine: F5-TTS Vietnamese (1000h)")
    print("  GPU: RTX 4060 | Mode: Zero-Shot & Fine-tune")
    print("=" * 60)

    # Check if weights exist
    ckpt = WEIGHTS_DIR / "model_last.pt"
    vocab = WEIGHTS_DIR / "vocab.txt"
    if not ckpt.exists() or not vocab.exists():
        print(f"\n[WARNING] Model weights not found in {WEIGHTS_DIR}/")
        print("Run: py -3.10 setup.py  to download weights\n")
    else:
        print(f"[OK] Model weights found: {ckpt.stat().st_size / 1e9:.2f} GB")

    app.run(host="127.0.0.1", port=9880, threaded=True, debug=False)
