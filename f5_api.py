"""
Voice AI - Vira-TTS Local API Server (v2.1.0)
=============================================
High-quality multi-lingual voice cloning and synthesis powered by Vira-TTS (MiraTTS).
Runs on port 9880, integrates with OpenClaw tab on web-mcbooks-export.

Features:
  - Dynamic pause insertion (breathing gaps) based on punctuation (commas, periods, etc.)
  - Safe Vietnamese text normalization (spells out numbers/dates but preserves punctuation)
  - Automatic intermediate file cleanup (after sending or after 30 minutes)
  - Zero-shot voice cloning and preset voices (37 distinct voices)
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
import re
from pathlib import Path
from datetime import datetime, timezone

# Force UTF-8 stdout
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
# Vira-TTS Model (lazy loaded)
# ──────────────────────────────────────────────
_mira_tts = None
_mira_lock = threading.Lock()

# Training state (F5-TTS training fallback)
_train_status = {
    "running": False,
    "progress": "",
    "error": None,
    "started_at": None,
    "finished_at": None,
}
_train_lock = threading.Lock()


def _get_mira_tts():
    """Lazy-load Vira-TTS model on first request."""
    global _mira_tts
    if _mira_tts is not None:
        return _mira_tts

    with _mira_lock:
        if _mira_tts is not None:
            return _mira_tts

        print("[Voice AI] Loading Vira-TTS model (dolly-vn/Vira-TTS)...")
        
        # Bypass Windows CUDA_PATH assertion by pointing to a dummy path with /bin
        dummy_cuda = os.path.join(str(ROOT), "dummy_cuda")
        os.makedirs(os.path.join(dummy_cuda, "bin"), exist_ok=True)
        os.environ["CUDA_PATH"] = dummy_cuda
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

        try:
            from mira.model import MiraTTS
            # We specify tp=1 for single GPU, and cache_max_entry_count=0.1 to save VRAM on RTX 4060
            _mira_tts = MiraTTS(model_dir='dolly-vn/Vira-TTS', tp=1, cache_max_entry_count=0.1)
            print("[Voice AI] Vira-TTS model loaded successfully!")
        except Exception as e:
            print(f"[Voice AI] Error loading Vira-TTS model: {e}")
            raise e

        return _mira_tts


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
# Safe Text Normalization (preserving punctuation)
# ──────────────────────────────────────────────
def normalize_text_safe(text):
    """Normalize numbers, dates, abbreviations, but preserve sentence punctuation."""
    punc_map = {
        ",": " _comma_ ",
        ".": " _period_ ",
        "?": " _question_ ",
        "!": " _exclamation_ ",
        ";": " _semicolon_ ",
        ":": " _colon_ "
    }
    original_text = text
    try:
        # 1. Replace punctuation with placeholders
        for p, placeholder in punc_map.items():
            text = text.replace(p, placeholder)
            
        from mira.utils import normalize_vietnamese, punc_norm
        # 2. Run standard normalizer (expands numbers, dates, etc.)
        norm_text = normalize_vietnamese(text)
        norm_text = punc_norm(norm_text)
        
        # 3. Restore punctuation
        for p, placeholder in punc_map.items():
            norm_text = re.sub(rf'\s*{placeholder.strip()}\s*', f'{p} ', norm_text)
            
        # 4. Clean up spacing
        norm_text = re.sub(r'\s+', ' ', norm_text).strip()
        norm_text = re.sub(r'\s+([.,!?;:])', r'\1', norm_text)
        norm_text = re.sub(r'([.,!?;:])(?=[^\s\d])', r'\1 ', norm_text)
        
        return norm_text.strip()
    except Exception as e:
        print(f"[Voice AI] Safe normalization failed, using fallback: {e}")
        return original_text


# ──────────────────────────────────────────────
# Trash File Cleanup (Background & On-close)
# ──────────────────────────────────────────────
def _cleanup_loop():
    """Background loop to clean up output & upload files older than 30 minutes."""
    while True:
        try:
            now = time.time()
            # Clean output/
            for f in OUTPUT_DIR.glob("*"):
                if f.is_file() and now - f.stat().st_mtime > 1800: # 30 mins
                    try:
                        f.unlink()
                        print(f"[Voice AI] Auto-cleaned old output file: {f.name}")
                    except Exception:
                        pass
            # Clean uploads/
            for f in UPLOADS_DIR.glob("*"):
                if f.is_file() and now - f.stat().st_mtime > 1800: # 30 mins
                    try:
                        f.unlink()
                        print(f"[Voice AI] Auto-cleaned old upload file: {f.name}")
                    except Exception:
                        pass
        except Exception as e:
            print(f"[Voice AI] Error in cleanup loop: {e}")
        time.sleep(600) # every 10 mins


# Start background cleanup thread
threading.Thread(target=_cleanup_loop, daemon=True).start()


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


def change_speed(wav_data, speed, sr=48000):
    """Change audio speed preserving pitch using librosa."""
    if abs(speed - 1.0) < 0.05:
        return wav_data
    try:
        import librosa
        return librosa.effects.time_stretch(wav_data, rate=speed)
    except Exception as e:
        print(f"[Voice AI] Warning: failed to stretch speed with librosa: {e}")
        return wav_data


# ──────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "name": "Voice AI",
        "engine": "Vira-TTS (MiraTTS)",
        "version": "2.1.0",
        "status": "running",
        "model_loaded": _mira_tts is not None,
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
    result.sort(key=lambda x: (0 if x["type"] == "preset" else 1, x["name"]))
    return jsonify(result)


@app.route("/voices", methods=["POST"])
def save_voice():
    """
    Save a new voice to the library.
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
    Generate speech from text with dynamic dynamic pause insertion.
    """
    import soundfile as sf
    import torch

    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    mode = data.get("mode", "preset")
    voice_id = data.get("voice", "nu6")
    speed = float(data.get("speed", data.get("speed_factor", 1.0)))
    output_path = data.get("outputPath", "")

    if not text:
        return jsonify({"success": False, "message": "Thiếu văn bản cần đọc"}), 400

    try:
        mira_tts = _get_mira_tts()
    except Exception as e:
        return jsonify({"success": False, "message": f"Lỗi tải model: {e}"}), 500

    # Determine reference audio
    if mode == "fast":
        ref_audio_path = data.get("ref_audio_path", "")
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

        if not os.path.exists(ref_audio_path):
            return jsonify({"success": False, "message": f"File audio mẫu không tồn tại: {ref_audio_path}"}), 404

    print(f"[Voice AI] TTS request: mode={mode}, voice={voice_id}, text={text[:60]}...")

    try:
        # 1. Encode reference audio
        context_tokens = mira_tts.encode_audio(ref_audio_path)

        # 2. Normalize text safely (spells out numbers but keeps punctuation)
        norm_text = normalize_text_safe(text)
        if not norm_text:
            return jsonify({"success": False, "message": "Văn bản không hợp lệ"}), 400

        # 3. Split by clause boundaries and sentence enders
        parts = re.split(r'([,;:!?]|\.\.\.|\.)', norm_text)
        segments = []
        current_text = ""
        for part in parts:
            if not part:
                continue
            if part in {",", ";", ":", ".", "!", "?", "..."}:
                if current_text.strip():
                    segments.append((current_text.strip() + part, part))
                    current_text = ""
                else:
                    if segments:
                        last_seg, last_punc = segments[-1]
                        segments[-1] = (last_seg + part, part)
            else:
                current_text += part
                
        if current_text.strip():
            segments.append((current_text.strip(), ""))

        # 4. Synthesize each segment and insert dynamic pauses
        audios = []
        sample_rate = 48000

        for seg_text, punc in segments:
            seg_text = seg_text.strip()
            if not seg_text:
                continue
                
            print(f"[Voice AI] Generating clause: '{seg_text}'")
            # Vira-TTS generate is extremely fast and high-quality
            audio_seg = mira_tts.generate(seg_text, context_tokens)
            
            # Determine pause duration (silence) based on punctuation
            if punc in {",", ";", ":"}:
                pause_secs = 0.25 # Short breathing pause
            elif punc in {".", "!", "?", "..."}:
                pause_secs = 0.55 # Sentence pause
            else:
                pause_secs = 0.40 # Default pause
                
            silence_len = int(pause_secs * sample_rate)
            silence = torch.zeros(silence_len, device=audio_seg.device, dtype=audio_seg.dtype)
            
            # Append silence
            combined_seg = torch.cat([audio_seg, silence], dim=0)
            audios.append(combined_seg)

        if not audios:
            return jsonify({"success": False, "message": "Không thể tạo giọng nói từ văn bản này"}), 400

        # 5. Concatenate all segments
        wav_tensor = torch.cat(audios, dim=0)
        wav_data = wav_tensor.float().cpu().numpy()

        # 6. Apply speed change if requested
        if abs(speed - 1.0) >= 0.05:
            wav_data = change_speed(wav_data, speed, sample_rate)

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
        
        # Stream file and delete immediately on response close
        response = send_file(
            tmp.name,
            mimetype="audio/wav",
            as_attachment=False,
            download_name="voice_ai_output.wav"
        )
        
        @response.call_on_close
        def cleanup_temp_file():
            try:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)
                    print(f"[Voice AI] Cleaned up temporary session file: {tmp.name}")
            except Exception as e:
                print(f"[Voice AI] Error cleaning up temporary file: {e}")
                
        return response

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
    """
    with _train_lock:
        if _train_status["running"]:
            return jsonify({
                "success": False,
                "message": "Huấn luyện đang diễn ra",
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
                voices = _load_voices()
                vid = f"trained_{int(time.time())}"
                
                # Copy a sample wav file from dataset to use as zero-shot reference
                sample_wav = None
                try:
                    for root_dir, _, files in os.walk(dataset_path):
                        for f in files:
                            if f.endswith(".wav"):
                                sample_wav = os.path.join(root_dir, f)
                                break
                        if sample_wav:
                            break
                except Exception:
                    pass

                voice_dir = VOICES_DIR / vid
                voice_dir.mkdir(exist_ok=True)
                ref_path = voice_dir / "ref.wav"

                if sample_wav and os.path.exists(sample_wav):
                    shutil.copy2(sample_wav, str(ref_path))
                else:
                    shutil.copy2(str(VOICES_DIR / "nu6" / "ref.wav"), str(ref_path))

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
                        "error": result.stderr[:500] if result.stderr else "Unknown error during F5 training",
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
    print("  Voice AI - Vira-TTS Local Server")
    print("  Port: 9880 | Engine: Vira-TTS (Multilingual)")
    print("  GPU: RTX 4060 | Mode: Zero-Shot & Fine-tune")
    print("=" * 60)

    app.run(host="127.0.0.1", port=9880, threaded=True, debug=False)
