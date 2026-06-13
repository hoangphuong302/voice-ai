"""
Voice AI - Local TTS Server (v1.1)
=====================================
High-quality multi-lingual voice cloning and synthesis.
Runs on port 9880, integrates with OpenClaw tab on web-mcbooks-export.

Features:
  - Studio-grade audio post-processing pipeline (denoise, normalize, crossfade)
  - Dynamic pause insertion with natural noise floor
  - Safe Vietnamese text normalization (spells out numbers/dates but preserves punctuation)
  - Automatic intermediate file cleanup (after sending or after 30 minutes)
  - Zero-shot voice cloning and preset voices
  - Multi-engine: Gwen-TTS (Vietnamese), Fish Speech (Multilingual), VieNeu-TTS, MiniMax Cloud
"""

VOICE_AI_VERSION = "1.3"

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
import requests
import numpy as np
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
# Auto-Start & Auto-Shutdown Backends Registry
# ──────────────────────────────────────────────
BACKENDS = {
    "gwen-tts": {
        "port": 8081,
        "cmd": [str(ROOT / "gwen-tts" / ".venv" / "Scripts" / "python.exe"), str(ROOT / "gwen-tts" / "api_server.py")],
        "cwd": str(ROOT / "gwen-tts"),
        "process": None,
        "last_active": 0.0,
        "health_url": "http://127.0.0.1:8081/health",
    },
    "xtts-vi": {
        "port": 8082,
        "cmd": [str(ROOT / "xtts-vi" / ".venv" / "Scripts" / "python.exe"), str(ROOT / "xtts-vi" / "api_server.py")],
        "cwd": str(ROOT / "xtts-vi"),
        "process": None,
        "last_active": 0.0,
        "health_url": "http://127.0.0.1:8082/health",
    }
}
_backends_lock = threading.Lock()
IDLE_SHUTDOWN_TIMEOUT = 600.0 # 10 minutes in seconds

def is_port_listening(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.connect(("127.0.0.1", port))
            return True
        except socket.error:
            return False

def ensure_backend_running(name):
    """Start backend server automatically if not running, and update active timestamp."""
    if name not in BACKENDS:
        return
    
    b = BACKENDS[name]
    with _backends_lock:
        b["last_active"] = time.time()
        
        # Check if port is already active (could be started manually or previously)
        if is_port_listening(b["port"]):
            return
            
        print(f"[Voice AI] Auto-starting backend model server: {name}...")
        
        # Start the process in a new session/process group so it detaches properly
        import subprocess
        creationflags = 0
        if os.name == 'nt':
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
            
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        
        proc = subprocess.Popen(
            b["cmd"],
            cwd=b["cwd"],
            env=env,
            creationflags=creationflags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        b["process"] = proc
        
        # Poll health endpoint until online
        t0 = time.time()
        max_wait = 45.0 # allow 45s for heavy models to load on CUDA
        online = False
        
        while time.time() - t0 < max_wait:
            if proc.poll() is not None:
                raise RuntimeError(f"Backend process {name} exited early with code {proc.poll()}")
                
            try:
                resp = requests.get(b["health_url"], timeout=1)
                if resp.status_code == 200:
                    online = True
                    break
            except Exception:
                pass
            time.sleep(1.0)
            
        if not online:
            try: proc.kill()
            except Exception: pass
            b["process"] = None
            raise TimeoutError(f"Backend server {name} failed to become healthy on port {b['port']} within {max_wait}s")
            
        print(f"[Voice AI] Backend server {name} is online and healthy!")

def monitor_backends_loop():
    """Background thread to monitor and terminate idle backends."""
    while True:
        try:
            time.sleep(30.0) # check every 30 seconds
            now = time.time()
            with _backends_lock:
                for name, b in BACKENDS.items():
                    if is_port_listening(b["port"]):
                        if b["last_active"] > 0 and (now - b["last_active"] > IDLE_SHUTDOWN_TIMEOUT):
                            print(f"[Voice AI] Engine {name} has been idle for {IDLE_SHUTDOWN_TIMEOUT}s. Shutting down to free VRAM/RAM...")
                            
                            if b["process"] is not None:
                                try:
                                    b["process"].terminate()
                                    b["process"].wait(timeout=3)
                                except Exception:
                                    try: b["process"].kill()
                                    except Exception: pass
                                b["process"] = None
                            else:
                                if os.name == 'nt':
                                    try:
                                        cmd = f'for /f "tokens=5" %a in (\'netstat -aon ^| findstr :{b["port"]} ^| findstr LISTENING\') do taskkill /f /pid %a'
                                        subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                                    except Exception as e:
                                        print(f"[Voice AI] Error killing idle port {b['port']}: {e}")
                            
                            b["last_active"] = 0.0
                            print(f"[Voice AI] Engine {name} successfully shut down.")
        except Exception as e:
            print(f"[Voice AI] Error in backend monitor loop: {e}")

# Start background monitor thread
threading.Thread(target=monitor_backends_loop, daemon=True).start()

# Register exit handler to clean up child processes
import atexit
def cleanup_child_backends():
    with _backends_lock:
        for name, b in BACKENDS.items():
            if b["process"] is not None:
                print(f"[Voice AI] Terminating {name} child process on shutdown...")
                try:
                    b["process"].terminate()
                    b["process"].wait(timeout=3)
                except Exception:
                    try: b["process"].kill()
                    except Exception: pass
atexit.register(cleanup_child_backends)

# ──────────────────────────────────────────────

# Vira-TTS & VieNeu-TTS Models (lazy loaded)
# ──────────────────────────────────────────────
_mira_tts = None
_vieneu_tts = None
_viterbox_tts = None
_current_model = None
_engine_lock = threading.Lock()

# Training state (F5-TTS training fallback)
_train_status = {
    "running": False,
    "progress": "",
    "error": None,
    "started_at": None,
    "finished_at": None,
}
_train_lock = threading.Lock()


# ──────────────────────────────────────────────
# Whisper Model for Automatic Transcription (lazy loaded on CPU)
# ──────────────────────────────────────────────
_whisper_model = None
_whisper_lock = threading.Lock()

def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                print("[Voice AI] Loading faster-whisper model (base)...")
                from faster_whisper import WhisperModel
                # Force CPU execution to prevent GPU VRAM issues
                _whisper_model = WhisperModel("base", device="cpu", compute_type="float32")
                print("[Voice AI] faster-whisper model loaded successfully!")
    return _whisper_model

def transcribe_audio(audio_path):
    """Transcribe reference audio to text using faster-whisper."""
    if not audio_path or not os.path.exists(audio_path):
        return ""
    try:
        model = _get_whisper_model()
        print(f"[Voice AI] Auto-transcribing reference audio: {audio_path}...")
        t0 = time.time()
        segments, info = model.transcribe(audio_path, beam_size=5, language="vi")
        text = " ".join([seg.text for seg in segments]).strip()
        dt = time.time() - t0
        print(f"[Voice AI] Auto-transcribed in {dt:.2f}s: '{text}'")
        return text
    except Exception as e:
        print(f"[Voice AI] Auto-transcription failed: {e}")
        return ""


def _get_engine(model_name="dolly-vn/Vira-TTS"):
    """Get active engine, dynamically swapping if needed and cleaning VRAM."""
    global _mira_tts, _vieneu_tts, _viterbox_tts, _current_model
    
    if _current_model == model_name:
        if model_name == "dolly-vn/Vira-TTS" and _mira_tts is not None:
            return _mira_tts
        elif model_name == "pnnbao-ump/VieNeu-TTS-v3-Turbo" and _vieneu_tts is not None:
            return _vieneu_tts
        elif model_name == "dolly-vn/viterbox" and _viterbox_tts is not None:
            return _viterbox_tts

    with _engine_lock:
        if _current_model == model_name:
            if model_name == "dolly-vn/Vira-TTS" and _mira_tts is not None:
                return _mira_tts
            elif model_name == "pnnbao-ump/VieNeu-TTS-v3-Turbo" and _vieneu_tts is not None:
                return _vieneu_tts
            elif model_name == "dolly-vn/viterbox" and _viterbox_tts is not None:
                return _viterbox_tts

        import gc
        import torch

        if model_name == "dolly-vn/Vira-TTS":
            if _vieneu_tts is not None:
                print("[Voice AI] Unloading VieNeu-TTS-v3-Turbo model...")
                try:
                    _vieneu_tts.close()
                except Exception as e:
                    print(f"[Voice AI] Warning closing Vieneu: {e}")
                _vieneu_tts = None
            if _viterbox_tts is not None:
                print("[Voice AI] Unloading Viterbox model...")
                _viterbox_tts = None
            
            gc.collect()
            torch.cuda.empty_cache()

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
                _current_model = "dolly-vn/Vira-TTS"
                print("[Voice AI] Vira-TTS model loaded successfully!")
            except Exception as e:
                print(f"[Voice AI] Error loading Vira-TTS model: {e}")
                raise e

            return _mira_tts

        elif model_name == "pnnbao-ump/VieNeu-TTS-v3-Turbo":
            if _mira_tts is not None:
                print("[Voice AI] Unloading Vira-TTS model...")
                try:
                    if hasattr(_mira_tts, "pipe") and hasattr(_mira_tts.pipe, "close"):
                        _mira_tts.pipe.close()
                except Exception as e:
                    print(f"[Voice AI] Warning closing Vira-TTS pipeline: {e}")
                _mira_tts = None
            if _viterbox_tts is not None:
                print("[Voice AI] Unloading Viterbox model...")
                _viterbox_tts = None

            gc.collect()
            torch.cuda.empty_cache()

            print("[Voice AI] Loading VieNeu-TTS-v3-Turbo model...")
            try:
                from vieneu import Vieneu
                _vieneu_tts = Vieneu()
                _current_model = "pnnbao-ump/VieNeu-TTS-v3-Turbo"
                print("[Voice AI] VieNeu-TTS-v3-Turbo model loaded successfully!")
            except Exception as e:
                print(f"[Voice AI] Error loading VieNeu-TTS-v3-Turbo model: {e}")
                raise e

            return _vieneu_tts

        elif model_name == "dolly-vn/viterbox":
            if _mira_tts is not None:
                print("[Voice AI] Unloading Vira-TTS model...")
                try:
                    if hasattr(_mira_tts, "pipe") and hasattr(_mira_tts.pipe, "close"):
                        _mira_tts.pipe.close()
                except Exception as e:
                    print(f"[Voice AI] Warning closing Vira-TTS pipeline: {e}")
                _mira_tts = None
            if _vieneu_tts is not None:
                print("[Voice AI] Unloading VieNeu-TTS-v3-Turbo model...")
                try:
                    _vieneu_tts.close()
                except Exception as e:
                    print(f"[Voice AI] Warning closing Vieneu: {e}")
                _vieneu_tts = None

            gc.collect()
            torch.cuda.empty_cache()

            print("[Voice AI] Loading Viterbox model (dolly-vn/viterbox)...")
            try:
                from viterbox import Viterbox
                _viterbox_tts = Viterbox.from_pretrained("cuda")
                _current_model = "dolly-vn/viterbox"
                print("[Voice AI] Viterbox model loaded successfully!")
            except Exception as e:
                print(f"[Voice AI] Error loading Viterbox model: {e}")
                raise e

            return _viterbox_tts
            
        else:
            raise ValueError(f"Unknown model name: {model_name}")


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
    original_text = text
    try:
        # 1. Replace punctuation with placeholders (using regex to avoid number separators)
        # We only replace when they are NOT between two digits.
        text = re.sub(r'(?<!\d),|,(?!\d)', ' XYZCOMMAXYZ ', text)
        text = re.sub(r'(?<!\d)\.|\.(?!\d)', ' XYZPERIODXYZ ', text)
        text = re.sub(r'(?<!\d):|:(?!\d)', ' XYZCOLONXYZ ', text)
        
        # Other punctuation can be replaced globally or safely
        text = text.replace("?", " XYZQUESTIONXYZ ")
        text = text.replace("!", " XYZEXCLAMATIONXYZ ")
        text = text.replace(";", " XYZSEMICOLONXYZ ")
            
        from mira.utils import normalize_vietnamese, punc_norm
        # 2. Run standard normalizer (expands numbers, dates, etc.)
        norm_text = normalize_vietnamese(text)
        norm_text = punc_norm(norm_text)
        
        # 3. Restore punctuation
        punc_map = {
            ",": "XYZCOMMAXYZ",
            ".": "XYZPERIODXYZ",
            "?": "XYZQUESTIONXYZ",
            "!": "XYZEXCLAMATIONXYZ",
            ";": "XYZSEMICOLONXYZ",
            ":": "XYZCOLONXYZ"
        }
        
        for p, placeholder in punc_map.items():
            norm_text = re.sub(rf'\s*{placeholder}\s*', f'{p} ', norm_text, flags=re.IGNORECASE)
            
        # 4. Clean up spacing
        norm_text = re.sub(r'\s+', ' ', norm_text).strip()
        norm_text = re.sub(r'\s+([.,!?;:])', r'\1', norm_text)
        norm_text = re.sub(r'([.,!?;:])(?=[^\s\d])', r'\1 ', norm_text)
        
        # Clean up double punctuation (including "? ." and ". .")
        norm_text = re.sub(r'([.,!?;:])\s*[.]', r'\1', norm_text)
        
        return re.sub(r'\s+', ' ', norm_text).strip()
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
    """Change audio speed preserving pitch. Uses rubberband (best quality) with fallback."""
    # Clamp speed to safe range
    if speed <= 0:
        speed = 1.0
    speed = max(0.25, min(4.0, speed))
    if abs(speed - 1.0) < 0.05:
        return wav_data

    # Method 1: pyrubberband (highest quality - WSOLA algorithm)
    try:
        import pyrubberband as pyrb
        result = pyrb.time_stretch(wav_data, sr, speed)
        print(f"[Voice AI] Speed changed to {speed}x using rubberband (HQ)")
        return result
    except ImportError:
        pass
    except Exception as e:
        print(f"[Voice AI] rubberband failed: {e}")

    # Method 2: scipy-based high-quality resampling
    # Resample to change speed, then pitch-correct via resampling back
    try:
        from scipy.signal import resample
        import numpy as np
        n_samples = len(wav_data)
        # Stretch/compress the audio by resampling
        n_target = int(n_samples / speed)
        stretched = resample(wav_data, n_target)
        print(f"[Voice AI] Speed changed to {speed}x using scipy resample")
        return stretched.astype(np.float32)
    except Exception as e:
        print(f"[Voice AI] scipy resample failed: {e}")

    # Method 3: librosa with better parameters (larger n_fft = less artifact)
    try:
        import librosa
        result = librosa.effects.time_stretch(wav_data, rate=speed, n_fft=4096)
        print(f"[Voice AI] Speed changed to {speed}x using librosa (n_fft=4096)")
        return result
    except Exception as e:
        print(f"[Voice AI] Warning: all speed methods failed: {e}")
        return wav_data


def detect_language(text):
    """
    Auto-detect language from text content.
    Returns 'vi' for Vietnamese, 'en' for English, or detected language code.
    """
    import re

    # Vietnamese diacritical marks (tonal marks + special letters)
    vi_chars = set('àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ'
                   'ÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ')

    # Count Vietnamese vs non-Vietnamese characters
    text_chars = re.sub(r'[\s\d\W]', '', text)  # Remove spaces, digits, punctuation
    if not text_chars:
        return 'vi'  # Default to Vietnamese

    vi_count = sum(1 for c in text_chars if c in vi_chars)
    total = len(text_chars)
    vi_ratio = vi_count / total if total > 0 else 0

    # If more than 5% of characters are Vietnamese diacritics → Vietnamese
    if vi_ratio > 0.05:
        return 'vi'

    # Check for common Vietnamese words without diacritics
    vi_words = {'cua', 'trong', 'ngoai', 'la', 'nhung', 'duoc', 'khong', 'nhu', 'voi',
                'cac', 'mot', 'hai', 'ba', 'bon', 'nam', 'sau', 'bay', 'tam', 'chin', 'muoi'}
    words_lower = set(text.lower().split())
    vi_word_hits = len(words_lower & vi_words)

    if vi_word_hits >= 2:
        return 'vi'

    # Default: if text is mostly ASCII letters → English
    ascii_count = sum(1 for c in text_chars if c.isascii())
    if ascii_count / total > 0.9:
        return 'en'

    return 'vi'  # Default to Vietnamese


# ──────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    active_name = "Vira-TTS" if _current_model == "dolly-vn/Vira-TTS" else ("Viterbox" if _current_model == "dolly-vn/viterbox" else "VieNeu-TTS-v3-Turbo")
    return jsonify({
        "name": "Voice AI",
        "engine": active_name,
        "version": "3.0.0",
        "status": "running",
        "model_loaded": (_mira_tts is not None or _vieneu_tts is not None or _viterbox_tts is not None),
        "active_model": _current_model or "None"
    })


def _load_minimax_key():
    # 1. Try env variable
    key = os.environ.get("MINIMAX_API_KEY", "").strip()
    if key:
        return key

    # 2. Try minimax_key.txt
    for path_str in ["D:/voice-ai/minimax_key.txt", "C:/Workspace/minimax_key.txt"]:
        key_path = Path(path_str)
        if key_path.exists():
            try:
                val = key_path.read_text(encoding="utf-8").strip()
                if val:
                    return val
            except Exception:
                pass

    # 3. Try openclaw config profiles
    profiles_path = Path("C:/Users/phuong 1.2024/Desktop/antigravity/TELE_auth-profiles.json")
    if profiles_path.exists():
        try:
            profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
            key = profiles.get("profiles", {}).get("minimax-portal:default", {}).get("access", "").strip()
            if key:
                return key
        except Exception:
            pass

    return None


@app.route("/models", methods=["GET"])
def list_models():
    return jsonify([
        {
            "id": "dolly-vn/viterbox",
            "name": "⭐ Viterbox (24→48kHz, Biểu cảm)",
            "desc": "Chất lượng cao nhất! Dựa trên Chatterbox, train 3000h+ tiếng Việt, kiểm soát biểu cảm"
        },
        {
            "id": "dolly-vn/Vira-TTS",
            "name": "Vira-TTS (48kHz)",
            "desc": "Giọng đọc truyền cảm, chuẩn tiếng Việt & Anh"
        },
        {
            "id": "pnnbao-ump/VieNeu-TTS-v3-Turbo",
            "name": "VieNeu-TTS-v3-Turbo (48kHz)",
            "desc": "Mô hình nhẹ, hỗ trợ biểu cảm & giọng preset đa dạng"
        },
        {
            "id": "minimax/speech-01-turbo",
            "name": "MiniMax Speech-01-Turbo (Cloud API)",
            "desc": "API đám mây từ MiniMax, chất lượng cao, phản hồi nhanh"
        },
        {
            "id": "minimax/speech-02-hd",
            "name": "MiniMax Speech-02-HD (Premium Cloud)",
            "desc": "Mô hình HD cao cấp nhất từ MiniMax, biểu cảm tự nhiên vượt trội"
        }
    ])


@app.route("/voices", methods=["GET"])
def list_voices():
    """Return all saved voices."""
    voices = _load_voices()
    result = []
    for vid, v in voices.items():
        ref_audio_rel = v.get("ref_audio", "")
        has_preview = False
        if ref_audio_rel:
            ref_path = ROOT / ref_audio_rel
            has_preview = ref_path.exists() and ref_path.is_file()
        result.append({
            "id": vid,
            "name": v.get("name", vid),
            "desc": v.get("desc", ""),
            "type": v.get("type", "preset"),
            "model": v.get("model", ""),
            "created_at": v.get("created_at", ""),
            "has_preview": has_preview,
        })
    result.sort(key=lambda x: (0 if x["type"] == "preset" else 1, x["name"]))
    return jsonify(result)


@app.route("/voices/<voice_id>/preview", methods=["GET"])
def preview_voice(voice_id):
    """Return the reference audio sample for a voice (for preview/listening)."""
    voices = _load_voices()
    if voice_id not in voices:
        return jsonify({"success": False, "error": "Voice not found"}), 404

    voice = voices[voice_id]
    ref_audio_rel = voice.get("ref_audio", "")
    if not ref_audio_rel:
        return jsonify({"success": False, "error": "No preview audio available"}), 404

    ref_path = ROOT / ref_audio_rel
    if not ref_path.exists() or not ref_path.is_file():
        return jsonify({"success": False, "error": "Preview audio file not found"}), 404

    return send_file(
        str(ref_path),
        mimetype="audio/wav",
        as_attachment=False,
        download_name=f"preview_{voice_id}.wav"
    )


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

    # Auto-transcribe if empty
    if not ref_text.strip():
        ref_text = transcribe_audio(str(ref_path))

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


@app.route("/version", methods=["GET"])
def version():
    """Return current Voice AI version."""
    return jsonify({
        "version": VOICE_AI_VERSION,
        "engine": "XTTS-v2-Vietnamese (941h PhoAudiobook)",
        "features": [
            "Sentence splitting for stable long text",
            "Dynamic decoding parameters (temperature, repetition penalty)",
            "Vietnamese text preprocessing",
            "Distortion-free linear peak normalization",
        ]
    })


@app.route("/tts", methods=["POST"])
def text_to_speech():
    """
    Generate speech from text with dynamic dynamic pause insertion and multiple models.
    """
    import soundfile as sf
    import torch

    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "").strip()
    mode = data.get("mode", "preset")
    voice_id = data.get("voice", "nu6")
    try:
        speed = float(data.get("speed", data.get("speed_factor", 1.0)))
        if speed <= 0:
            speed = 1.0
        speed = max(0.25, min(4.0, speed))
    except (ValueError, TypeError):
        speed = 1.0
    output_path = data.get("outputPath", "")

    if not text:
        return jsonify({"success": False, "message": "Thiếu văn bản cần đọc"}), 400

    # Determine reference voice config
    voices = _load_voices()
    voice_cfg = voices.get(voice_id, {})
    
    # Auto-resolve model from voice if it's model-specific
    voice_model = voice_cfg.get("model")
    requested_model = voice_model if voice_model else data.get("model", "xtts-vi")

    if requested_model == "minimax":
        client_model = data.get("model", "")
        if client_model.startswith("minimax/"):
            requested_model = client_model
        else:
            requested_model = "minimax/speech-01-turbo"

    # ── Early validation BEFORE loading models (avoids 30-60s model load for invalid requests) ──
    if mode != "fast" and not requested_model.startswith("minimax/") and requested_model not in ("fish-speech-1.5", "gwen-tts", "xtts-vi"):
        # For preset/saved voice mode: validate voice exists before loading model
        if requested_model == "dolly-vn/Vira-TTS":
            if voice_id not in voices:
                return jsonify({"success": False, "message": f"Voice '{voice_id}' không tồn tại"}), 404
            ref_audio_rel = voice_cfg.get("ref_audio", "")
            ref_audio_check = str(ROOT / ref_audio_rel) if ref_audio_rel else ""
            if not ref_audio_check or not os.path.exists(ref_audio_check) or os.path.isdir(ref_audio_check):
                return jsonify({"success": False, "message": f"File audio mẫu không tồn tại hoặc không hợp lệ: {ref_audio_check}"}), 404
    elif mode == "fast" and not requested_model.startswith("minimax/"):
        # For fast clone mode: validate ref_audio_path exists before loading model
        ref_audio_path_check = data.get("ref_audio_path", "")
        if requested_model in ("dolly-vn/Vira-TTS", "pnnbao-ump/VieNeu-TTS-v3-Turbo"):
            if not ref_audio_path_check or not os.path.exists(ref_audio_path_check) or os.path.isdir(ref_audio_path_check):
                return jsonify({"success": False, "message": "Không tìm thấy file âm thanh mẫu hoặc đường dẫn không hợp lệ"}), 400

    if not requested_model.startswith("minimax/") and requested_model not in ("fish-speech-1.5", "gwen-tts", "xtts-vi"):
        try:
            engine = _get_engine(requested_model)
        except Exception as e:
            return jsonify({"success": False, "message": f"Lỗi tải model: {e}"}), 500


    print(f"[Voice AI] TTS request: model={requested_model}, mode={mode}, voice={voice_id}, text={text[:60]}...")

    # ────────────────────────────────────────────────────────
    # MiniMax-TTS Synthesis Path (Cloud API)
    # ────────────────────────────────────────────────────────
    if requested_model.startswith("minimax/"):
        minimax_model_id = requested_model.split("/", 1)[1] # "speech-01-turbo" or "speech-02-hd"
        api_key = _load_minimax_key()
        if not api_key:
            return jsonify({
                "success": False,
                "message": "Chưa cấu hình API Key cho MiniMax. Vui lòng ghi key vào file D:\\voice-ai\\minimax_key.txt hoặc thiết lập biến môi trường MINIMAX_API_KEY."
            }), 400

        # Resolve voice ID from preset or use as-is
        voice_id_val = voice_cfg.get("voice_id", voice_id) # fallback to voice_id itself

        print(f"[Voice AI] Synthesizing with MiniMax Cloud API: model={minimax_model_id}, voice={voice_id_val}, text={text[:60]}...")
        
        try:
            url = "https://api.minimax.io/v1/t2a_v2"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": minimax_model_id,
                "text": text,
                "voice_setting": {
                    "voice_id": voice_id_val,
                    "speed": 1,
                    "vol": 1,
                    "pitch": 0
                },
                "audio_setting": {
                    "format": "mp3",
                    "audio_sample_rate": 32000,
                    "bitrate": 128000,
                    "channel": 1
                }
            }

            res = requests.post(url, headers=headers, json=payload, timeout=30)
            if res.status_code != 200:
                return jsonify({"success": False, "message": f"Lỗi gọi API MiniMax: HTTP {res.status_code} - {res.text}"}), 500

            content_type = res.headers.get("Content-Type", "")
            if "application/json" in content_type:
                err_data = res.json()
                status_msg = err_data.get("base_resp", {}).get("status_msg", "Unknown error")
                return jsonify({"success": False, "message": f"MiniMax API trả về lỗi: {status_msg}"}), 400

            # Save MP3 content
            tmp_mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False, dir=str(OUTPUT_DIR))
            tmp_mp3.write(res.content)
            tmp_mp3.close()

            import librosa
            import soundfile as sf
            
            try:
                wav_data, sample_rate = librosa.load(tmp_mp3.name, sr=32000)
                # Apply speed change if requested
                if abs(speed - 1.0) >= 0.05:
                    wav_data = change_speed(wav_data, speed, sample_rate)
                
                # Save to WAV
                tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(OUTPUT_DIR))
                sf.write(tmp_wav.name, wav_data, sample_rate)
                tmp_wav.close()
                
                final_tmp_path = tmp_wav.name
                mimetype_val = "audio/wav"
                download_name_val = "voice_ai_output.wav"
                
                try: os.unlink(tmp_mp3.name)
                except Exception: pass
            except Exception as e:
                print(f"[Voice AI] Warning: failed to convert MiniMax MP3 to WAV: {e}. Returning raw MP3.")
                final_tmp_path = tmp_mp3.name
                mimetype_val = "audio/mpeg"
                download_name_val = "voice_ai_output.mp3"

            # Save to outputPath if specified
            if output_path and output_path.strip():
                try:
                    os.makedirs(output_path.strip(), exist_ok=True)
                    final_name = os.path.join(
                        output_path.strip(),
                        f"voice_ai_{voice_id}_{int(time.time())}.wav" if mimetype_val == "audio/wav" else f"voice_ai_{voice_id}_{int(time.time())}.mp3"
                    )
                    shutil.copy2(final_tmp_path, final_name)
                    print(f"[Voice AI] MiniMax Output saved to: {final_name}")
                except Exception as e:
                    print(f"[Voice AI] Could not save MiniMax to outputPath: {e}")

            print(f"[Voice AI] MiniMax Cloud TTS done successfully!")

            response = send_file(
                final_tmp_path,
                mimetype=mimetype_val,
                as_attachment=False,
                download_name=download_name_val
            )

            @response.call_on_close
            def cleanup_temp_file():
                try:
                    if os.path.exists(final_tmp_path):
                        os.unlink(final_tmp_path)
                        print(f"[Voice AI] Cleaned up temporary MiniMax file: {final_tmp_path}")
                except Exception as e:
                    print(f"[Voice AI] Error cleaning up temporary file: {e}")

            return response

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "message": f"Lỗi tổng hợp MiniMax: {str(e)}"}), 500

    # ────────────────────────────────────────────────────────
    # VieNeu-TTS Synthesis Path
    # ────────────────────────────────────────────────────────
    if requested_model == "pnnbao-ump/VieNeu-TTS-v3-Turbo":
        try:
            # 1. Preset voice mode
            if voice_cfg.get("model") == "pnnbao-ump/VieNeu-TTS-v3-Turbo":
                voice_name = voice_cfg.get("voice_name", "Ngọc Lan")
                print(f"[Voice AI] Synthesizing with VieNeu preset: {voice_name}")
                audio_np = engine.infer(text=text, voice=voice_name, temperature=0.65)
            # 2. Fast/Cloned voice mode
            else:
                if mode == "fast":
                    ref_audio_path = data.get("ref_audio_path", "")
                    ref_text_val = data.get("ref_text", "").strip()
                else:
                    ref_audio_rel = voice_cfg.get("ref_audio", "")
                    ref_audio_path = str(ROOT / ref_audio_rel)
                    ref_text_val = voice_cfg.get("ref_text", "").strip()

                if not ref_audio_path or not os.path.exists(ref_audio_path) or os.path.isdir(ref_audio_path):
                    return jsonify({"success": False, "message": "Không tìm thấy file âm thanh mẫu hoặc đường dẫn không hợp lệ"}), 400

                # Auto-transcribe if empty
                if not ref_text_val:
                    ref_text_val = transcribe_audio(ref_audio_path)

                print(f"[Voice AI] Cloning with VieNeu reference: {ref_audio_path} (ref_text: '{ref_text_val}')")
                audio_np = engine.infer(text=text, ref_audio=ref_audio_path, ref_text=ref_text_val or None, temperature=0.65)

            sample_rate = 48000
            if abs(speed - 1.0) >= 0.05:
                audio_np = change_speed(audio_np, speed, sample_rate)

            # Post-processing enhancement
            try:
                from audio_enhance import enhance_audio
                audio_np = enhance_audio(audio_np, sample_rate, target_sr=48000)
                print(f"[Voice AI] VieNeu audio enhanced successfully")
            except Exception as e:
                print(f"[Voice AI] Enhancement skipped: {e}")

            # Save to temp file
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(OUTPUT_DIR))
            sf.write(tmp.name, audio_np, sample_rate)
            tmp.close()

            # Save to outputPath if specified
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

            print(f"[Voice AI] VieNeu TTS done, {len(audio_np)} samples at {sample_rate}Hz")

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
            return jsonify({"success": False, "message": f"Lỗi tổng hợp VieNeu: {str(e)}"}), 500

    # ────────────────────────────────────────────────────────
    # Gwen-TTS Synthesis Path (Qwen3-TTS 0.6B, 1000h Vietnamese)
    # ────────────────────────────────────────────────────────
    if requested_model == "gwen-tts":
        try:
            ensure_backend_running("gwen-tts")
            GWEN_API_URL = "http://127.0.0.1:8081/tts"
            
            # Build reference audio path
            ref_audio_path = None
            ref_text = ""
            
            if mode == "fast":
                ref_audio_path = data.get("ref_audio_path", "")
                ref_text = data.get("ref_text", "")
            else:
                ref_audio_rel = voice_cfg.get("ref_audio", "")
                if ref_audio_rel:
                    ref_audio_path = str(ROOT / ref_audio_rel)
                ref_text = voice_cfg.get("ref_text", "")
            
            gwen_payload = {
                "text": text,
                "ref_audio": ref_audio_path or "",
                "ref_text": ref_text or "Xin chào",
            }
            
            print(f"[Voice AI] Gwen-TTS: text={text[:50]}..., ref={ref_audio_path}")
            
            try:
                resp = requests.post(GWEN_API_URL, json=gwen_payload, timeout=300)
            except requests.exceptions.ConnectionError:
                return jsonify({
                    "success": False,
                    "message": "Gwen-TTS server chưa chạy! Khởi động: D:/voice-ai/gwen-tts/api_server.py"
                }), 503
            
            if resp.status_code != 200:
                err_msg = resp.text[:200] if resp.text else "Unknown error"
                return jsonify({"success": False, "message": f"Gwen-TTS error: {err_msg}"}), 500
            
            audio_data = resp.content
            sample_rate = 24000  # Gwen-TTS native 24kHz
            
            # Apply speed change if requested
            if abs(speed - 1.0) >= 0.05:
                import io
                import soundfile as sf_temp
                audio_np_raw, sr_raw = sf_temp.read(io.BytesIO(audio_data))
                audio_np_raw = change_speed(audio_np_raw, speed, sr_raw)
                buf = io.BytesIO()
                sf_temp.write(buf, audio_np_raw, sr_raw, format='WAV')
                audio_data = buf.getvalue()
                sample_rate = sr_raw
            
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(OUTPUT_DIR))
            tmp.write(audio_data)
            tmp.close()
            
            if output_path and output_path.strip():
                try:
                    os.makedirs(output_path.strip(), exist_ok=True)
                    out_name = f"voice_{voice_id}_{int(time.time())}.wav"
                    out_full = os.path.join(output_path.strip(), out_name)
                    with open(out_full, 'wb') as f:
                        f.write(audio_data)
                except Exception as e:
                    print(f"[Voice AI] Warning: Could not save to outputPath: {e}")
            
            print(f"[Voice AI] Gwen-TTS done, saved to {tmp.name}")
            
            return send_file(
                tmp.name,
                mimetype="audio/wav",
                as_attachment=True,
                download_name="output.wav"
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "message": f"Lỗi Gwen-TTS: {str(e)}"}), 500

    # ────────────────────────────────────────────────────────
    # XTTS-v2 Vietnamese Synthesis Path (941h PhoAudiobook)
    # ────────────────────────────────────────────────────────
    if requested_model == "xtts-vi":
        try:
            ensure_backend_running("xtts-vi")
            XTTS_API_URL = "http://127.0.0.1:8082/tts"
            
            ref_audio_path = None
            ref_text = ""
            
            if mode == "fast":
                ref_audio_path = data.get("ref_audio_path", "")
                ref_text = data.get("ref_text", "")
            else:
                ref_audio_rel = voice_cfg.get("ref_audio", "")
                if ref_audio_rel:
                    ref_audio_path = str(ROOT / ref_audio_rel)
                ref_text = voice_cfg.get("ref_text", "")
            
            xtts_payload = {
                "text": text,
                "ref_audio": ref_audio_path or "",
                "ref_text": ref_text or "",
                "temperature": data.get("temperature", 0.7),
                "repetition_penalty": data.get("repetition_penalty", 2.0),
                "top_k": data.get("top_k", 50),
                "top_p": data.get("top_p", 0.85),
            }
            
            print(f"[Voice AI] XTTS-Vi: text={text[:50]}..., ref={ref_audio_path}")
            
            try:
                resp = requests.post(XTTS_API_URL, json=xtts_payload, timeout=300)
            except requests.exceptions.ConnectionError:
                return jsonify({
                    "success": False,
                    "message": "XTTS-Vi server chưa chạy! Khởi động: D:/voice-ai/xtts-vi/api_server.py"
                }), 503
            
            if resp.status_code != 200:
                err_msg = resp.text[:200] if resp.text else "Unknown error"
                return jsonify({"success": False, "message": f"XTTS-Vi error: {err_msg}"}), 500
            
            audio_data = resp.content
            sample_rate = 22050  # XTTS native 22050Hz
            
            if abs(speed - 1.0) >= 0.05:
                import io
                import soundfile as sf_temp
                audio_np_raw, sr_raw = sf_temp.read(io.BytesIO(audio_data))
                audio_np_raw = change_speed(audio_np_raw, speed, sr_raw)
                buf = io.BytesIO()
                sf_temp.write(buf, audio_np_raw, sr_raw, format='WAV')
                audio_data = buf.getvalue()
                sample_rate = sr_raw
            
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(OUTPUT_DIR))
            tmp.write(audio_data)
            tmp.close()
            
            if output_path and output_path.strip():
                try:
                    os.makedirs(output_path.strip(), exist_ok=True)
                    out_name = f"voice_{voice_id}_{int(time.time())}.wav"
                    out_full = os.path.join(output_path.strip(), out_name)
                    with open(out_full, 'wb') as f:
                        f.write(audio_data)
                except Exception as e:
                    print(f"[Voice AI] Warning: Could not save to outputPath: {e}")
            
            print(f"[Voice AI] XTTS-Vi done, saved to {tmp.name}")
            
            return send_file(
                tmp.name,
                mimetype="audio/wav",
                as_attachment=True,
                download_name="output.wav"
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "message": f"Lỗi XTTS-Vi: {str(e)}"}), 500

    # ────────────────────────────────────────────────────────
    # Fish Speech 1.5 Synthesis Path (multilingual fallback)
    # ────────────────────────────────────────────────────────
    if requested_model == "fish-speech-1.5":
        try:
            import base64
            import ormsgpack
            
            FISH_API_URL = "http://127.0.0.1:8080/v1/tts"
            
            # Build reference audio for voice cloning
            references = []
            ref_audio_path = None
            ref_text = ""
            
            if mode == "fast":
                # Fast clone mode: use uploaded audio
                ref_audio_path = data.get("ref_audio_path", "")
                ref_text = data.get("ref_text", "")
            else:
                # Preset mode: use voice library reference
                ref_audio_rel = voice_cfg.get("ref_audio", "")
                if ref_audio_rel:
                    ref_audio_path = str(ROOT / ref_audio_rel)
                ref_text = voice_cfg.get("ref_text", "")
            
            # Read reference audio bytes
            if ref_audio_path and os.path.exists(ref_audio_path):
                with open(ref_audio_path, "rb") as f:
                    ref_audio_bytes = f.read()
                references.append({
                    "audio": ref_audio_bytes,
                    "text": ref_text or ""
                })
            
            # Fish Speech request payload
            fish_payload = {
                "text": text,
                "references": references,
                "reference_id": None,
                "format": "wav",
                "max_new_tokens": 2048,
                "chunk_length": 300,
                "top_p": 0.8,
                "repetition_penalty": 1.1,
                "temperature": 0.7,
                "streaming": False,
                "use_memory_cache": "on",
                "seed": None,
            }
            
            print(f"[Voice AI] Fish Speech 1.5: text={text[:50]}..., ref={ref_audio_path}")
            
            # Send request to Fish Speech API server
            try:
                packed = ormsgpack.packb(fish_payload)
                resp = requests.post(
                    FISH_API_URL,
                    data=packed,
                    headers={"content-type": "application/msgpack"},
                    timeout=120,
                )
            except requests.exceptions.ConnectionError:
                return jsonify({
                    "success": False, 
                    "message": "Fish Speech server chưa chạy! Hãy khởi động: D:/voice-ai/fish-speech/start_server.bat"
                }), 503
            
            if resp.status_code != 200:
                err_msg = resp.text[:200] if resp.text else "Unknown error"
                return jsonify({"success": False, "message": f"Fish Speech error: {err_msg}"}), 500
            
            # Save response WAV
            sample_rate = 44100  # Fish Speech native 44.1kHz
            audio_data = resp.content
            
            # Apply speed change if requested
            if abs(speed - 1.0) >= 0.05:
                import io
                import soundfile as sf_temp
                audio_np_raw, sr_raw = sf_temp.read(io.BytesIO(audio_data))
                audio_np_raw = change_speed(audio_np_raw, speed, sr_raw)
                buf = io.BytesIO()
                sf_temp.write(buf, audio_np_raw, sr_raw, format='WAV')
                audio_data = buf.getvalue()
                sample_rate = sr_raw
            
            # Save to temp file
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(OUTPUT_DIR))
            tmp.write(audio_data)
            tmp.close()
            
            # Also save to outputPath if specified
            if output_path and output_path.strip():
                try:
                    os.makedirs(output_path.strip(), exist_ok=True)
                    out_name = f"voice_{voice_id}_{int(time.time())}.wav"
                    out_full = os.path.join(output_path.strip(), out_name)
                    with open(out_full, 'wb') as f:
                        f.write(audio_data)
                    print(f"[Voice AI] Also saved to: {out_full}")
                except Exception as e:
                    print(f"[Voice AI] Warning: Could not save to outputPath: {e}")
            
            print(f"[Voice AI] Fish Speech 1.5 done, saved to {tmp.name}")
            
            return send_file(
                tmp.name,
                mimetype="audio/wav",
                as_attachment=True,
                download_name="output.wav"
            )
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"success": False, "message": f"Lỗi Fish Speech: {str(e)}"}), 500

    # ────────────────────────────────────────────────────────
    # Viterbox Synthesis Path (Chatterbox-based, emotion control)
    # ────────────────────────────────────────────────────────
    if requested_model == "dolly-vn/viterbox":
        try:
            from audio_enhance import enhance_audio
            
            # Viterbox advanced params (all tuneable from frontend)
            exaggeration = float(data.get("exaggeration", 0.5))     # 0.0-2.0: emotion intensity
            cfg_weight = float(data.get("cfg_weight", 2.5))         # 0.0-5.0: voice similarity
            vb_temperature = float(data.get("vb_temperature", 0.5)) # 0.1-1.0: randomness
            top_p = float(data.get("top_p", 0.95))                  # nucleus sampling
            rep_penalty = float(data.get("repetition_penalty", 1.2)) # repetition penalty
            pause_ms = int(data.get("sentence_pause_ms", 400))      # pause between sentences
            
            # ── Auto Language Detection ──
            lang = detect_language(text)
            print(f"[Voice AI] Auto-detected language: {lang}")
            
            # ── Try to load pre-computed embeddings (FAST path) ──
            cached_embeddings_path = None
            ref_audio_path = None
            use_cached = False
            
            if mode == "fast":
                # Fast clone: always compute from uploaded audio
                ref_audio_path = data.get("ref_audio_path", "")
                if not ref_audio_path or not os.path.exists(ref_audio_path) or os.path.isdir(ref_audio_path):
                    ref_audio_path = None
            else:
                # Preset mode: check for cached embeddings first
                cache_path = ROOT / "voices" / voice_id / "embeddings.pt"
                if cache_path.exists():
                    cached_embeddings_path = str(cache_path)
                    use_cached = True
                else:
                    # Fallback: compute from ref.wav
                    ref_audio_rel = voice_cfg.get("ref_audio", "")
                    if ref_audio_rel:
                        ref_path_candidate = str(ROOT / ref_audio_rel)
                        if os.path.exists(ref_path_candidate):
                            ref_audio_path = ref_path_candidate
            
            if use_cached:
                # ── FAST PATH: Load pre-computed embeddings ──
                import torch
                from viterbox.tts import TTSConds
                from viterbox.models.t3.modules.cond_enc import T3Cond
                
                conds = TTSConds.load(cached_embeddings_path, engine.device)
                
                # Override emotion_adv with current exaggeration
                if hasattr(conds.t3, 'emotion_adv'):
                    conds.t3.emotion_adv = (exaggeration * torch.ones(1, 1, 1)).to(engine.device)
                
                engine.conds = conds
                
                print(f"[Voice AI] Viterbox: CACHED embeddings for '{voice_id}', exag={exaggeration}, cfg={cfg_weight}, temp={vb_temperature}, lang={lang}")
                
                # Generate without audio_prompt (uses cached conds)
                audio_tensor = engine.generate(
                    text=text,
                    language=lang,
                    audio_prompt=None,  # Use cached conds
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=vb_temperature,
                    top_p=top_p,
                    repetition_penalty=rep_penalty,
                    split_sentences=True,
                    crossfade_ms=50,
                    sentence_pause_ms=pause_ms,
                )
            else:
                # ── SLOW PATH: Compute from ref.wav ──
                print(f"[Voice AI] Viterbox: computing from ref audio, exag={exaggeration}, cfg={cfg_weight}, temp={vb_temperature}, lang={lang}, ref={'YES' if ref_audio_path else 'random'}")
                
                audio_tensor = engine.generate(
                    text=text,
                    language=lang,
                    audio_prompt=ref_audio_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=vb_temperature,
                    top_p=top_p,
                    repetition_penalty=rep_penalty,
                    split_sentences=True,
                    crossfade_ms=50,
                    sentence_pause_ms=pause_ms,
                )
            
            # Convert to numpy (output is [1, samples] at 24kHz)
            audio_np = audio_tensor[0].cpu().numpy()
            sample_rate = 24000  # Viterbox native SR
            
            # ── Post-processing: aggressive anti-click + quality enhancement ──
            import numpy as np
            from scipy import signal as scipy_signal
            
            # 1. High-pass filter at 60Hz to remove low-freq pops/clicks
            try:
                sos = scipy_signal.butter(4, 60, btype='highpass', fs=sample_rate, output='sos')
                audio_np = scipy_signal.sosfilt(sos, audio_np).astype(np.float32)
            except Exception:
                pass
            
            # 2. De-click: find and smooth isolated amplitude spikes
            # A "click" is a sample that's >3x the local RMS
            window = 512
            for i in range(0, len(audio_np) - window, window):
                chunk = audio_np[i:i+window]
                local_rms = np.sqrt(np.mean(chunk**2))
                if local_rms > 0:
                    spike_mask = np.abs(chunk) > 4.0 * local_rms
                    if np.any(spike_mask):
                        # Smooth spikes via linear interpolation
                        for j in np.where(spike_mask)[0]:
                            idx = i + j
                            if 1 < idx < len(audio_np) - 1:
                                audio_np[idx] = (audio_np[idx-1] + audio_np[idx+1]) / 2
            
            # 3. Trim trailing silence (below -40dB)
            threshold = 0.008
            abs_audio = np.abs(audio_np)
            above_thresh = np.where(abs_audio > threshold)[0]
            if len(above_thresh) > 0:
                last_voice = above_thresh[-1]
                pad_samples = int(0.03 * sample_rate)  # 30ms padding
                trim_end = min(last_voice + pad_samples, len(audio_np))
                audio_np = audio_np[:trim_end]
            
            # 4. Aggressive fade-out (100ms) — eliminates any end artifact
            fade_out_ms = 100
            fade_out_samples = int(fade_out_ms / 1000 * sample_rate)
            if len(audio_np) > fade_out_samples:
                fade_out = np.linspace(1.0, 0.0, fade_out_samples) ** 2  # quadratic = smoother
                audio_np[-fade_out_samples:] *= fade_out
            
            # 5. Smooth fade-in (20ms)
            fade_in_samples = int(0.02 * sample_rate)
            if len(audio_np) > fade_in_samples:
                fade_in = np.linspace(0.0, 1.0, fade_in_samples) ** 2
                audio_np[:fade_in_samples] *= fade_in
            
            # Apply speed change if requested
            if abs(speed - 1.0) >= 0.05:
                audio_np = change_speed(audio_np, speed, sample_rate)
            
            # Post-processing: upscale 24kHz → 48kHz + enhance
            try:
                audio_np = enhance_audio(audio_np, sample_rate, target_sr=48000)
                sample_rate = 48000
                print(f"[Voice AI] Viterbox audio enhanced + upsampled to 48kHz")
            except Exception as e:
                print(f"[Voice AI] Enhancement skipped: {e}")
            
            # Save to temp file
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=str(OUTPUT_DIR))
            sf.write(tmp.name, audio_np, sample_rate)
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
            
            print(f"[Voice AI] Viterbox TTS done, {len(audio_np)} samples at {sample_rate}Hz")
            
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
            return jsonify({"success": False, "message": f"Lỗi tổng hợp Viterbox: {str(e)}"}), 500

    # ────────────────────────────────────────────────────────
    # Vira-TTS Synthesis Path
    # ────────────────────────────────────────────────────────
    else:
        # Determine reference audio
        if mode == "fast":
            ref_audio_path = data.get("ref_audio_path", "")
            if not ref_audio_path or not os.path.exists(ref_audio_path) or os.path.isdir(ref_audio_path):
                return jsonify({"success": False, "message": "Không tìm thấy file âm thanh mẫu hoặc đường dẫn không hợp lệ"}), 400
        else:
            if voice_id not in voices:
                return jsonify({"success": False, "message": f"Voice '{voice_id}' không tồn tại"}), 404

            ref_audio_rel = voice_cfg.get("ref_audio", "")
            ref_audio_path = str(ROOT / ref_audio_rel)

            if not os.path.exists(ref_audio_path) or os.path.isdir(ref_audio_path):
                return jsonify({"success": False, "message": f"File audio mẫu không tồn tại hoặc không hợp lệ: {ref_audio_path}"}), 404

        try:
            # 1. Encode reference audio
            context_tokens = engine.encode_audio(ref_audio_path)

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

            # 4. Synthesize each segment and insert natural pauses
            from audio_enhance import crossfade_segments, add_natural_pause, enhance_audio
            
            audio_parts = []  # list of numpy arrays for crossfade blending
            sample_rate = 48000

            for seg_text, punc in segments:
                seg_text = seg_text.strip()
                if not seg_text:
                    continue
                    
                audio_seg = engine.generate(seg_text, context_tokens)
                audio_np_seg = audio_seg.float().cpu().numpy()
                
                # Trim leading/trailing silence from raw segment (keep a tiny 40ms margin)
                try:
                    import librosa
                    audio_np_seg, _ = librosa.effects.trim(audio_np_seg, top_db=30)
                except Exception as e:
                    print(f"[Voice AI] Segment trim warning: {e}")
                
                # Apply smooth fade-in and fade-out to prevent click artifacts
                seg_len = len(audio_np_seg)
                fade_in_len = min(int(sample_rate * 0.005), seg_len // 4)   # 5ms
                fade_out_len = min(int(sample_rate * 0.01), seg_len // 4)    # 10ms
                if fade_in_len > 1:
                    audio_np_seg[:fade_in_len] *= np.linspace(0.0, 1.0, fade_in_len, dtype=np.float32)
                if fade_out_len > 1:
                    audio_np_seg[-fade_out_len:] *= np.linspace(1.0, 0.0, fade_out_len, dtype=np.float32)
                    
                audio_parts.append(audio_np_seg)
                
                # Add natural pause with noise floor (not dead silence)
                pause = add_natural_pause(sample_rate, punc)
                audio_parts.append(pause)

            if not audio_parts:
                return jsonify({"success": False, "message": "Không thể tạo giọng nói từ văn bản này"}), 400

            # 5. Join segments with smooth crossfade blending
            wav_data = crossfade_segments(audio_parts, sample_rate, crossfade_ms=40)

            # 6. Apply speed change if requested
            if abs(speed - 1.0) >= 0.05:
                wav_data = change_speed(wav_data, speed, sample_rate)
            
            # 7. Post-processing enhancement pipeline
            try:
                wav_data = enhance_audio(wav_data, sample_rate, target_sr=48000)
                print(f"[Voice AI] Vira-TTS audio enhanced successfully")
            except Exception as e:
                print(f"[Voice AI] Enhancement skipped: {e}")

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
    print(f"  Voice AI - Local TTS Server v{VOICE_AI_VERSION}")
    print("  Port: 9880 | Engine: Gwen-TTS (Qwen3-TTS 0.6B)")
    print("  GPU: RTX 4060 | Audio Enhancement: ON")
    print("=" * 60)

    app.run(host="127.0.0.1", port=9880, threaded=True, debug=False)
