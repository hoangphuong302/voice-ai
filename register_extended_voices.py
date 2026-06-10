import os
import sys
import json
import shutil
import urllib.request
import subprocess
from pathlib import Path

# Force UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8")

E_DRIVE = "E:\\"
ROOT = Path("D:\\voice-ai")
VOICES_DIR = ROOT / "voices"
VOICES_JSON_PATH = ROOT / "voices.json"

for d in [VOICES_DIR]:
    d.mkdir(exist_ok=True, parents=True)

# Vietnamese mapping from E:\
VI_MAPPING = {
    "vi_vo": {"filename": "giọng con vợ.mp3", "name": "Vợ Sam", "desc": "Giọng nữ gia đình trẻ trung, tự nhiên."},
    "vi_duong2": {"filename": "giọng dương 2.MP3", "name": "Nam Dương 2", "desc": "Giọng nam trầm ấm, rõ ràng, tốc độ vừa phải."},
    "vi_grifin": {"filename": "giọng grifin.MP3", "name": "Nam Grifin", "desc": "Giọng nam năng động, thích hợp làm video review."},
    "vi_nu2_dieu": {"filename": "giọng nữ  2 (giọng cao quảng cáo hơi điệu).MP3", "name": "Nữ Điệu (Quảng cáo)", "desc": "Giọng nữ cao, điệu đà, thích hợp đọc quảng cáo."},
    "vi_nu11": {"filename": "giọng nữ 11.MP3", "name": "Nữ 11", "desc": "Giọng nữ miền Bắc ấm áp, truyền cảm."},
    "vi_nu13": {"filename": "giọng nữ 13.MP3", "name": "Nữ 13", "desc": "Giọng nữ miền Bắc dịu dàng, nhẹ nhàng."},
    "vi_nu14": {"filename": "giong nữ 14.MP3", "name": "Nữ 14", "desc": "Giọng nữ miền Bắc trong sáng, hoạt ngôn."},
    "vi_nu15": {"filename": "giọng nữ 15.MP3", "name": "Nữ 15", "desc": "Giọng nữ miền Bắc thanh lịch, đĩnh đạc."},
    "vi_nu4": {"filename": "giọng nữ 4.MP3", "name": "Nữ 4", "desc": "Giọng nữ miền Bắc truyền cảm, kể chuyện hay."},
    "vi_nu5": {"filename": "giọng nữ 5.MP3", "name": "Nữ 5", "desc": "Giọng nữ miền Bắc ngọt ngào, ấm áp."},
    "vi_nu7": {"filename": "giọng nữ 7.MP3", "name": "Nữ 7", "desc": "Giọng nữ miền Bắc trẻ trung, tự nhiên."},
    "vi_nu9_thoisu": {"filename": "giọng nữ 9 (thời sự bản tin).MP3", "name": "Nữ Thời Sự (Bản tin)", "desc": "Giọng nữ đọc bản tin, thời sự trang trọng, chuyên nghiệp."},
    "vi_gai1": {"filename": "gái 1.MP3", "name": "Nữ Trẻ 1", "desc": "Giọng nữ trẻ năng động."},
    "vi_gai2": {"filename": "gái 2.MP3", "name": "Nữ Trẻ 2", "desc": "Giọng nữ trẻ dễ thương."},
    "vi_gai3": {"filename": "gái 3 tây ban nha.MP3", "name": "Nữ Trẻ 3", "desc": "Giọng nữ trẻ trong trẻo."},
    "vi_gai4": {"filename": "gái 4.MP3", "name": "Nữ Trẻ 4", "desc": "Giọng nữ trẻ ngọt ngào."},
    "vi_tre_con2": {"filename": "mẫu giọng trẻ con.MP3", "name": "Bé gái mẫu giọng", "desc": "Giọng bé gái dễ thương, ngây thơ."},
    "vi_nu16": {"filename": "nữ 16.MP3", "name": "Nữ 16", "desc": "Giọng nữ miền Bắc hoạt bát."},
    "vi_nu17": {"filename": "nữ 17.MP3", "name": "Nữ 17", "desc": "Giọng nữ miền Bắc ấm áp."},
    "vi_nu19": {"filename": "nữ 19.MP3", "name": "Nữ 19", "desc": "Giọng nữ miền Bắc dịu dàng."},
    "vi_nu20": {"filename": "nữ 20.MP3", "name": "Nữ 20", "desc": "Giọng nữ miền Bắc trong trẻo."}
}

# English URLs
EN_MAPPING = {
    "en_female_1": {
        "url": "https://raw.githubusercontent.com/coqui-ai/TTS/main/tests/data/ljspeech/wavs/LJ001-0002.wav",
        "name": "[English] Nữ 1 (Mạnh mẽ)",
        "desc": "Giọng nữ tiếng Anh chuẩn, rõ ràng, tốc độ vừa phải."
    },
    "en_female_2": {
        "url": "https://raw.githubusercontent.com/coqui-ai/TTS/main/tests/data/ljspeech/wavs/LJ001-0003.wav",
        "name": "[English] Nữ 2 (Tự nhiên)",
        "desc": "Giọng nữ tiếng Anh chuẩn, tự nhiên."
    },
    "en_female_3": {
        "url": "https://raw.githubusercontent.com/coqui-ai/TTS/main/tests/data/ljspeech/wavs/LJ001-0004.wav",
        "name": "[English] Nữ 3 (Trang trọng)",
        "desc": "Giọng nữ tiếng Anh chuyên nghiệp, đĩnh đạc."
    },
    "en_female_4": {
        "url": "https://raw.githubusercontent.com/coqui-ai/TTS/main/tests/data/ljspeech/wavs/LJ001-0005.wav",
        "name": "[English] Nữ 4 (Kể chuyện)",
        "desc": "Giọng nữ tiếng Anh ấm áp, truyền cảm."
    }
}

# Chinese URLs (from AISHELL-3 test files)
ZH_MAPPING = {
    "zh_male_1": {
        "url": "https://huggingface.co/datasets/shenberg1/aishell3/resolve/main/test/wav/SSB0011/SSB00110049.wav",
        "name": "[Chinese] Nam 1",
        "desc": "Giọng nam tiếng Trung Quốc phổ thông chuẩn, đĩnh đạc."
    },
    "zh_female_2": {
        "url": "https://huggingface.co/datasets/shenberg1/aishell3/resolve/main/test/wav/SSB0005/SSB00050353.wav",
        "name": "[Chinese] Nữ 1",
        "desc": "Giọng nữ tiếng Trung Quốc phổ thông trong sáng, tự nhiên."
    },
    "zh_female_3": {
        "url": "https://huggingface.co/datasets/shenberg1/aishell3/resolve/main/test/wav/SSB0016/SSB00160056.wav",
        "name": "[Chinese] Nữ 2",
        "desc": "Giọng nữ tiếng Trung Quốc ngọt ngào, truyền cảm."
    }
}

def convert_audio(src, dst):
    """Convert audio to 24kHz Mono WAV using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-ar", "24000", "-ac", "1", str(dst)
    ]
    res = subprocess.run(cmd, capture_output=True)
    return res.returncode == 0

def slice_audio(path, limit=8.0):
    """Slice audio to first N seconds."""
    tmp = path.with_name(path.stem + "_tmp.wav")
    cmd = [
        "ffmpeg", "-y", "-i", str(path),
        "-ss", "0", "-t", str(limit),
        "-c", "copy", str(tmp)
    ]
    subprocess.run(cmd, capture_output=True)
    if tmp.exists():
        path.unlink()
        tmp.rename(path)

def download_file(url, dst):
    """Download a file with User-Agent header to avoid HTTP blocks."""
    print(f"Downloading: {url} -> {dst}")
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    with urllib.request.urlopen(req) as response, open(dst, 'wb') as out_file:
        out_file.write(response.read())

def main():
    if VOICES_JSON_PATH.exists():
        with open(VOICES_JSON_PATH, "r", encoding="utf-8") as f:
            voices_db = json.load(f)
    else:
        voices_db = {}

    # Initialize Whisper Model for automatic transcription
    print("Loading Whisper Model...")
    from faster_whisper import WhisperModel
    whisper = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

    # 1. Process Vietnamese voices from E:\
    for vid, info in VI_MAPPING.items():
        src_path = Path(E_DRIVE) / info["filename"]
        if not src_path.exists():
            print(f"Source file {src_path} does not exist, skipping.")
            continue

        target_dir = VOICES_DIR / vid
        target_dir.mkdir(exist_ok=True, parents=True)
        target_wav = target_dir / "ref.wav"

        print(f"\nProcessing VI voice: {info['name']} ({vid})...")
        if convert_audio(src_path, target_wav):
            slice_audio(target_wav, 8.0) # Slice to 8 seconds
            
            # Transcribe
            print(f"Transcribing {vid}...")
            segments, _ = whisper.transcribe(str(target_wav), language="vi")
            transcript = " ".join([seg.text for seg in segments]).strip()
            print("Transcript:", transcript)

            # Register
            voices_db[vid] = {
                "name": info["name"],
                "desc": info["desc"],
                "ref_audio": f"voices/{vid}/ref.wav",
                "ref_text": transcript,
                "type": "preset",
                "created_at": "2026-06-10T00:00:00Z"
            }
        else:
            print(f"Failed to convert {src_path}")

    # 2. Process English voices
    for vid, info in EN_MAPPING.items():
        target_dir = VOICES_DIR / vid
        target_dir.mkdir(exist_ok=True, parents=True)
        target_wav = target_dir / "ref.wav"

        print(f"\nProcessing EN voice: {info['name']} ({vid})...")
        try:
            download_file(info["url"], target_wav)
            slice_audio(target_wav, 8.0)

            # Transcribe
            print(f"Transcribing {vid}...")
            segments, _ = whisper.transcribe(str(target_wav), language="en")
            transcript = " ".join([seg.text for seg in segments]).strip()
            print("Transcript:", transcript)

            voices_db[vid] = {
                "name": info["name"],
                "desc": info["desc"],
                "ref_audio": f"voices/{vid}/ref.wav",
                "ref_text": transcript,
                "type": "preset",
                "created_at": "2026-06-10T00:00:00Z"
            }
        except Exception as e:
            print(f"Failed to process EN voice {vid}: {e}")

    # 3. Process Chinese voices
    for vid, info in ZH_MAPPING.items():
        target_dir = VOICES_DIR / vid
        target_dir.mkdir(exist_ok=True, parents=True)
        target_wav = target_dir / "ref.wav"

        print(f"\nProcessing ZH voice: {info['name']} ({vid})...")
        try:
            download_file(info["url"], target_wav)
            slice_audio(target_wav, 8.0)

            # Transcribe
            print(f"Transcribing {vid}...")
            segments, _ = whisper.transcribe(str(target_wav), language="zh")
            transcript = " ".join([seg.text for seg in segments]).strip()
            print("Transcript:", transcript)

            voices_db[vid] = {
                "name": info["name"],
                "desc": info["desc"],
                "ref_audio": f"voices/{vid}/ref.wav",
                "ref_text": transcript,
                "type": "preset",
                "created_at": "2026-06-10T00:00:00Z"
            }
        except Exception as e:
            print(f"Failed to process ZH voice {vid}: {e}")

    # Write back to voices.json
    with open(VOICES_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(voices_db, f, ensure_ascii=False, indent=2)
    print("\nPreset voices registered successfully in voices.json!")

if __name__ == "__main__":
    main()
