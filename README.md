# 🎙️ Voice AI

**Local Vietnamese Voice Cloning** powered by [F5-TTS](https://github.com/SWivid/F5-TTS) with 1,000-hour pre-trained Vietnamese model.

> Zero-shot voice cloning quality on par with ElevenLabs & Minimax — running 100% locally on your GPU.

## Features

- 🚀 **Clone Nhanh (Zero-Shot)**: Upload 3-10s audio → instant voice clone
- 🎯 **Clone Chi Tiết (Fine-tune)**: Train on longer recordings (1-10 min) for perfect accuracy
- 💾 **Persistent Storage**: Voices saved to both local disk and GitHub
- 🌐 **Web UI**: Integrated into OpenClaw tab on web-mcbooks-export
- ⚡ **RTX 4060 Optimized**: Fast inference on consumer GPU (~4GB VRAM)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download model weights (~1.2GB)
python setup.py

# 3. Start the API server
python f5_api.py
# Server runs on http://127.0.0.1:9880
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server status |
| `/voices` | GET | List all saved voices |
| `/voices` | POST | Save a new voice (multipart form) |
| `/voices/<id>` | DELETE | Delete a voice |
| `/tts` | POST | Generate speech (JSON body) |
| `/upload` | POST | Upload reference audio |
| `/train` | POST | Start fine-tune training |
| `/train/status` | GET | Check training progress |
| `/sync` | POST | Push voices to GitHub |

## Architecture

```
voice-ai/
├── f5_api.py          # Flask API server (port 9880)
├── setup.py           # Download model weights
├── voices.json        # Voice database (synced to GitHub)
├── voices/            # Reference audio per voice
├── weights/           # F5-TTS model (gitignored, ~1.2GB)
├── output/            # Generated audio (gitignored)
├── requirements.txt
├── start.bat          # One-click launcher
└── README.md
```

## Tech Stack

- **F5-TTS**: Flow Matching DiT architecture for voice synthesis
- **Model**: `hynt/F5-TTS-Vietnamese-ViVoice` (1,000h Vietnamese audio)
- **Backend**: Flask + PyTorch + CUDA
- **Frontend**: React (OpenClaw tab integration)
- **Storage**: Local + GitHub sync

## License

MIT
