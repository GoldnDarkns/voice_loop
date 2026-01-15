# 🎙️ Voice Loop

A local Speech AI Assistant that creates a complete voice interaction loop:
**Speech → STT → LLM → TTS → Speech**

All processing happens locally on your machine - no cloud APIs required!

## Features

- 🎤 **Speech-to-Text (STT)**: Uses [faster-whisper](https://github.com/guillaumekln/faster-whisper) for local transcription
- 🧠 **LLM**: Uses [Ollama](https://ollama.ai/) for local AI responses
- 🔊 **Text-to-Speech (TTS)**: Uses pyttsx3 (Windows SAPI) for speech synthesis
- 🌐 **Beautiful Web UI**: Interactive interface with real-time workflow visualization
- 📊 **Performance Metrics**: Track recording, transcription, LLM, and TTS timings

## Prerequisites

1. **Python 3.10+**
2. **Ollama** - Install from [ollama.ai](https://ollama.ai/)
3. **A microphone**

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/voice-loop.git
cd voice-loop

# Install dependencies
pip install -r requirements.txt

# Pull a small LLM model
ollama pull phi
```

## Usage

### 1. Start Ollama (in one terminal)
```bash
ollama run phi
```

### 2. Start the server (in another terminal)
```bash
python server.py --port 7860 --stt-model small --llm-model phi
```

### 3. Open the UI
Open `ui/index.html` in your browser

### 4. Talk to the AI!
- Click "Start Recording"
- Speak your question (recording starts immediately!)
- Wait for transcription
- Review/edit the transcript if needed
- Click "Send to AI"
- Listen to the response!

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `--port` | 7860 | Server port |
| `--stt-model` | small | Whisper model (tiny/base/small/medium) |
| `--llm-model` | phi | Ollama model name |
| `--seconds` | 5 | Default recording duration |
| `--rate` | 1.0 | TTS speech rate |

## STT Models

| Model | Speed | Accuracy | RAM |
|-------|-------|----------|-----|
| tiny | Fastest | Lower | ~1GB |
| base | Fast | Good | ~1GB |
| small | Balanced | Better | ~2GB |
| medium | Slow | Best | ~5GB |

## Project Structure

```
voice_loop/
├── server.py          # Main Flask server
├── app.py             # CLI version (alternative)
├── ui/
│   └── index.html     # Web interface
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Troubleshooting

### "No speech detected"
- Speak louder and closer to the microphone
- Reduce recording duration to 2-3 seconds
- Use the "small" STT model instead of "medium"

### Transcription is slow
- Use "small" or "base" model instead of "medium"
- The first transcription is slower (model loading)

### Hallucinations (wrong transcription)
- Reduce recording duration to avoid silence
- Speak immediately when recording starts
- Use VAD (Voice Activity Detection) - enabled by default

## License

MIT License
