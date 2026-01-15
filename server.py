import os
import time
import tempfile
import argparse
import json
import subprocess
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
import pyttsx3
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from faster_whisper import WhisperModel

# Try importing TTS engines
PIPER_AVAILABLE = False
COQUI_AVAILABLE = False

try:
    import piper
    PIPER_AVAILABLE = True
except ImportError:
    pass

try:
    from TTS.api import TTS as CoquiTTS
    COQUI_AVAILABLE = True
except ImportError:
    pass

# ---------- Config ----------
DEFAULT_SECONDS = int(os.environ.get("VOICE_SECONDS", "5"))
DEFAULT_STT_MODEL = os.environ.get("VOICE_STT_MODEL", "small")
DEFAULT_LLM_MODEL = os.environ.get("VOICE_LLM_MODEL", "phi")
DEFAULT_TTS_ENGINE = os.environ.get("VOICE_TTS_ENGINE", "pyttsx3")
DEFAULT_RATE = float(os.environ.get("VOICE_TTS_RATE", "1.0"))
DEFAULT_PROMPT = os.environ.get("VOICE_SYSTEM_PROMPT", "Answer briefly.")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

app = Flask(__name__)
CORS(app)

stt_model: Optional[WhisperModel] = None
coqui_tts = None  # Type: Optional[TTS.api.TTS]
recording_active = False
tts_active = False
tts_engine_instance = None
current_recording_duration = 5
last_audio_path = None


def load_stt_model(name: str):
    global stt_model
    if stt_model is None:
        print(f"[init] Loading faster-whisper model '{name}' ...")
        stt_model = WhisperModel(name, device="cpu", compute_type="int8")
    return stt_model


def simple_record_audio(duration_seconds: int) -> str:
    """Simple blocking audio recording."""
    print(f"[record] Recording {duration_seconds}s with blocking method...")
    print(f"[record] Available devices: {sd.query_devices()}")
    samplerate = 16000
    
    try:
        # Record audio (blocking)
        print("[record] Starting sd.rec()...")
        audio = sd.rec(int(duration_seconds * samplerate), 
                       samplerate=samplerate, 
                       channels=1, 
                       dtype='float32')
        print(f"[record] Waiting for recording to complete...")
        sd.wait()  # Wait until recording is finished
        print(f"[record] Recording complete! Got {len(audio)} samples")
        
        # Save to file
        path = os.path.join(tempfile.gettempdir(), f"record_{int(time.time())}.wav")
        print(f"[record] Saving to {path}...")
        sf.write(path, audio, samplerate)
        file_size = os.path.getsize(path)
        print(f"[record] Saved {len(audio)} samples to {path} ({file_size} bytes)")
        return path
    except Exception as e:
        print(f"[record] ERROR in simple_record_audio: {e}")
        import traceback
        traceback.print_exc()
        raise


def transcribe(path: str, model: WhisperModel) -> str:
    """Transcribe audio with faster-whisper."""
    print("[stt] Transcribing locally with faster-whisper...")
    try:
        # Force English language for better accuracy
        segments, info = model.transcribe(
            path, 
            beam_size=1,
            language="en",  # Force English
            vad_filter=True,  # Voice Activity Detection - filters out silence
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        text = " ".join([s.text.strip() for s in segments]).strip()
        # Safe print for Windows consoles
        safe_text = text.encode("utf-8", "replace").decode("utf-8", "replace")
        print(f"[stt] Transcript: {safe_text}")
        print(f"[stt] Language detected: {info.language}, probability: {info.language_probability:.2f}")
        return text
    except Exception as e:
        print(f"[stt] Transcription error: {e}")
        return ""


def ask_ollama(prompt: str, model_name: str, system_prompt: str) -> str:
    """Send the prompt to local Ollama chat API and return response text."""
    url = f"{OLLAMA_HOST.rstrip('/')}/api/chat"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    print(f"[llm] Querying {model_name} at {url} ...")
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    content = r.json()["message"]["content"]
    print(f"[llm] Response: {content}")
    return content


def speak_pyttsx3(text: str, rate_factor: float = 1.0) -> str:
    """Speak text via pyttsx3 (Windows SAPI)."""
    global tts_active, tts_engine_instance
    print("[tts] Speaking with pyttsx3 (Windows SAPI)...")
    tts_active = True
    engine = pyttsx3.init()
    tts_engine_instance = engine
    rate = engine.getProperty("rate")
    engine.setProperty("rate", int(rate * rate_factor))
    engine.say(text)
    
    # Check tts_active periodically during playback
    engine.startLoop(False)
    while tts_active and engine.isBusy():
        engine.iterate()
        time.sleep(0.1)
    engine.endLoop()
    
    tts_active = False
    tts_engine_instance = None
    return "pyttsx3 (Windows SAPI)"


def speak_piper(text: str, rate_factor: float = 1.0) -> str:
    """Speak text via Piper TTS (neural)."""
    if not PIPER_AVAILABLE:
        print("[tts] Piper not installed, falling back to pyttsx3...")
        return speak_pyttsx3(text, rate_factor)
    
    print("[tts] Speaking with Piper (neural TTS)...")
    # Use piper CLI if available
    try:
        output_path = os.path.join(tempfile.gettempdir(), f"piper_{int(time.time())}.wav")
        # Try to find piper executable or use piper-tts python package
        # For simplicity, fall back to pyttsx3 if not configured
        print("[tts] Piper requires manual setup, falling back to pyttsx3...")
        return speak_pyttsx3(text, rate_factor)
    except Exception as e:
        print(f"[tts] Piper error: {e}, falling back to pyttsx3...")
        return speak_pyttsx3(text, rate_factor)


def speak_coqui(text: str, rate_factor: float = 1.0) -> str:
    """Speak text via Coqui TTS (neural)."""
    global coqui_tts
    if not COQUI_AVAILABLE:
        print("[tts] Coqui TTS not installed, falling back to pyttsx3...")
        return speak_pyttsx3(text, rate_factor)
    
    print("[tts] Speaking with Coqui TTS (neural)...")
    try:
        if coqui_tts is None:
            # Use a fast, small model
            coqui_tts = CoquiTTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
        
        output_path = os.path.join(tempfile.gettempdir(), f"coqui_{int(time.time())}.wav")
        coqui_tts.tts_to_file(text=text, file_path=output_path)
        
        # Play the audio
        data, samplerate = sf.read(output_path)
        sd.play(data, samplerate)
        sd.wait()
        
        return "Coqui TTS (Tacotron2-DDC)"
    except Exception as e:
        print(f"[tts] Coqui error: {e}, falling back to pyttsx3...")
        return speak_pyttsx3(text, rate_factor)


def speak(text: str, engine: str = "pyttsx3", rate_factor: float = 1.0) -> str:
    """Speak text using the specified TTS engine."""
    if engine == "piper":
        return speak_piper(text, rate_factor)
    elif engine == "coqui":
        return speak_coqui(text, rate_factor)
    else:
        return speak_pyttsx3(text, rate_factor)


@app.route("/health", methods=["GET"])
def health():
    return {"status": "ok"}, 200


@app.route("/engines", methods=["GET"])
def list_engines():
    """List available TTS engines."""
    engines = [
        {"id": "pyttsx3", "name": "Windows SAPI (pyttsx3)", "available": True, "type": "OS Voice"},
        {"id": "piper", "name": "Piper TTS", "available": PIPER_AVAILABLE, "type": "Neural TTS"},
        {"id": "coqui", "name": "Coqui TTS", "available": COQUI_AVAILABLE, "type": "Neural TTS"},
    ]
    return jsonify({"engines": engines})


@app.route("/record/start", methods=["POST"])
def start_recording():
    """Start recording immediately and save to file."""
    global recording_active, current_recording_duration, last_audio_path
    data = request.get_json(force=True, silent=True) or {}
    seconds = int(data.get("seconds", DEFAULT_SECONDS))
    
    print(f"[record] Starting IMMEDIATE recording for {seconds}s...")
    
    recording_active = True
    current_recording_duration = seconds
    
    # Record immediately (blocking in a way that captures audio NOW)
    try:
        samplerate = 16000
        print(f"[record] Recording {seconds}s NOW...")
        audio = sd.rec(int(seconds * samplerate), 
                       samplerate=samplerate, 
                       channels=1, 
                       dtype='float32')
        sd.wait()  # Wait for recording to complete
        
        # Save to file
        path = os.path.join(tempfile.gettempdir(), f"record_{int(time.time())}.wav")
        sf.write(path, audio, samplerate)
        file_size = os.path.getsize(path)
        print(f"[record] Saved {len(audio)} samples to {path} ({file_size} bytes)")
        
        last_audio_path = path
        recording_active = False
        
        response_data = {"status": "recorded", "max_seconds": seconds, "audio_path": path}
        print(f"[record] Returning response: {response_data}")
        return jsonify(response_data)
    except Exception as e:
        print(f"[record] Recording error: {e}")
        import traceback
        traceback.print_exc()
        recording_active = False
        return jsonify({"error": str(e)}), 500


@app.route("/record/stop", methods=["POST"])
def stop_recording_endpoint():
    """Return the audio path from the recording that already happened."""
    global recording_active, last_audio_path
    recording_active = False
    
    print(f"[record] Stop called, returning audio path: {last_audio_path}")
    
    if not last_audio_path or not os.path.exists(last_audio_path):
        return jsonify({"error": "no audio recorded"}), 400
    
    return jsonify({"status": "stopped", "audio_path": last_audio_path})


@app.route("/transcribe", methods=["POST"])
def transcribe_endpoint():
    """Transcribe audio file."""
    data = request.get_json(force=True, silent=True) or {}
    audio_path = data.get("audio_path", "")
    stt_name = data.get("stt_model", DEFAULT_STT_MODEL)
    
    if not audio_path or not os.path.exists(audio_path):
        return jsonify({"error": "audio file not found"}), 400
    
    # Check file size - if too small, likely no audio
    file_size = os.path.getsize(audio_path)
    print(f"[transcribe] Audio file size: {file_size} bytes")
    
    if file_size < 1000:
        return jsonify({"error": "audio file too small - no audio captured"}), 400
    
    # Load STT model
    model = load_stt_model(stt_name)
    
    t0 = time.time()
    user_text = transcribe(audio_path, model)
    duration_ms = int((time.time() - t0) * 1000)
    
    # If no speech detected, return a placeholder instead of error
    if not user_text or user_text.strip() == "":
        print("[transcribe] No speech detected - returning placeholder")
        user_text = "(no speech detected - please speak louder)"
    
    return jsonify({
        "transcript": user_text,
        "stt_model": f"faster-whisper ({stt_name})",
        "duration_ms": duration_ms
    })


@app.route("/llm/ask", methods=["POST"])
def ask_llm_endpoint():
    """Send text to LLM and get response."""
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    llm_name = data.get("model", DEFAULT_LLM_MODEL)
    system_prompt = data.get("prompt", DEFAULT_PROMPT)
    
    if not text:
        return jsonify({"error": "no text provided"}), 400
    
    t0 = time.time()
    llm_reply = ask_ollama(text, llm_name, system_prompt)
    duration_ms = int((time.time() - t0) * 1000)
    
    return jsonify({
        "reply": llm_reply,
        "llm_model": llm_name,
        "ollama_host": OLLAMA_HOST,
        "duration_ms": duration_ms
    })


@app.route("/tts/speak", methods=["POST"])
def speak_endpoint():
    """Speak text using TTS."""
    data = request.get_json(force=True, silent=True) or {}
    text = data.get("text", "")
    tts_engine = data.get("tts_engine", DEFAULT_TTS_ENGINE)
    rate = float(data.get("rate", DEFAULT_RATE))
    
    if not text:
        return jsonify({"error": "no text provided"}), 400
    
    t0 = time.time()
    actual_engine = speak(text, engine=tts_engine, rate_factor=rate)
    duration_ms = int((time.time() - t0) * 1000)
    
    return jsonify({
        "tts_engine": actual_engine,
        "duration_ms": duration_ms
    })


@app.route("/tts/stop", methods=["POST"])
def stop_tts_endpoint():
    """Stop active TTS playback."""
    global tts_active, tts_engine_instance
    tts_active = False
    if tts_engine_instance:
        try:
            tts_engine_instance.stop()
        except:
            pass
    sd.stop()  # Stop any sounddevice playback
    return jsonify({"status": "stopped"})


@app.route("/run", methods=["POST"])
def run_pipeline():
    data = request.get_json(force=True, silent=True) or {}
    seconds = int(data.get("seconds", DEFAULT_SECONDS))
    stt_name = data.get("stt_model", DEFAULT_STT_MODEL)
    llm_name = data.get("model", DEFAULT_LLM_MODEL)
    system_prompt = data.get("prompt", DEFAULT_PROMPT)
    rate = float(data.get("rate", DEFAULT_RATE))
    tts_engine = data.get("tts_engine", DEFAULT_TTS_ENGINE)

    timings = {}
    t_start = time.time()

    # Load STT once
    model = load_stt_model(stt_name)

    # Record
    t0 = time.time()
    wav_path = record_audio(seconds=seconds)
    timings["record_ms"] = int((time.time() - t0) * 1000)
    if not wav_path:
        return jsonify({"error": "no recording"}), 400

    # STT
    t1 = time.time()
    user_text = transcribe(wav_path, model)
    timings["stt_ms"] = int((time.time() - t1) * 1000)
    if not user_text:
        return jsonify({"error": "no speech detected"}), 400

    # LLM
    t2 = time.time()
    llm_reply = ask_ollama(user_text, llm_name, system_prompt)
    timings["llm_ms"] = int((time.time() - t2) * 1000)

    # TTS (plays locally)
    t3 = time.time()
    actual_engine = speak(llm_reply, engine=tts_engine, rate_factor=rate)
    timings["tts_ms"] = int((time.time() - t3) * 1000)
    timings["total_ms"] = int((time.time() - t_start) * 1000)

    return jsonify(
        {
            "seconds": seconds,
            "stt_model": f"faster-whisper ({stt_name})",
            "llm_model": llm_name,
            "prompt": system_prompt,
            "transcript": user_text,
            "reply": llm_reply,
            "rate": rate,
            "ollama_host": OLLAMA_HOST,
            "tts_engine": actual_engine,
            "timings_ms": timings,
        }
    )


def main():
    global DEFAULT_SECONDS, DEFAULT_LLM_MODEL, DEFAULT_RATE

    parser = argparse.ArgumentParser(description="Local voice UI server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind")
    parser.add_argument("--stt-model", default=DEFAULT_STT_MODEL, help="Default faster-whisper model")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Default Ollama model")
    parser.add_argument("--seconds", type=int, default=DEFAULT_SECONDS, help="Default recording seconds")
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE, help="Default TTS rate")
    args = parser.parse_args()

    # Preload STT model to reduce first-call latency
    load_stt_model(args.stt_model)

    DEFAULT_SECONDS = args.seconds
    DEFAULT_LLM_MODEL = args.llm_model
    DEFAULT_RATE = args.rate

    print(f"[server] Starting on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
