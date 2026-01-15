import os
import time
import tempfile
import argparse
import numpy as np
import requests
import sounddevice as sd
import soundfile as sf
import pyttsx3
from faster_whisper import WhisperModel


def record_audio(seconds: int = 5, samplerate: int = 16000) -> str:
    """Record from default microphone and save to a temp WAV file."""
    print(f"[record] Recording {seconds}s of audio at {samplerate} Hz...")
    audio = sd.rec(int(seconds * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    path = os.path.join(tempfile.gettempdir(), f"record_{int(time.time())}.wav")
    sf.write(path, audio, samplerate)
    print(f"[record] Saved to {path}")
    return path


def record_audio_manual(samplerate: int = 16000) -> str:
    """Record until user presses Enter; saves to a temp WAV file."""
    print("[record] Manual mode: press Enter to start, Enter again to stop.")
    input("Press Enter to start recording...")

    frames = []

    def callback(indata, frames_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=samplerate, channels=1, dtype="float32", callback=callback)
    stream.start()
    print("[record] Recording... press Enter to stop.")

    if os.name == "nt":
        import msvcrt

        while True:
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b"\r", b"\n"):
                    break
            time.sleep(0.05)
    else:
        input()

    stream.stop()
    stream.close()

    if not frames:
        print("[warn] No audio captured.")
        return ""

    audio = np.concatenate(frames, axis=0)
    path = os.path.join(tempfile.gettempdir(), f"record_{int(time.time())}.wav")
    sf.write(path, audio, samplerate)
    print(f"[record] Saved to {path}")
    return path


def transcribe(path: str, model: WhisperModel) -> str:
    """Transcribe audio with faster-whisper."""
    print("[stt] Transcribing locally with faster-whisper...")
    segments, _ = model.transcribe(path, beam_size=1)
    text = " ".join([s.text.strip() for s in segments]).strip()
    print(f"[stt] Transcript: {text}")
    return text


def ask_ollama(prompt: str, model_name: str, system_prompt: str, host: str) -> str:
    """Send the prompt to local Ollama chat API and return response text."""
    url = f"{host.rstrip('/')}/api/chat"
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


def speak(text: str, rate_factor: float = 1.0) -> None:
    """Speak text via pyttsx3 (Windows SAPI)."""
    print("[tts] Speaking answer locally...")
    engine = pyttsx3.init()
    rate = engine.getProperty("rate")
    engine.setProperty("rate", int(rate * rate_factor))
    engine.say(text)
    engine.runAndWait()


def main():
    parser = argparse.ArgumentParser(description="Speech -> Local STT -> Local LLM (Ollama) -> Local TTS")
    parser.add_argument("--seconds", type=int, default=5, help="Recording length in seconds")
    parser.add_argument("--manual", action="store_true", help="Manual record: press Enter to start/stop")
    parser.add_argument("--stt-model", type=str, default="small", help="faster-whisper model size (tiny/base/small/medium)")
    parser.add_argument("--model", type=str, default="phi", help="Ollama model name (e.g., phi, phi3, mistral:7b)")
    parser.add_argument("--prompt", type=str, default="You are a helpful assistant.", help="System prompt for LLM")
    parser.add_argument("--rate", type=float, default=1.0, help="TTS rate factor (e.g., 0.9 slower, 1.1 faster)")
    parser.add_argument("--ollama-host", type=str, default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
                        help="Ollama host URL")
    args = parser.parse_args()

    # Load STT model once
    print(f"[init] Loading faster-whisper model '{args.stt_model}' ...")
    stt_model = WhisperModel(args.stt_model, device="cpu", compute_type="int8")

    # Record
    if args.manual:
        wav_path = record_audio_manual()
    else:
        wav_path = record_audio(seconds=args.seconds)

    if not wav_path:
        print("[warn] No recording created. Exiting.")
        return

    # STT
    user_text = transcribe(wav_path, stt_model)
    if not user_text:
        print("[warn] No speech detected. Exiting.")
        return

    # LLM
    llm_reply = ask_ollama(user_text, args.model, args.prompt, args.ollama_host)

    # TTS
    speak(llm_reply, rate_factor=args.rate)


if __name__ == "__main__":
    main()
