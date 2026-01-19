"""Helper script to download a Piper voice model."""
import os
import requests
from pathlib import Path

def download_piper_voice(voice_name: str = "en_US-lessac-medium"):
    """Download a Piper voice model from Hugging Face."""
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    
    # Parse voice name (e.g., "en_US-lessac-medium" -> "en/en_US/lessac/medium")
    parts = voice_name.split("-")
    if len(parts) >= 3:
        lang_code = parts[0]  # en
        region = parts[1]      # US
        voice = "-".join(parts[2:-1])  # lessac
        quality = parts[-1]    # medium
        
        voice_path = f"{lang_code}/{lang_code}_{region}/{voice}/{quality}"
    else:
        print(f"Invalid voice name format: {voice_name}")
        print("Expected format: en_US-lessac-medium")
        return None
    
    # Create voices directory
    voices_dir = Path.home() / ".piper" / "voices"
    voices_dir.mkdir(parents=True, exist_ok=True)
    
    # Download model files
    model_file = f"{voice_name}.onnx"
    json_file = f"{voice_name}.onnx.json"
    
    model_url = f"{base_url}/{voice_path}/{model_file}"
    json_url = f"{base_url}/{voice_path}/{json_file}"
    
    model_path = voices_dir / model_file
    json_path = voices_dir / json_file
    
    print(f"Downloading {voice_name}...")
    print(f"Model URL: {model_url}")
    
    # Download model
    if not model_path.exists():
        print("Downloading model file...")
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded: {model_path}")
    else:
        print(f"Model already exists: {model_path}")
    
    # Download JSON
    if not json_path.exists():
        print("Downloading JSON config...")
        response = requests.get(json_url, stream=True)
        response.raise_for_status()
        with open(json_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {json_path}")
    else:
        print(f"JSON already exists: {json_path}")
    
    print(f"\nVoice model ready at: {model_path}")
    print(f"Set environment variable: PIPER_MODEL_PATH={model_path}")
    
    return str(model_path)

if __name__ == "__main__":
    import sys
    voice_name = sys.argv[1] if len(sys.argv) > 1 else "en_US-lessac-medium"
    download_piper_voice(voice_name)
