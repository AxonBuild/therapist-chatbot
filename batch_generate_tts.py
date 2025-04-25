import os
import glob
import tempfile
from openai import OpenAI
from dotenv import load_dotenv

# Optional: for .docx support
try:
    from docx import Document
except ImportError:
    Document = None

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set in environment.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

SCRIPTS_ROOT = "scripts"  # Adjust if your scripts folder is elsewhere
VOICE = "nova"  # Or your preferred default

def read_script_text(filepath):
    if filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    elif filepath.endswith(".docx") and Document:
        doc = Document(filepath)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        print(f"Unsupported file type or missing python-docx: {filepath}")
        return None

def generate_tts(text, out_path, voice=VOICE):
    try:
        with openai_client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=text
        ) as audio_response:
            audio_response.stream_to_file(out_path)
        print(f"Generated TTS: {out_path}")
        return True
    except Exception as e:
        print(f"Error generating TTS for {out_path}: {e}")
        return False

def main():
    for disorder in os.listdir(SCRIPTS_ROOT):
        disorder_dir = os.path.join(SCRIPTS_ROOT, disorder, "scripts")
        if not os.path.isdir(disorder_dir):
            continue
        for script_file in glob.glob(os.path.join(disorder_dir, "SCRIPT_*.*")):
            if script_file.endswith(".mp3"):
                continue  # Skip already-generated audio
            base_name = os.path.splitext(os.path.basename(script_file))[0]
            mp3_path = os.path.join(disorder_dir, f"{base_name}.mp3")
            if os.path.exists(mp3_path):
                print(f"Already exists: {mp3_path}")
                continue
            text = read_script_text(script_file)
            if not text:
                print(f"Skipping (no text): {script_file}")
                continue
            print(f"Generating TTS for {script_file} -> {mp3_path}")
            generate_tts(text, mp3_path, voice=VOICE)

if __name__ == "__main__":
    main()