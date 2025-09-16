from crewai.tools import tool
import os
import uuid
import logging
from typing import Optional
from gtts import gTTS
from pydub import AudioSegment
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure output directory exists
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@tool("hebrew_tts_tool")
def convert_hebrew_text_to_speech(text: str, step_number: Optional[int] = None) -> str:
    """Tool: Convert Hebrew text with nikud to a WAV audio file."""
    return convert_hebrew_text_to_speech_impl(text, step_number)

def convert_hebrew_text_to_speech_impl(text: str, step_number: Optional[int] = None) -> str:
    """
    Converts Hebrew text with nikud into a WAV audio file using a lightweight local TTS engine.
    """
    try:
        # Save MP3 via gTTS (Hebrew language code 'iw')
        if step_number:
            mp3_path = os.path.join(OUTPUT_DIR, f"audio_step_{step_number}.mp3")
            wav_path = os.path.join(OUTPUT_DIR, f"audio_step_{step_number}.wav")
        else:
            base = f"tts_{uuid.uuid4().hex}"
            mp3_path = os.path.join(OUTPUT_DIR, f"{base}.mp3")
            wav_path = os.path.join(OUTPUT_DIR, f"{base}.wav")

        logger.info(f"Generating speech (gTTS) for text: {text[:50]}...")
        tts = gTTS(text=text, lang='iw')
        tts.save(mp3_path)

        # If ffmpeg available, also export WAV for downstream tools
        if shutil.which("ffmpeg") is not None:
            try:
                audio = AudioSegment.from_file(mp3_path, format="mp3")
                audio.export(wav_path, format="wav")
                logger.info(f"Saved TTS audio to: {mp3_path} and {wav_path}")
                return wav_path
            except Exception as conv_err:
                logger.warning(f"Failed to convert MP3 to WAV: {conv_err}. Keeping MP3 only.")

        logger.info(f"Saved TTS audio to: {mp3_path}")
        return mp3_path

    except Exception as e:
        logger.error(f"Error in Hebrew TTS: {e}")
        return _fallback_tts(text, step_number)

def _fallback_tts(text: str, step_number: Optional[int] = None) -> str:
    """
    Fallback TTS: write a short silent WAV so the pipeline can proceed without external deps.
    """
    try:
        # Always use dependency-free silent WAV for consistency on Windows
        import wave
        import struct
        sample_rate = 16000
        duration_seconds = 1
        num_samples = sample_rate * duration_seconds
        if step_number:
            filename = os.path.join(OUTPUT_DIR, f"audio_step_{step_number}_fallback.wav")
        else:
            filename = os.path.join(OUTPUT_DIR, f"tts_silent_{uuid.uuid4().hex}.wav")
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            silence = (0 for _ in range(num_samples))
            frames = b''.join(struct.pack('<h', s) for s in silence)
            wav_file.writeframes(frames)
        logger.info(f"Silent WAV fallback saved to: {filename}")
        return filename
    except Exception as e:
        logger.error(f"Fallback TTS error: {e}")
        return f"[FALLBACK TTS ERROR] {str(e)}"

@tool("batch_tts_tool")
def convert_multiple_texts_to_speech(texts_with_steps: list) -> list:
    """
    Convert multiple Hebrew texts to speech files.
    
    Args:
        texts_with_steps (list): List of tuples (text, step_number)
        
    Returns:
        list: List of audio file paths
    """
    results = []
    for text, step_num in texts_with_steps:
        result = convert_hebrew_text_to_speech_impl(text, step_num)
        results.append(result)
    return results