from crewai.tools import tool
import os
import logging
import whisper
from typing import Optional
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load Whisper model once at module level for efficiency
try:
    whisper_model = whisper.load_model("small")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    whisper_model = None

@tool("hebrew_stt_tool")
def transcribe_hebrew_audio_to_text(audio_file_path: str) -> str:
    """Tool: Transcribe Hebrew WAV audio file to text using Whisper."""
    return transcribe_hebrew_audio_to_text_impl(audio_file_path)

def transcribe_hebrew_audio_to_text_impl(audio_file_path: str) -> str:
    """
    Transcribes Hebrew audio file back to text using OpenAI Whisper.
    """
    try:
        # Check if audio file exists
        if not os.path.exists(audio_file_path):
            error_msg = f"Audio file not found: {audio_file_path}"
            logger.warning(error_msg)
            return ""
        
        # Check if Whisper model is loaded
        if whisper_model is None:
            logger.warning("Whisper model not loaded; returning empty transcription")
            return ""

        # Ensure FFmpeg is available (required by Whisper audio loader)
        if shutil.which("ffmpeg") is None:
            logger.warning("FFmpeg not found on PATH; skipping transcription. Install FFmpeg and try again.")
            return ""
        
        logger.info(f"Transcribing audio file: {audio_file_path}")
        
        # Transcribe the audio with Hebrew language specification
        result = whisper_model.transcribe(
            audio_file_path, 
            language="he",  # Hebrew language code
            task="transcribe"
        )
        
        transcribed_text = result["text"].strip()
        
        logger.info(f"Successfully transcribed: {transcribed_text[:50]}...")
        return transcribed_text
        
    except Exception as e:
        logger.warning(f"Error transcribing audio: {e}")
        return ""

@tool("stt_with_confidence_tool")
def transcribe_with_confidence(audio_file_path: str) -> dict:
    """Tool: Transcribe Hebrew audio and return text with metadata."""
    return transcribe_with_confidence_impl(audio_file_path)

def transcribe_with_confidence_impl(audio_file_path: str) -> dict:
    """
    Transcribes Hebrew audio and returns result with confidence scores.
    """
    try:
        if not os.path.exists(audio_file_path):
            return {"error": f"Audio file not found: {audio_file_path}"}
        
        if whisper_model is None:
            return {"error": "Whisper model not loaded"}
        
        logger.info(f"Transcribing with confidence: {audio_file_path}")
        
        # Transcribe with additional options for detailed results
        result = whisper_model.transcribe(
            audio_file_path,
            language="he",
            task="transcribe",
            word_timestamps=True,
            temperature=0.0  # More deterministic results
        )
        
        return {
            "text": result["text"].strip(),
            "language": result.get("language", "he"),
            "segments": len(result.get("segments", [])),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error in confidence transcription: {e}")
        return {"error": str(e), "success": False}

@tool("batch_stt_tool")
def transcribe_multiple_audio_files(audio_file_paths: list) -> list:
    """
    Transcribe multiple Hebrew audio files efficiently.
    
    Args:
        audio_file_paths (list): List of audio file paths
        
    Returns:
        list: List of transcribed texts
    """
    results = []
    for audio_path in audio_file_paths:
        result = transcribe_hebrew_audio_to_text_impl(audio_path)
        results.append(result)
    return results