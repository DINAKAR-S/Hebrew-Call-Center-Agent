from crewai.tools import tool
import os
import uuid
import logging
from typing import Optional
from pathlib import Path
import torch
import torchaudio as ta
from phonikud import phonemize
from phonikud_onnx import Phonikud

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure output directory exists
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Initialize phonikud model
phonikud_path = os.getenv("PHONIKUD_MODEL_PATH", "./phonikud-1.0.int8.onnx")
phonikud_model = None

if os.path.exists(phonikud_path):
    try:
        phonikud_model = Phonikud(phonikud_path)
        logger.info(f"Phonikud model loaded from {phonikud_path}")
    except Exception as e:
        logger.warning(f"Failed to load Phonikud model: {e}")

# CRITICAL FIX: Monkey patch torch.load for CPU-only systems
original_torch_load = torch.load

def patched_torch_load(f, map_location=None, **kwargs):
    """Patched torch.load that automatically maps CUDA tensors to CPU"""
    if map_location is None:
        # Force CPU mapping for all model loads
        map_location = 'cpu'
    logger.debug(f"Loading with map_location={map_location}")
    return original_torch_load(f, map_location=map_location, **kwargs)

# Apply the patch immediately
torch.load = patched_torch_load
logger.info("✅ Applied torch.load CPU mapping patch for Chatterbox compatibility")

# Initialize Chatterbox TTS model
chatterbox_model = None

def initialize_chatterbox():
    """Initialize Chatterbox multilingual model with CPU compatibility"""
    global chatterbox_model
    if chatterbox_model is None:
        try:
            # Import Chatterbox multilingual TTS
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            
            # Always use CPU since we're patching torch.load
            device = "cpu"
            
            logger.info(f"Loading Chatterbox model on device: {device}")
            chatterbox_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
            logger.info("Chatterbox multilingual model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Chatterbox model: {e}")
            chatterbox_model = None
        finally:
            # Restore original torch.load after model loading
            torch.load = original_torch_load
            logger.info("✅ Restored original torch.load function")

@tool("hebrew_tts_tool")
def convert_hebrew_text_to_speech(text: str, step_number: Optional[int] = None) -> str:
    """Tool: Convert Hebrew text with nikud to a WAV audio file using Chatterbox TTS."""
    return convert_hebrew_text_to_speech_impl(text, step_number)

def convert_hebrew_text_to_speech_impl(text: str, step_number: Optional[int] = None) -> str:
    """
    Converts Hebrew text with nikud into a WAV audio file using Chatterbox multilingual TTS.
    """
    try:
        # Initialize Chatterbox if not already done
        initialize_chatterbox()
        
        if chatterbox_model is None:
            logger.error("Chatterbox model not available, falling back to silent audio")
            return _fallback_tts(text, step_number)
        
        # Step 1: Add nikud if model available
        processed_text = text
        if phonikud_model:
            try:
                text_with_nikud = phonikud_model.add_diacritics(text)
                processed_text = text_with_nikud
                logger.info(f"Added nikud to text: {text[:30]}...")
            except Exception as e:
                logger.warning(f"Phonikud processing failed: {e}, using original text")
        
        # Step 2: Generate speech with Chatterbox
        logger.info(f"Generating Hebrew speech for: {processed_text[:50]}...")
        
        # Generate audio using Hebrew language ID
        wav = chatterbox_model.generate(processed_text, language_id="he")
        
        # Step 3: Save audio file
        if step_number:
            wav_path = OUTPUT_DIR / f"audio_step_{step_number}.wav"
        else:
            wav_path = OUTPUT_DIR / f"tts_{uuid.uuid4().hex}.wav"
        
        # Save with correct sample rate
        ta.save(str(wav_path), wav, chatterbox_model.sr)
        
        logger.info(f"✅ Saved Chatterbox TTS audio to: {wav_path}")
        return str(wav_path)
        
    except Exception as e:
        logger.error(f"Error in Chatterbox Hebrew TTS: {e}")
        return _fallback_tts(text, step_number)

def _fallback_tts(text: str, step_number: Optional[int] = None) -> str:
    """
    Fallback TTS: write a short silent WAV so the pipeline can proceed.
    """
    try:
        import wave
        import struct
        sample_rate = 16000
        duration_seconds = 2
        num_samples = sample_rate * duration_seconds
        
        if step_number:
            filename = OUTPUT_DIR / f"audio_step_{step_number}_fallback.wav"
        else:
            filename = OUTPUT_DIR / f"tts_silent_{uuid.uuid4().hex}.wav"
            
        with wave.open(str(filename), 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            silence = (0 for _ in range(num_samples))
            frames = b''.join(struct.pack('<h', s) for s in silence)
            wav_file.writeframes(frames)
            
        logger.info(f"Silent WAV fallback saved to: {filename}")
        return str(filename)
        
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
