from crewai.tools import tool
import subprocess
import os
import logging
from typing import Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool("nikud_tool")
def add_nikud_to_hebrew_text(text: str) -> str:
    """Tool: Add nikud to Hebrew text using the Phonikud ONNX model."""
    return add_nikud_to_hebrew_text_impl(text)

def add_nikud_to_hebrew_text_impl(text: str) -> str:
    """
    Adds Nikud (vowel marks) to Hebrew text using Phonikud ONNX model.
    
    Args:
        text (str): Hebrew text without nikud
        
    Returns:
        str: Hebrew text with nikud marks for proper pronunciation
    """
    try:
        # Prefer phonikud-tts library (bundles ONNX Phonikud and helpers)
        from phonikud_tts import Phonikud
        try:
            # If available, use its normalize; otherwise fallback to identity
            from phonikud_tts import phonemize  # not used here but validates install
            def normalize(x: str) -> str:
                return x
        except Exception:
            def normalize(x: str) -> str:
                return x
        
        # Load the ONNX model
        model_path = os.getenv("PHONIKUD_MODEL_PATH", "./phonikud-1.0.int8.onnx")
        
        if not os.path.exists(model_path):
            logger.warning(f"Phonikud model not found at {model_path}. Returning original text without nikud.")
            return text
        
        # Initialize the model
        model = Phonikud(model_path)
        
        # Normalize the Hebrew text first
        normalized_text = normalize(text)
        
        # Add nikud to the text
        vocalized_text = model.add_diacritics(normalized_text)
        
        logger.info(f"Successfully added nikud to: {text[:50]}...")
        return vocalized_text
        
    except ImportError as e:
        logger.warning(f"Phonikud not available ({e}). Returning original text without nikud.")
        return text
    except Exception as e:
        logger.error(f"Error adding nikud: {e}")
        return f"[NIKUD ERROR] {str(e)}"

@tool("nikud_batch_tool") 
def add_nikud_batch(texts: list) -> list:
    """
    Add nikud to multiple Hebrew texts efficiently.
    
    Args:
        texts (list): List of Hebrew text strings
        
    Returns:
        list: List of Hebrew texts with nikud
    """
    results = []
    for text in texts:
        result = add_nikud_to_hebrew_text_impl(text)
        results.append(result)
    return results