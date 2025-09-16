from crewai.tools import tool
import os
import datetime
import logging
import json
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure output and logs directories exist
OUTPUT_DIR = "output"
LOGS_DIR = "logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

TRANSCRIPT_FILE = os.path.join(OUTPUT_DIR, "transcript.txt")
CALL_LOG_FILE = os.path.join(LOGS_DIR, "call_log.txt")

@tool("transcript_logging_tool")
def log_conversation_step(
    step_number: int,
    speaker: str,
    original_text: str,
    nikud_text: str,
    audio_file: str,
    transcribed_text: str,
    timestamp: Optional[str] = None
) -> str:
    """Tool: Log one conversation step into transcript and logs."""
    return log_conversation_step_impl(
        step_number, speaker, original_text, nikud_text, audio_file, transcribed_text, timestamp
    )

def log_conversation_step_impl(
    step_number: int,
    speaker: str,
    original_text: str,
    nikud_text: str,
    audio_file: str,
    transcribed_text: str,
    timestamp: Optional[str] = None
) -> str:
    """
    Logs a complete conversation step including all processing stages.
    """
    try:
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create detailed log entry
        log_entry = f"""
=== CONVERSATION STEP {step_number} ===
Timestamp: {timestamp}
Speaker: {speaker}
Original Text: {original_text}
Nikud Text: {nikud_text}
Audio File: {audio_file}
Transcribed Text: {transcribed_text}
{'='*50}
"""
        
        # Append to transcript file
        with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
            f.write(log_entry)
        
        logger.info(f"Logged conversation step {step_number} for {speaker}")
        return f"Successfully logged step {step_number}"
        
    except Exception as e:
        error_msg = f"Error logging transcript: {str(e)}"
        logger.error(error_msg)
        return f"[TRANSCRIPT ERROR] {error_msg}"

@tool("call_summary_tool")
def create_call_summary(
    total_steps: int,
    outcome: str,
    customer_satisfaction: str,
    issues_resolved: bool,
    additional_notes: str = ""
) -> str:
    """Tool: Append a human-readable call summary to transcript."""
    return create_call_summary_impl(total_steps, outcome, customer_satisfaction, issues_resolved, additional_notes)

def create_call_summary_impl(
    total_steps: int,
    outcome: str,
    customer_satisfaction: str,
    issues_resolved: bool,
    additional_notes: str = ""
) -> str:
    """
    Creates a comprehensive summary of the entire call.
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        summary = f"""
{'='*60}
CALL SUMMARY
{'='*60}
Call Date: {timestamp}
Total Conversation Steps: {total_steps}
Call Outcome: {outcome}
Customer Satisfaction: {customer_satisfaction}
Issues Resolved: {'Yes' if issues_resolved else 'No'}

Additional Notes: {additional_notes}

Generated Files:
- Transcript: {TRANSCRIPT_FILE}
- Audio Files: {OUTPUT_DIR}/audio_step_*.wav
- Call Log: {CALL_LOG_FILE}

{'='*60}
"""
        
        # Append summary to transcript
        with open(TRANSCRIPT_FILE, "a", encoding="utf-8") as f:
            f.write(summary)
        
        logger.info("Call summary created successfully")
        return "Call summary created successfully"
        
    except Exception as e:
        error_msg = f"Error creating call summary: {str(e)}"
        logger.error(error_msg)
        return f"[SUMMARY ERROR] {error_msg}"

@tool("system_log_tool")
def log_system_event(event_type: str, message: str, data: Optional[Dict[Any, Any]] = None) -> str:
    """Tool: Write a structured system event to call log file."""
    return log_system_event_impl(event_type, message, data)

def log_system_event_impl(event_type: str, message: str, data: Optional[Dict[Any, Any]] = None) -> str:
    """
    Logs system events, errors, and performance metrics.
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write to call log file
        with open(CALL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {event_type}: {message}\n")
            if data:
                f.write(f"Data: {json.dumps(data, indent=2, ensure_ascii=False)}\n")
            f.write("\n")
        
        return "System event logged successfully"
        
    except Exception as e:
        logger.error(f"Error logging system event: {e}")
        return f"[SYSTEM LOG ERROR] {str(e)}"

@tool("initialize_call_log_tool")
def initialize_call_session() -> str:
    """Tool: Start a new call session by resetting transcript and logs."""
    return initialize_call_session_impl()

def initialize_call_session_impl() -> str:
    """
    Initializes a new call session and clears previous logs.
    """
    try:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Clear previous transcript
        with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as f:
            f.write(f"HEBREW CALL CENTER TRANSCRIPT\n")
            f.write(f"Session Started: {timestamp}\n")
            f.write(f"{'='*60}\n\n")
        
        # Initialize call log
        with open(CALL_LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"CALL CENTER SYSTEM LOG\n")
            f.write(f"Session Started: {timestamp}\n")
            f.write(f"{'='*60}\n\n")
        
        logger.info("Call session initialized")
        return "Call session initialized successfully"
        
    except Exception as e:
        error_msg = f"Error initializing call session: {str(e)}"
        logger.error(error_msg)
        return f"[INIT ERROR] {error_msg}"