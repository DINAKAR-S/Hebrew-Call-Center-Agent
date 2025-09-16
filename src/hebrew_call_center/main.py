#!/usr/bin/env python3
"""
Hebrew Call Center AI Agent - Main Entry Point

This script runs the complete Hebrew customer support call simulation
using CrewAI framework with multi-agent coordination.

Author: Dinakar Selvakumar
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

# Import our crew
from crew import HebrewCallCenterCrew

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main_execution.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_prerequisites():
    """Check if all required files and environment variables are set up"""
    
    logger.info("Checking prerequisites...")
    
    # Check environment variables
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("Please set up your .env file with the required API keys")
        return False
    
    # Check if required directories exist
    required_dirs = ["output", "logs", "config", "tools"]
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"Created directory: {dir_name}")
    
    # Check for phonikud model
    model_path = os.getenv("PHONIKUD_MODEL_PATH", "./phonikud-1.0.int8.onnx")
    if not os.path.exists(model_path):
        logger.warning(f"Phonikud model not found at {model_path}")
    
    logger.info("Prerequisites check completed")
    return True

def run_hebrew_call_simulation():
    """Run the complete Hebrew call center simulation"""
    
    logger.info("="*60)
    logger.info("STARTING HEBREW CALL CENTER AI AGENT SIMULATION")
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        # Initialize the crew
        logger.info("Initializing Hebrew Call Center Crew...")
        crew_instance = HebrewCallCenterCrew()
        
        # Run the conversation simulation
        logger.info("Starting conversation simulation...")
        results = crew_instance.run_conversation_simulation()
        
        # Display results
        print("\n" + "="*60)
        print("SIMULATION RESULTS")
        print("="*60)
        
        if results["status"] == "completed":
            print(f"[OK] Status: {results['status'].upper()}")
            print(f"[CALL] Total Steps: {results['total_steps']}")
            print(f"[OK] Successful Steps: {results['successful_steps']}")
            print(f"[RESULT] Outcome: {results['outcome']}")
            
            print(f"\n[FILES] Generated Files:")
            print(f"   - Transcript: output/transcript.txt")
            print(f"   - Audio Files: output/audio_step_*.wav")
            print(f"   - System Log: logs/call_log.txt")
            
            # Display audio files
            audio_files = [f for f in os.listdir("output") if f.startswith("audio_step_") and f.endswith(".wav")]
            if audio_files:
                print(f"\n[AUDIO] Audio Files Generated:")
                for audio_file in sorted(audio_files):
                    file_size = os.path.getsize(f"output/{audio_file}")
                    print(f"   - {audio_file} ({file_size} bytes)")
            
        else:
            print(f"[ERROR] Status: {results['status'].upper()}")
            print(f"[ERROR] Error: {results.get('error', 'Unknown error')}")
        
        execution_time = time.time() - start_time
        print(f"\n[TIME] Total Execution Time: {execution_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Fatal error during simulation: {str(e)}")
        print(f"\n[FATAL ERROR] {str(e)}")
        print("Check logs/main_execution.log for detailed error information")
        
    finally:
        print("\n" + "="*60)
        print("SIMULATION COMPLETED")
        print("="*60)

def display_welcome_message():
    """Display welcome message and project information"""
    
    print("""
==============================================================================
                           HEBREW CALL CENTER AI AGENT                         
                                                                              
  - CrowdWisdomTrading Internship Assessment                                  
  - Multi-Agent CrewAI Hebrew Customer Support Simulation                     
                                                                              
  Features:                                                                   
  - Hebrew Text Processing with Nikud (Phonikud)                              
  - Text-to-Speech with Chatterbox TTS                                        
  - Speech-to-Text with OpenAI Whisper                                        
  - Complete Conversation Transcript Logging                                  
  - Multi-Agent Coordination with Guardrails                                   
                                                                              
==============================================================================
""")

def main():
    """Main entry point"""
    
    display_welcome_message()
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed. Please fix the issues and try again.")
        sys.exit(1)
    
    # Run the simulation
    try:
        run_hebrew_call_simulation()
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        print("\n[STOPPED] Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()