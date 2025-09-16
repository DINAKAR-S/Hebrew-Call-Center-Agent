import os
import sys
import logging
from pathlib import Path
import yaml
from typing import Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew

# Tools imports
from tools.nikud_tool import add_nikud_to_hebrew_text, add_nikud_to_hebrew_text_impl
from tools.tts_tool import convert_hebrew_text_to_speech, convert_hebrew_text_to_speech_impl
from tools.stt_tool import transcribe_hebrew_audio_to_text, transcribe_hebrew_audio_to_text_impl
from tools.transcript_tool import (
    log_conversation_step, 
    create_call_summary, 
    log_system_event,
    initialize_call_session,
    log_conversation_step_impl,
    create_call_summary_impl,
    log_system_event_impl,
    initialize_call_session_impl
)

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO if os.getenv("DEBUG_MODE", "false").lower() == "true" else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@CrewBase
class HebrewCallCenterCrew:
    """Hebrew Call Center Crew for simulating customer support calls"""
    
    agents_config_path = Path(__file__).parent / "config" / "agents.yaml"
    tasks_config_path = Path(__file__).parent / "config" / "tasks.yaml"
    
    def __init__(self):
        """Initialize the crew with conversation tracking"""
        self.conversation_step = 0
        self.max_turns = int(os.getenv("MAX_CONVERSATION_TURNS", "6"))
        self.token_usage = 0
        
        # Load and sanitize agents config to strip string tool names
        try:
            with open(self.agents_config_path, 'r', encoding='utf-8') as f:
                loaded_agents = yaml.safe_load(f) or {}
            # Remove 'tools' entries; we'll inject real Tool objects in code
            for key, cfg in list(loaded_agents.items()):
                if isinstance(cfg, dict) and 'tools' in cfg:
                    cfg = dict(cfg)
                    cfg.pop('tools', None)
                    loaded_agents[key] = cfg
            self.agents_config = loaded_agents
        except Exception as e:
            logger.warning(f"Failed to sanitize agents config: {e}")
            self.agents_config = {}

        # Load tasks config as dict
        try:
            with open(self.tasks_config_path, 'r', encoding='utf-8') as f:
                self.tasks_config = yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load tasks config: {e}")
            self.tasks_config = {}
        
        # Initialize call session
        initialize_call_session_impl()
        logger.info("Hebrew Call Center Crew initialized")

    def _agent_config_without_tools(self, agent_key: str) -> dict:
        """Return agent config dict without 'tools' entry to avoid name resolution issues."""
        cfg = dict(self.agents_config[agent_key])
        if 'tools' in cfg:
            cfg.pop('tools')
        return cfg

    @agent
    def coordinator_agent(self) -> Agent:
        """Orchestrates the entire call flow"""
        return Agent(
            config=self._agent_config_without_tools('coordinator_agent'),
            verbose=True,
            tools=[
                log_conversation_step,
                create_call_summary,
                log_system_event
            ],
            max_iter=25,
            max_execution_time=300
        )

    @agent  
    def customer_agent(self) -> Agent:
        """Customer wanting to cancel TV subscription"""
        return Agent(
            config=self.agents_config['customer_agent'],
            verbose=True,
            max_iter=10
        )

    @agent
    def support_agent(self) -> Agent:
        """Customer support representative"""
        return Agent(
            config=self.agents_config['support_agent'],
            verbose=True,
            max_iter=10
        )

    @agent
    def nikud_agent(self) -> Agent:
        """Adds nikud to Hebrew text"""
        return Agent(
            config=self._agent_config_without_tools('nikud_agent'),
            verbose=True,
            tools=[add_nikud_to_hebrew_text],
            max_iter=5
        )

    @agent
    def tts_agent(self) -> Agent:
        """Converts text to speech"""
        return Agent(
            config=self._agent_config_without_tools('tts_agent'),
            verbose=True,
            tools=[convert_hebrew_text_to_speech],
            max_iter=5
        )

    @agent
    def stt_agent(self) -> Agent:
        """Converts speech to text"""
        return Agent(
            config=self._agent_config_without_tools('stt_agent'),
            verbose=True,
            tools=[transcribe_hebrew_audio_to_text],
            max_iter=5
        )

    @agent
    def transcript_agent(self) -> Agent:
        """Logs all conversation steps"""
        return Agent(
            config=self._agent_config_without_tools('transcript_agent'),
            verbose=True,
            tools=[
                log_conversation_step,
                create_call_summary,
                log_system_event
            ],
            max_iter=5
        )

    @task
    def simulate_hebrew_call_task(self) -> Task:
        """Main task for simulating the Hebrew call"""
        return Task(
            config=self.tasks_config['simulate_hebrew_call'],
            agent=self.coordinator_agent(),
            context=[],  # Will be populated during execution
        )

    @crew
    def crew(self) -> Crew:
        """Defines the crew and its execution process"""
        return Crew(
            agents=[
                self.coordinator_agent(),
                self.customer_agent(),
                self.support_agent(),
                self.nikud_agent(),
                self.tts_agent(),
                self.stt_agent(),
                self.transcript_agent()
            ],
            tasks=[self.simulate_hebrew_call_task()],
            process=Process.sequential,
            verbose=True,
            memory=True,
            max_rpm=30,  # Rate limiting for API calls
            max_execution_time=600  # 10 minutes max
        )

    def process_hebrew_message(self, text: str, speaker: str, step_num: int) -> Dict[str, str]:
        """
        Process a Hebrew message through the complete pipeline:
        Original Text -> Nikud -> TTS -> STT -> Transcript
        """
        try:
            logger.info(f"Processing message {step_num} from {speaker}")
            
            # Step 1: Add nikud
            nikud_text = add_nikud_to_hebrew_text_impl(text)
            
            # Step 2: Convert to speech
            audio_file = convert_hebrew_text_to_speech_impl(nikud_text, step_num)
            
            # Step 3: Transcribe back to text
            transcribed_text = transcribe_hebrew_audio_to_text_impl(audio_file)
            # Fallback if STT couldn't produce text
            if not transcribed_text or not str(transcribed_text).strip():
                transcribed_text = nikud_text or text
            
            # Step 4: Log everything
            log_result = log_conversation_step_impl(
                step_num, speaker, text, nikud_text, 
                audio_file, transcribed_text
            )
            
            return {
                "original": text,
                "nikud": nikud_text,
                "audio_file": audio_file,
                "transcribed": transcribed_text,
                "status": "success"
            }
            
        except Exception as e:
            error_msg = f"Error processing message {step_num}: {str(e)}"
            logger.error(error_msg)
            log_system_event_impl("ERROR", error_msg, {"step": step_num, "speaker": speaker})
            
            return {
                "original": text,
                "error": error_msg,
                "status": "failed"
            }

    def run_conversation_simulation(self) -> Dict[str, any]:
        """
        Runs the complete Hebrew call center simulation
        """
        try:
            logger.info("Starting Hebrew call center simulation")
            
            # Predefined conversation for demonstration
            conversation_script = [
                {"speaker": "customer", "text": "שלום, אני רוצה לבטל את המנוי לטלוויזיה שלי"},
                {"speaker": "support", "text": "שלום, אני מבין שאתה רוצה לבטל את המנוי. האם אתה יכול להסביר לי מה הבעיה?"},
                {"speaker": "customer", "text": "החשבונות יקרים מדי והשירות לא טוב"},
                {"speaker": "support", "text": "אני מבין את הבעיה. בואו נראה איך אפשר לעזור לך. יש לנו הצעות מיוחדות"},
                {"speaker": "customer", "text": "לא מעוניין, אני רוצה לבטל עכשיו"},
                {"speaker": "support", "text": "בסדר, אני אעבד את הביטול. תקבל אישור במייל תוך 24 שעות"}
            ]
            
            results = []
            
            # Process each conversation step
            for i, step in enumerate(conversation_script[:self.max_turns], 1):
                result = self.process_hebrew_message(
                    step["text"], 
                    step["speaker"], 
                    i
                )
                results.append(result)
                
                # Break if processing failed
                if result["status"] == "failed":
                    logger.error(f"Processing failed at step {i}, stopping simulation")
                    break
            
            # Create call summary
            successful_steps = sum(1 for r in results if r["status"] == "success")
            outcome = "Cancellation processed" if successful_steps >= 4 else "Call incomplete"
            
            create_call_summary_impl(
                total_steps=len(results),
                outcome=outcome,
                customer_satisfaction="Resolved" if successful_steps >= 4 else "Unresolved",
                issues_resolved=successful_steps >= 4,
                additional_notes=f"Processed {successful_steps}/{len(results)} steps successfully"
            )
            
            logger.info(f"Call simulation completed. {successful_steps}/{len(results)} steps successful")
            
            return {
                "status": "completed",
                "total_steps": len(results),
                "successful_steps": successful_steps,
                "outcome": outcome,
                "results": results
            }
            
        except Exception as e:
            error_msg = f"Fatal error in conversation simulation: {str(e)}"
            logger.error(error_msg)
            log_system_event_impl("FATAL_ERROR", error_msg)
            
            return {
                "status": "failed",
                "error": error_msg
            }