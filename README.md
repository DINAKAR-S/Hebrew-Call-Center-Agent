# CrowdWisdomTrading AI Agent - Hebrew Call Center

This project implements a multi-agent CrewAI system that simulates a Hebrew customer support call for TV subscription cancellation.

## Features

- Hebrew text processing with Nikud (vowel marks) using Phonikud (optional)
- Text-to-Speech with gTTS (Google Text-to-Speech) by default
- Speech-to-Text with OpenAI Whisper
- Multi-agent conversation flow with guardrails
- Complete call transcript logging
- Audio file generation for each conversation step

## Installation

1) Create/activate virtual environment and install dependencies:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Download Phonikud ONNX model for Hebrew nikud:
```bash
# Windows PowerShell
curl -L -o phonikud-1.0.int8.onnx https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx
# macOS/Linux
wget https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx -O phonikud-1.0.int8.onnx
```

3) Set environment variables:
```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "your-openai-api-key"  # required for Whisper if using OpenAI API elsewhere
$env:PHONIKUD_MODEL_PATH = "$PWD\phonikud-1.0.int8.onnx"  # optional, only for nikud

# macOS/Linux
export OPENAI_API_KEY="your-openai-api-key"
export PHONIKUD_MODEL_PATH="./phonikud-1.0.int8.onnx"
```

## Project Structure

```
hebrew_call_center/
├── src/
│   └── hebrew_call_center/
│       ├── config/
│       │   ├── agents.yaml
│       │   └── tasks.yaml
│       ├── tools/
│       │   ├── nikud_tool.py
│       │   ├── tts_tool.py
│       │   ├── stt_tool.py
│       │   └── transcript_tool.py
│       ├── crew.py
│       └── main.py
├── output/
├── logs/
├── phonikud-1.0.int8.onnx
└── requirements.txt
```

## Usage

Run the complete Hebrew call center simulation:

```bash
python src/hebrew_call_center/main.py
```

## Output

The system generates:
- `output/transcript.txt` - Complete Hebrew conversation log
- `output/audio_step_*.wav` - Audio files for each conversation step
- `logs/call_log.txt` - Detailed execution logs

## Agent Architecture

1. **Coordinator Agent** - Orchestrates the entire call flow
2. **Customer Agent** - Plays the role of the client wanting to cancel TV subscription
3. **Support Agent** - Customer service representative handling the call
4. **Nikud Agent** - Adds Hebrew vowel marks for pronunciation (optional)
5. **TTS Agent** - Converts Hebrew text to speech (gTTS-first)
6. **STT Agent** - Transcribes speech back to text
7. **Transcript Agent** - Logs all conversation steps

## Guardrails

- Maximum 6 conversation turns to prevent infinite loops
- Token usage monitoring and logging
- Error handling for all external tool calls
- Conversation flow validation

## Requirements

- Python 3.10-3.13
- OpenAI API key (for Whisper and LLM usage)
- Sufficient disk space for audio files
- Internet connection for model downloads

## Notes
- If gTTS is blocked on your network, the code generates a short silent WAV so the pipeline still completes.
- To switch to higher-quality TTS later (e.g., Chatterbox), add and install the library, then set the preferred TTS in `tts_tool.py`.
