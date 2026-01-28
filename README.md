# Alternative Project - Real-Time Speech Translation System

This project is a real-time speech recognition and translation system that is portable and mobile-compatible. The goal is to instantly convert speech from a lapel microphone or phone microphone into text and translate it to a selected language. All processing runs locally, requires no internet connection, and provides a modern web interface.

## Features

- 🎤 **Real-time speech recognition** (Whisper - local, fast, multilingual)
- 🌐 **Instant translation** (Ollama LLM - local, fast, private)
- 📱 **Mobile compatible** (iPhone 14 Pro and above, MacBook, portable systems)
- 🖥️ **Web interface** (live dB level, transcript, translation, language detection)
- 🔊 **dB level and signal analysis** (live visual bar)
- 🛠️ **Easy configuration** (target language, translation service selection)
- 🔒 **Privacy** (all data processed locally)

## Requirements

- Python 3.8+
- macOS or Linux (tested on MacBook, iPhone)
- [Ollama](https://ollama.com/) (for local LLM)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (for local STT)
- Microphone access

## Installation

1. **Clone the project:**
   ```bash
   git clone https://github.com/derinege/alternative-project.git
   cd alternative-project
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Download Ollama model:**
   ```bash
   ollama pull llama3.2:1b
   ollama run llama3.2:1b
   ```

4. **Start the server:**
   ```bash
   python app.py
   ```

5. **Access the web interface:**
   - [http://localhost:8080](http://localhost:8080)

## Usage

- Click the "Start Listening" button.
- Speak, and the system will automatically transcribe and translate.
- Monitor your audio level in real-time with the dB bar.
- You can select the target language and translation service.

## System Architecture

```
Microphone → Whisper (STT) → Transcript → Ollama (LLM) → Translation → Web Interface
```

## Technical Details

- **Whisper (faster-whisper):**
  - Model: `base` (optimized for mobile, fast and accurate)
  - Accuracy improved with initial_prompt
  - Turkish language forcing and automatic language detection
- **Ollama (Llama 3.2:1b):**
  - Fast translation with local LLM
  - Target language can be selected
- **Web Interface:**
  - Live dB bar, transcript, translation, language detection
  - Modern and mobile-responsive design

## Troubleshooting

- **HTTP 404 Translation Error:** Make sure the Ollama model is fully downloaded and running.
- **Incorrect Transcript:** Move the microphone closer, or upgrade the Whisper model to `base` or higher.
- **dB too low:** Speak louder or change the microphone.

## Development and Contributing

- You can share your code and improvements.
- Open to suggestions for hardware integration (lapel microphone, ESP32, etc.).

## License

This project is developed as open source.
