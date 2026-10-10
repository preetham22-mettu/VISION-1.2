# Vision – Voice Assistant with a Holographic Interface

Vision is a simple voice assistant for **macOS**, written in Python. Clap twice (or say **"Hey Vision"**), and an animated holographic interface opens. It stays blue while idle and turns **red while Vision is speaking**. You can talk to it in English, Hindi or Hinglish.

## Features

- **Activation:** double clap, or the wake phrase "Hey Vision"
- **Interface (`interface.py`):** animated hologram, optional hand-gesture control (rotate with one hand, zoom with a two-hand pinch)
- **Basic commands:**
  - Open YouTube, Safari, Chrome
  - Search on Google / YouTube
  - Play a song (opens YouTube results)
  - Take a screenshot or a photo
  - Create a folder on the Desktop / open a folder
  - Volume up / down / set / mute, battery status
  - Open / close the interface
- **Memory:** remembers your name, saves notes, shows recent conversations (`memory.json`)
- **AI chat:** any other question is answered by a local LLM through [Ollama](https://ollama.com) – no API key needed
- **Voice:** natural neural voice via `edge-tts` (falls back to the macOS `say` voice)

## Project structure

```
vision-assistant/
├── vision.py          # main assistant (voice, commands, memory, AI)
├── interface.py       # holographic interface (OpenCV window)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Setup (macOS)

### 1. Prerequisites

- macOS with a microphone (camera optional – only for hand tracking)
- Python **3.10 – 3.12** (recommended; `mediapipe` may not support newer versions yet)
- [Homebrew](https://brew.sh)

```bash
brew install portaudio     # required to build PyAudio
brew install sox           # optional: metallic "Ultron" voice effect
```

### 2. Get the code

```bash
git clone https://github.com/<your-username>/vision-assistant.git
cd vision-assistant
```

### 3. Create a virtual environment and install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Install Ollama and a model (for AI chat)

```bash
brew install ollama
ollama serve &            # or open the Ollama app
ollama pull llama3
```

You can use another model by changing `OLLAMA_MODEL` in `vision.py`.

### 5. Allow macOS permissions

`System Settings → Privacy & Security`:

| Permission | Why |
|---|---|
| **Microphone** | voice commands and clap detection |
| **Camera** | photo command and hand tracking |
| **Accessibility / Automation** | volume control |
| **Screen Recording** | screenshots |

Give these to the app you run Vision from (Terminal, iTerm or VS Code), then **fully quit and reopen** it.

### 6. Run

```bash
python vision.py
```

Clap twice (or, if no input device is found, say "Hey Vision"). To test only the interface:

```bash
python interface.py            # add --sd (lighter) or --hd (Full HD)
```

## Example commands

| Say | What happens |
|---|---|
| "Open YouTube" / "youtube kholo" | opens YouTube |
| "Play Believer" / "believer gana sunao" | opens YouTube results |
| "Search Python tutorials on Google" | Google search |
| "Take a screenshot" | saves to `screenshots/` |
| "Take a photo" | saves to `photos/` |
| "Create a folder named projects" | creates it on the Desktop |
| "Volume 40" / "increase volume" / "mute" | controls volume |
| "Battery" | reads battery status |
| "My name is Alex" / "What is my name" | memory |
| "Remember that my exam is on Friday" | saves a note |
| "Close interface" / "Open interface" | controls the UI window |
| "Go to sleep" | Vision stops listening until "Hey Vision" |
| "Stop Vision" | shuts down |
| anything else (e.g. "Who invented the telephone?") | answered by the local AI |

## Interface controls

| Input | Action |
|---|---|
| Mouse drag | rotate |
| Scroll / `+` `-` | zoom |
| `[` `]` | glow down / up |
| `R` | reset view |
| `F` | fullscreen |
| `Q` / `Esc` | quit |
| One hand (camera) | rotate |
| Pinch with both hands | zoom |

## Configuration

Edit the top of `vision.py`:

- `CREATOR_NAME` – the name Vision says it was created by
- `EDGE_VOICE`, `EDGE_RATE`, `EDGE_PITCH` – voice style
- `CLAP_THRESHOLD` – lower it if claps aren't detected, raise it if it triggers by itself
- `OLLAMA_MODEL` – which local model to use
- `INTERFACE_ARGS` – e.g. `["--hd"]`

## Troubleshooting

- **`PyAudio` fails to install:** run `brew install portaudio`, then `pip install pyaudio` again.
- **Camera not found:** check the Camera permission and close Zoom/FaceTime, then restart your terminal.
- **Hand model download error:** download [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) and place it next to `interface.py`. The interface still works with the mouse.
- **"AI is not responding":** make sure Ollama is running and the model is pulled (`ollama list`).
- **Voice sounds robotic:** `edge-tts` needs internet; offline it falls back to the macOS voice.
- Speech recognition uses Google's free web service, so it also needs internet.

## Notes

- Built for macOS (uses `afplay`, `say`, `open`, `osascript`, `pmset`, `screencapture`).
- `memory.json`, screenshots and photos stay on your machine and are git-ignored.

## License

MIT – see [LICENSE](LICENSE).
