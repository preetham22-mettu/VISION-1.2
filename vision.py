import asyncio
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import webbrowser
from datetime import datetime

import cv2
import ollama
import pyaudio
import speech_recognition as sr

try:
    import edge_tts          # natural neural voices (needs internet)
    HAS_EDGE = True
except Exception:
    HAS_EDGE = False


# ==================== CONFIG ==================== #
ASSISTANT_NAME = "Vision"
CREATOR_NAME = "Taha"                 # change to your name
OLLAMA_MODEL = "llama3"               # any model you have pulled with `ollama pull`
MEMORY_FILE = "memory.json"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_FILE = os.path.join(tempfile.gettempdir(), "vision_state.txt")   # interface.py reads this
INTERFACE_PATH = os.path.join(SCRIPT_DIR, "interface.py")
INTERFACE_ARGS = []                   # e.g. ["--hd"] for Full HD, ["--sd"] for lighter/faster

# ---- Voice ----
TTS_ENGINE = "edge"                   # "edge" = natural neural voice (recommended) | "macos" = built-in `say`
EDGE_VOICE = "en-US-ChristopherNeural"  # others: en-US-GuyNeural, en-GB-RyanNeural, en-US-AndrewNeural
EDGE_RATE = "-12%"
EDGE_PITCH = "-18Hz"
EDGE_VOLUME = "+50%"
PLAY_VOLUME = 2.5                     # afplay volume: 1.0 normal, 2.5 loud
ULTRON_EFFECT = True                  # needs `brew install sox`; adds a doubled/metallic effect
ULTRON_FX = ["compand", "0.02,0.2", "6:-70,-60,-20", "-5", "-90", "0.1",
             "pitch", "-80",
             "chorus", "0.7", "0.9", "55", "0.4", "0.25", "2", "-t",
             "equalizer", "2800", "1.0q", "+3",
             "bass", "+3",
             "pad", "0", "0.5",
             "reverb", "20", "50", "60",
             "norm", "-0.5"]
SOX_PATH = shutil.which("sox")
VOICE_CANDIDATES = ["Ralph", "Daniel", "Alex", "Fred"]   # fallback macOS voices
VOICE_RATE = 150

# ---- Double clap ----
CLAP_THRESHOLD = 2000
CLAP_MIN_INTERVAL = 0.1
CLAP_MAX_INTERVAL = 0.8
CLAP_DEBOUNCE = 0.05
AUDIO_CHUNK = 1024
AUDIO_RATE = 44100
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1

# ---- Listening ----
WAKE_LISTEN_TIMEOUT = 4
WAKE_PHRASE_TIME = 3
COMMAND_LISTEN_TIMEOUT = 4
COMMAND_PHRASE_TIME = 8
MAX_AUDIO_FAILURES = 3
CALIBRATION_INTERVAL = 120
WAKE_PHRASES = ["hey vision", "hey visions", "a vision", "hey vishon", "hey vizhan"]

# ==================== STATE ==================== #
conversation_mode = False
initial_clap_activation_done = False
clap_trigger_event = threading.Event()
audio_lock = threading.Lock()
audio_failure_count = 0
last_calibration_time = 0
interface_proc = None


def pick_voice():
    try:
        out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=5).stdout
        for name in VOICE_CANDIDATES:
            if re.search(rf"^{re.escape(name)}\s", out, re.MULTILINE):
                return name
    except Exception:
        pass
    return None


VOICE = pick_voice()


def set_speaking(flag):
    """Tell the interface whether Vision is speaking (interface turns red)."""
    try:
        with open(STATE_FILE, "w") as f:
            f.write("speaking" if flag else "idle")
    except Exception:
        pass


set_speaking(False)

# ==================== SPEECH RECOGNIZER ==================== #
def create_recognizer():
    r = sr.Recognizer()
    r.pause_threshold = 0.6
    r.phrase_threshold = 0.2
    r.non_speaking_duration = 0.3
    r.energy_threshold = 150
    r.dynamic_energy_threshold = True
    return r


recognizer = create_recognizer()


def calibrate_microphone_once(force=False):
    global last_calibration_time
    now = time.time()
    if not force and last_calibration_time and (now - last_calibration_time) < CALIBRATION_INTERVAL:
        return
    try:
        print("Calibrating microphone for ambient noise...")
        with audio_lock:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=0.8)
        recognizer.energy_threshold = max(80, min(250, recognizer.energy_threshold))
        last_calibration_time = time.time()
        print(f"Microphone calibrated (energy threshold {int(recognizer.energy_threshold)})")
    except Exception as e:
        print(f"Microphone calibration warning: {e}")


def reset_audio_pipeline(reason="audio recovery"):
    global recognizer, audio_failure_count
    print(f"[Audio] Resetting recognizer: {reason}")
    recognizer = create_recognizer()
    audio_failure_count = 0


def note_audio_failure(reason):
    global audio_failure_count
    audio_failure_count += 1
    print(f"[Audio] warning ({audio_failure_count}/{MAX_AUDIO_FAILURES}): {reason}")
    if audio_failure_count >= MAX_AUDIO_FAILURES:
        reset_audio_pipeline(reason)


def safe_listen(timeout=5, phrase_time=6, mode="command"):
    """Open the microphone for one bounded listen cycle, then release it."""
    try:
        with audio_lock:
            with sr.Microphone() as source:
                print(f"Listening ({mode})...")
                audio = recognizer.listen(source, timeout=max(1, int(timeout)),
                                          phrase_time_limit=max(1, int(phrase_time)))
        text = recognizer.recognize_google(audio, language="en-IN").strip()
        if text:
            global audio_failure_count
            audio_failure_count = 0
            print(f"Heard ({mode}): {text}")
        return text
    except sr.WaitTimeoutError:
        return ""
    except sr.UnknownValueError:
        note_audio_failure("speech was not understood")
    except sr.RequestError as e:
        note_audio_failure(f"recognition service error: {e}")
    except OSError as e:
        note_audio_failure(f"microphone error: {e}")
    except Exception as e:
        note_audio_failure(f"unexpected listen error: {e}")
    return ""


def listen_for_command(timeout=COMMAND_LISTEN_TIMEOUT, phrase_time=COMMAND_PHRASE_TIME):
    return safe_listen(timeout, phrase_time, "command")


def listen_for_wake_word():
    return safe_listen(WAKE_LISTEN_TIMEOUT, WAKE_PHRASE_TIME, "wake")


# ==================== MEMORY ==================== #
def load_memory():
    default = {"name": "", "notes": [], "last_conversation": []}
    try:
        if not os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(default, f, indent=4)
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            memory = json.load(f)
        for key, val in default.items():
            memory.setdefault(key, val)
        return memory
    except Exception as e:
        print("Memory load error:", e)
        return default


def save_memory(memory):
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=4)
    except Exception as e:
        print("Memory save error:", e)


def save_conversation(user_text, vision_reply):
    memory = load_memory()
    memory["last_conversation"].append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user_text,
        "vision": vision_reply,
    })
    memory["last_conversation"] = memory["last_conversation"][-10:]
    save_memory(memory)


# ==================== INTERFACE (interface.py runs as a separate process) ==================== #
def interface_running():
    return interface_proc is not None and interface_proc.poll() is None


def open_interface():
    global interface_proc
    if interface_running():
        return True
    if not os.path.exists(INTERFACE_PATH):
        speak("Sir, I could not find the interface file")
        return False
    try:
        interface_proc = subprocess.Popen([sys.executable, INTERFACE_PATH] + INTERFACE_ARGS, cwd=SCRIPT_DIR)
        print("Interface opened")
        return True
    except Exception as e:
        print("Interface start error:", e)
        speak("Sir, the interface failed to start")
        return False


def close_interface():
    global interface_proc
    if interface_running():
        try:
            interface_proc.terminate()
            interface_proc.wait(timeout=3)
        except Exception:
            try:
                interface_proc.kill()
            except Exception:
                pass
        interface_proc = None
        return True
    interface_proc = None
    return False


def shutdown_app():
    set_speaking(False)
    close_interface()
    threading.Timer(0.5, os._exit, args=(0,)).start()


# ==================== SPEAK ==================== #
def _tmp(suffix):
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.close()
    return f.name


def _rm(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass


async def _edge_save(text, path):
    comm = edge_tts.Communicate(text, EDGE_VOICE, rate=EDGE_RATE, pitch=EDGE_PITCH, volume=EDGE_VOLUME)
    await comm.save(path)


def speak_edge(text):
    """Neural voice -> optional sox effect -> afplay. Returns False on failure."""
    mp3 = wav = fx = None
    try:
        mp3 = _tmp(".mp3")
        asyncio.run(_edge_save(text, mp3))
        play_path = mp3
        if shutil.which("afconvert"):
            wav = _tmp(".wav")
            subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@44100", mp3, wav], check=True)
            play_path = wav
            if ULTRON_EFFECT and SOX_PATH:
                fx = _tmp(".wav")
                subprocess.run([SOX_PATH, wav, fx] + ULTRON_FX, check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                play_path = fx
        set_speaking(True)
        subprocess.run(["afplay", "-v", str(PLAY_VOLUME), play_path], check=True)
        return True
    except Exception as e:
        print("Edge voice failed (check internet / edge-tts), falling back to macOS voice:", e)
        return False
    finally:
        _rm(mp3, wav, fx)


def speak_macos(text):
    cmd = ["say"] + (["-v", VOICE] if VOICE else []) + ["-r", str(VOICE_RATE), text]
    set_speaking(True)
    subprocess.run(cmd)


def speak(text):
    try:
        text = str(text).replace("\n", " ").strip().lstrip("-")
        if not text:
            return
        print("vision:", text)
        if not (TTS_ENGINE == "edge" and HAS_EDGE and speak_edge(text)):
            speak_macos(text)
    except Exception as e:
        print("Speech error:", e)
    finally:
        set_speaking(False)


def time_greeting():
    hour = datetime.now().hour
    return "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"


# ==================== AI CHAT (Ollama) ==================== #
def ask_local_ai(prompt):
    try:
        memory = load_memory()
        name = memory.get("name", "").strip()
        notes = memory.get("notes", [])
        system = (f"You are {ASSISTANT_NAME}, a powerful and intelligent AI assistant created by {CREATOR_NAME}. "
                  "You speak in a calm, deep, confident tone like a futuristic AI. Address the user as Sir.\n"
                  f"User name: {name or 'Unknown'}\n"
                  f"Important notes: {notes[-5:] if notes else []}\n"
                  "Reply naturally and briefly unless detail is needed.")
        response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ])
        return response["message"]["content"]
    except Exception as e:
        print("Ollama error:", e)
        return "Sorry Sir, AI is not responding."


# ==================== BASIC ACTIONS ==================== #
def open_folder_anywhere(name):
    try:
        result = subprocess.run(["mdfind", name], capture_output=True, text=True)
        folders = [f for f in result.stdout.strip().split("\n") if os.path.isdir(f)]
        if folders:
            speak("Opening folder")
            subprocess.run(["open", folders[0]])
        else:
            speak("Folder not found Sir")
    except Exception as e:
        print("Folder search error:", e)
        speak("Error while opening folder")


def create_folder(name):
    try:
        path = os.path.join(os.path.expanduser("~/Desktop"), name)
        if os.path.exists(path):
            speak("Folder already exists")
        else:
            os.makedirs(path)
            speak(f"Folder {name} created successfully")
    except Exception as e:
        print("Folder error:", e)
        speak("Error creating folder")


def play_song(song):
    """Opens YouTube search results for the song."""
    if not song:
        speak("Please tell me the song name.")
        return
    speak(f"Playing {song} on YouTube")
    webbrowser.open(f"https://www.youtube.com/results?search_query={urllib.parse.quote(song)}")


def take_screenshot():
    os.makedirs("screenshots", exist_ok=True)
    path = f"screenshots/screenshot_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    subprocess.run(["screencapture", "-x", path])       # built into macOS
    subprocess.run(["open", path])
    return path


def take_photo():
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            speak("Camera not found")
            return None
        speak("Opening camera. Get ready.")
        time.sleep(3)
        ret, frame = camera.read()
        camera.release()
        if not ret:
            speak("Failed to capture photo")
            return None
        os.makedirs("photos", exist_ok=True)
        path = f"photos/photo_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
        cv2.imwrite(path, frame)
        subprocess.run(["open", path])
        speak("Photo captured successfully")
        return path
    except Exception as e:
        print("Camera error:", e)
        speak("Error while taking photo")
        return None


PERMISSION_HINT = "Sorry Sir, I could not do that. Please allow Accessibility and Automation permission for Terminal."
VOLUME_STEP = 10


def run_osascript(script):
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("osascript error:", result.stderr.strip())
            return None
        return result.stdout.strip()
    except Exception as e:
        print("osascript exception:", e)
        return None


def _to_int(value, default=None):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def control_volume(mode, value=None):
    mode = str(mode or "").lower()
    step = _to_int(value)
    if mode == "mute":
        speak("Muting")
        run_osascript("set volume with output muted")
        return "Muted"
    if mode == "unmute":
        run_osascript("set volume without output muted")
        speak("Unmuted")
        return "Unmuted"

    current = _to_int(run_osascript("output volume of (get volume settings)"), 50)
    if mode == "set":
        target = step if step is not None else current
    elif mode == "up":
        target = current + (step if step is not None else VOLUME_STEP)
    elif mode == "down":
        target = current - (step if step is not None else VOLUME_STEP)
    else:
        speak("Should I increase or decrease the volume?")
        return "Volume direction not clear"

    target = max(0, min(100, target))
    if run_osascript(f"set volume output volume {target}") is None:
        speak(PERMISSION_HINT)
        return "Volume change failed"
    if target > 0:
        run_osascript("set volume without output muted")
    reply = f"Volume set to {target} percent"
    speak(reply)
    return reply


def get_battery_status():
    try:
        out = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"(\d+)%", out)
        if not m:
            reply = "I could not read the battery level"
        else:
            charging = re.search(r"\bcharging\b|\bfinishing charge\b", out) and "discharging" not in out
            reply = f"Battery is at {m.group(1)} percent and {'charging' if charging else 'not charging'}"
    except Exception as e:
        print("Battery error:", e)
        reply = "I could not read the battery level"
    speak(reply)
    return reply


def open_app(app_name):
    if app_name == "youtube":
        reply = "Opening YouTube"
        speak(reply)
        webbrowser.open("https://youtube.com")
    elif app_name == "safari":
        reply = "Opening Safari"
        speak(reply)
        subprocess.run(["open", "-a", "Safari"])
    elif app_name == "chrome":
        reply = "Opening Chrome"
        speak(reply)
        subprocess.run(["open", "-a", "Google Chrome"])
    else:
        reply = "Sorry Sir, I do not know that app"
        speak(reply)
    return reply


def introduce_yourself():
    name = load_memory().get("name", "").strip()
    known = f" I remember you, {name}." if name else ""
    speak(f"Hello. I am {ASSISTANT_NAME}. Created by {CREATOR_NAME}.{known} "
          "I am not just an assistant. I am fast, intelligent, and always watching. What is your command?")


# ==================== ACTIVE MODE ==================== #
def enter_active_mode():
    global conversation_mode
    conversation_mode = True


def exit_active_mode():
    global conversation_mode
    conversation_mode = False
    print("Vision is idle. Say 'Hey Vision' to resume.")


def run_active_listening_loop():
    print("[Active mode] Continuous listening ON...")
    while conversation_mode:
        command = listen_for_command()
        if command:
            process_command(command)
        else:
            time.sleep(0.05)


# ==================== COMMAND UNDERSTANDING ==================== #
FILLER_PHRASES = ["can you", "could you", "would you", "will you", "can u", "please", "kindly",
                  "zara", "jara", "ek baar", "yaar", "bhai", "mere liye", "for me", "hey", "okay", "ok", "vision"]
OPEN_WORDS = ["open", "launch", "start", "run", "kholo", "khol", "kholna", "kholiye", "kholdo",
              "khol do", "khol de", "chalu", "chalao", "chala do", "chalana"]
APP_ALIASES = {
    "youtube": ["youtube", "you tube"],
    "safari": ["safari"],
    "chrome": ["google chrome", "chrome", "crome"],
}
PLAY_WORDS = ["play", "bajao", "baja do", "bajana", "sunao", "suna do", "sunna", "lagao", "laga do"]
SONG_WORDS = ["song", "songs", "gana", "gaana", "gaane", "music"]
SEARCH_WORDS = ["search", "find", "look up", "dhundo", "dhundh", "dhoondo", "khojo", "khoj"]
CREATE_WORDS = ["create", "make", "new", "generate", "banao", "bana do", "bana", "naya", "nayi"]
CLEAR_WORDS = ["clear", "reset", "delete", "erase", "wipe", "mita", "hata", "bhula", "saaf"]
INTERFACE_WORDS = ["interface", "hologram", "holo gram", "holographic", "holo"]
VOLUME_WORDS = ["volume", "sound", "awaaz", "awaz", "avaaz", "aawaz"]
UP_WORDS = ["increase", "up", "raise", "badhao", "badha", "badhana", "zyada", "jyada", "tez", "louder", "more", "higher", "high"]
DOWN_WORDS = ["decrease", "down", "lower", "reduce", "kam", "kamm", "kum", "dheema", "dheemi", "less", "quieter", "low"]
NUM_WORDS = {"zero": 0, "ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
             "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100}
QUESTION_STARTS = ("what", "why", "how", "who", "when", "where", "which", "tell me", "explain",
                   "kya", "kaun", "kyun", "kyu", "kaise", "kitna", "kitne", "kab", "kahan", "batao")

SLEEP_RE = re.compile(r"\b(?:go to sleep|sleep mode|sleep now|sleep vision|vision sleep|stop listening|"
                      r"so jao|so ja|chup ho jao|chup ho ja|chup raho|take a break)\b")
STOP_RE = re.compile(r"\b(?:(?:stop|exit|quit|shut ?down|terminate|close|band)\s+(?:karo\s+)?vision|"
                     r"vision\s+(?:stop|exit|quit|shut ?down|terminate|band(?:\s+(?:karo|ho\s+jao|kar\s+do))?)|"
                     r"exit program|quit program)\b")


def _pattern(words):
    return r"\b(?:" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + r")\b"


def has_any(text, words):
    return re.search(_pattern(words), text) is not None


def strip_words(text, words):
    return re.sub(r"\s+", " ", re.sub(_pattern(words), " ", text)).strip()


def prepare_command(text):
    """Returns (raw, cmd): raw = lowercase without punctuation, cmd = raw without filler words."""
    raw = re.sub(r"[.,!?;:\"]", " ", text.lower())
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw, strip_words(raw, FILLER_PHRASES)


def find_app(text):
    best, best_len = None, 0
    for app_name, aliases in APP_ALIASES.items():
        for alias in aliases:
            if re.search(rf"\b{re.escape(alias)}\b", text) and len(alias) > best_len:
                best, best_len = app_name, len(alias)
    return best


def looks_like_question(cmd):
    return cmd.startswith(QUESTION_STARTS) or cmd.endswith(" kya hai")


def parse_level_request(cmd):
    """Returns (mode, value): ('set', 0-100) / ('up'|'down', step|None) / (None, None)."""
    direction = "down" if has_any(cmd, DOWN_WORDS) else "up" if has_any(cmd, UP_WORDS) else None
    m = re.search(r"\bby\s+(\d{1,3})\b", cmd) or re.search(r"\b(\d{1,3})\s+(?:percent\s+)?se\b", cmd)
    if m and direction:
        return direction, min(100, int(m.group(1)))
    m = re.search(r"\b(\d{1,3})\b", cmd)
    if m:
        return "set", min(100, int(m.group(1)))
    for word, num in NUM_WORDS.items():
        if re.search(rf"\b{word}\b", cmd):
            return "set", num
    if has_any(cmd, ["full", "max", "maximum"]):
        return "set", 100
    if has_any(cmd, ["half"]):
        return "set", 50
    if has_any(cmd, ["min", "minimum"]):
        return "set", 0
    return (direction, None) if direction else (None, None)


def detect_intent(cmd, raw):
    """Rule-based intent detection. Returns (intent, args) or (None, {})."""
    toks = cmd.split()
    app_found = find_app(cmd)

    if SLEEP_RE.search(raw) or ("sleep" in toks and len(toks) <= 3):
        return "sleep", {}
    if STOP_RE.search(raw):
        return "stop", {}

    if has_any(cmd, ["battery"]):
        return "battery", {}

    if has_any(cmd, INTERFACE_WORDS):
        if has_any(cmd, ["close", "stop", "exit", "quit", "hide", "off", "band", "hatao", "hata", "kill"]):
            return "interface", {"action": "close"}
        return "interface", {"action": "open"}

    if has_any(cmd, ["unmute"]):
        return "volume", {"mode": "unmute", "value": None}
    if has_any(cmd, ["mute"]):
        return "volume", {"mode": "mute", "value": None}
    if has_any(cmd, VOLUME_WORDS):
        mode, value = parse_level_request(cmd)
        if mode:
            return "volume", {"mode": mode, "value": value}

    # memory: name
    if has_any(cmd, ["my name", "mera naam", "mera name", "call me"]):
        if has_any(cmd, ["what", "kya", "do you know", "batao", "pata", "who"]):
            return "get_name", {}
        m = re.search(r"(?:my name is|my name's|call me|mera naam|mera name)\s+(.+?)(?:\s+(?:hai|he|h|hain))?$", cmd)
        return "set_name", {"name": m.group(1).strip() if m else ""}

    # memory: notes / history
    m = re.search(r"\b(?:remember this|remember that|yaad rakho|yaad rakhna|note this|note down|note kar lo|note karo|note kar)\b\s*(?:ki\s+|that\s+)?(.*)$", cmd)
    if m:
        return "add_note", {"note": m.group(1).strip()}
    if has_any(cmd, ["notes"]):
        return ("clear_notes", {}) if has_any(cmd, CLEAR_WORDS) else ("show_notes", {})
    if has_any(cmd, ["what did we talk about", "last conversation", "previous conversation", "chat history",
                     "humne kya baat", "hamne kya baat", "pichli baat", "pehle kya baat"]):
        return "history", {}
    if has_any(cmd, ["memory"]) and has_any(cmd, CLEAR_WORDS):
        return "clear_memory", {}

    if has_any(cmd, ["tell me about yourself", "introduce yourself", "who are you", "what are you",
                     "apne bare mein", "apne baare mein", "apna parichay", "tum kaun ho", "aap kaun ho",
                     "kaun ho tum", "kaun ho aap"]):
        return "introduce", {}

    if has_any(cmd, ["screenshot", "screen shot", "screen capture", "capture screen", "capture the screen"]):
        return "screenshot", {}
    if has_any(cmd, ["photo", "picture", "selfie", "pic"]) and (
        has_any(cmd, ["take", "click", "capture", "nikalo", "nikal", "khicho", "kheencho", "le lo", "lo", "lena", "kar"])
        or "selfie" in toks
    ):
        return "take_photo", {}

    # folders
    if has_any(cmd, ["folder"]) and has_any(cmd, CREATE_WORDS):
        name = ""
        m = re.search(r"(?:named|called)\s+(.+)$", cmd)
        if m:
            name = m.group(1)
        else:
            m = re.search(r"(.+?)\s+(?:naam|name)\s+(?:ka|ki|se)\s+folder", cmd)
            if m:
                name = strip_words(m.group(1), CREATE_WORDS + ["a", "the", "ek", "folder"])
        return "create_folder", {"name": strip_words(name, ["folder", "on desktop", "desktop"]) if name else ""}
    if has_any(cmd, ["folder"]) and has_any(cmd, OPEN_WORDS + ["show", "dikhao", "dikha do", "find", "dhundo"]):
        name = strip_words(cmd, OPEN_WORDS + ["show", "dikhao", "dikha do", "find", "dhundo", "folder", "the", "my", "a",
                                              "ka", "ki", "ko", "me", "mein", "wala", "mera", "meri", "mujhe"])
        return "open_folder", {"name": name}

    # search
    if has_any(cmd, SEARCH_WORDS):
        platform = "youtube" if app_found == "youtube" else "google"
        query = strip_words(cmd, SEARCH_WORDS + ["youtube", "you tube", "google", "on", "in", "for", "at", "about",
                                                 "par", "pe", "per", "mein", "me", "se", "ko", "ka", "ki", "karo",
                                                 "kar", "do", "de", "kardo", "the", "a"])
        return "search", {"platform": platform, "query": query}

    # play song
    if has_any(cmd, PLAY_WORDS) or (has_any(cmd, SONG_WORDS) and has_any(cmd, OPEN_WORDS)):
        song = strip_words(cmd, PLAY_WORDS + OPEN_WORDS + SONG_WORDS + [
            "video", "youtube", "you tube", "on", "pe", "par", "per", "mujhe", "ek", "the", "a", "for", "me", "mein",
            "ka", "ki", "ko", "do", "de", "karo", "kar", "hai", "chahiye", "suna", "baja", "laga", "chala"])
        return "play_song", {"song": song}

    # open app
    if app_found and (has_any(cmd, OPEN_WORDS) or cmd in [a for al in APP_ALIASES.values() for a in al]):
        return "open_app", {"app": app_found}

    return None, {}


# ==================== INTENT EXECUTION ==================== #
def ask_followup(question, timeout=8, phrase_time=8):
    speak(question)
    return listen_for_command(timeout=timeout, phrase_time=phrase_time)


def execute_intent(intent, args, original_command):
    reply = ""

    if intent == "open_app":
        reply = open_app(args.get("app"))

    elif intent == "interface":
        if str(args.get("action", "open")).lower() == "close":
            reply = "Interface closed" if close_interface() else "The interface is not running"
            speak(reply)
        elif interface_running():
            reply = "The interface is already running"
            speak(reply)
        else:
            reply = "Activating interface"
            speak(reply)
            open_interface()

    elif intent == "open_folder":
        name = str(args.get("name", "")).strip() or ask_followup("Which folder should I open?")
        if name:
            open_folder_anywhere(name.lower())
            reply = f"Tried to open folder {name}"
        else:
            reply = "I did not get the folder name"
            speak(reply)

    elif intent == "play_song":
        song = str(args.get("song", "")).strip()
        play_song(song)
        reply = f"Play request processed for {song}"

    elif intent == "search":
        platform = str(args.get("platform", "google")).lower()
        query = str(args.get("query", "")).strip() or ask_followup("What should I search for?", 8, 10)
        if query:
            encoded = urllib.parse.quote(query)
            if platform == "youtube":
                webbrowser.open(f"https://www.youtube.com/results?search_query={encoded}")
                reply = f"Searching YouTube for {query}"
            else:
                webbrowser.open(f"https://www.google.com/search?q={encoded}")
                reply = f"Searching Google for {query}"
        else:
            reply = "I did not get the search query"
        speak(reply)

    elif intent == "screenshot":
        path = take_screenshot()
        reply = f"Screenshot taken and saved at {path}"
        speak("Screenshot taken")

    elif intent == "take_photo":
        path = take_photo()
        reply = f"Photo captured and saved at {path}" if path else "Photo capture failed"

    elif intent == "create_folder":
        name = str(args.get("name", "")).strip() or ask_followup("What should I name the folder?", 10, 12)
        if name:
            create_folder(name.lower().strip())
            reply = f"Folder creation processed for {name}"
        else:
            reply = "I did not get the folder name"
            speak(reply)

    elif intent == "set_name":
        name = str(args.get("name", "")).strip().title()
        if name:
            memory = load_memory()
            memory["name"] = name
            save_memory(memory)
            reply = f"Okay Sir, I will remember that your name is {name}"
        else:
            reply = "Please tell me your name properly"
        speak(reply)

    elif intent == "get_name":
        name = load_memory().get("name", "").strip()
        reply = f"Your name is {name}" if name else "I do not know your name yet"
        speak(reply)

    elif intent == "add_note":
        note = str(args.get("note", "")).strip()
        if note:
            memory = load_memory()
            memory["notes"].append(note)
            save_memory(memory)
            reply = "Okay Sir, I saved that in memory"
        else:
            reply = "Please say something to remember"
        speak(reply)

    elif intent == "show_notes":
        notes = load_memory().get("notes", [])
        if notes:
            speak("Your saved notes are")
            for i, note in enumerate(notes, start=1):
                speak(f"Note {i}: {note}")
            reply = "Notes read"
        else:
            reply = "You do not have any saved notes"
            speak(reply)

    elif intent == "clear_notes":
        memory = load_memory()
        memory["notes"] = []
        save_memory(memory)
        reply = "All your notes have been cleared"
        speak(reply)

    elif intent == "history":
        conversations = load_memory().get("last_conversation", [])
        if conversations:
            speak("Here are your last conversations. Check the terminal.")
            for item in conversations:
                print("-" * 50)
                print("Time:", item.get("time", "Unknown"))
                print("You:", item.get("user", ""))
                print("Vision:", item.get("vision", ""))
            reply = "Conversations shown"
        else:
            reply = "No previous conversation found"
            speak(reply)

    elif intent == "clear_memory":
        save_memory({"name": "", "notes": [], "last_conversation": []})
        reply = "Memory cleared successfully"
        speak(reply)

    elif intent == "introduce":
        introduce_yourself()
        reply = "Introduction completed"

    elif intent == "volume":
        reply = control_volume(args.get("mode"), args.get("value"))

    elif intent == "battery":
        reply = get_battery_status()

    elif intent == "sleep":
        reply = "Understood Sir. Going to sleep. Say Hey Vision to wake me up."
        speak(reply)
        exit_active_mode()

    elif intent == "stop":
        reply = "Shutting down. Goodbye Sir"
        speak(reply)
        save_conversation(original_command, reply)
        raise SystemExit

    return reply


def process_command(command):
    original = command.strip()
    try:
        raw, cmd = prepare_command(original)
        intent, args = detect_intent(cmd, raw)

        if intent is None:
            if len(cmd.split()) < 2:
                print(f"Ignored short/noisy input: {cmd}")
                return
            intent, args = "chat", {}

        print(f"[Intent] {intent} {args}")

        if intent == "chat":
            speak("Thinking Sir")
            reply = ask_local_ai(original)
            speak(reply)
        else:
            reply = execute_intent(intent, args, original)

        save_conversation(original, reply)

    except SystemExit:
        raise
    except Exception as e:
        print("Command error:", e)
        speak("Error Sir")
        save_conversation(original, "Error Sir")


# ==================== DOUBLE CLAP DETECTION ==================== #
def calculate_rms(data):
    count = len(data) // 2
    if count == 0:
        return 0
    shorts = struct.unpack(f"{count}h", data)
    return math.sqrt(sum(s * s for s in shorts) / count)


def find_input_device_index(pa):
    for i in range(pa.get_device_count()):
        try:
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                print(f"[Clap detector] Using device [{i}]: {info.get('name', 'Unknown')}")
                return i
        except Exception:
            continue
    print("[Clap detector] No valid input device found.")
    return None


def check_clap_device_available():
    try:
        pa = pyaudio.PyAudio()
        idx = find_input_device_index(pa)
        pa.terminate()
        return idx is not None
    except Exception as e:
        print(f"[Clap detector] PyAudio check failed: {e}")
        return False


def detect_double_clap():
    """Runs in a thread. After one successful double clap it releases the mic and exits."""
    try:
        pa = pyaudio.PyAudio()
        device_index = find_input_device_index(pa)
        if device_index is None:
            pa.terminate()
            return
        stream = pa.open(format=AUDIO_FORMAT, channels=AUDIO_CHANNELS, rate=AUDIO_RATE, input=True,
                         input_device_index=device_index, frames_per_buffer=AUDIO_CHUNK)
    except Exception as e:
        print(f"[Clap detector] Could not open audio stream: {e}")
        return

    def close_stream():
        for fn in (stream.stop_stream, stream.close, pa.terminate):
            try:
                fn()
            except Exception:
                pass

    print("Clap detector running. Clap twice to activate Vision...")
    clap_count, first_clap_time, last_clap_time = 0, 0.0, 0.0

    while True:
        if initial_clap_activation_done:
            close_stream()
            return
        try:
            data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
        except Exception:
            time.sleep(0.01)
            continue

        now = time.time()
        if calculate_rms(data) > CLAP_THRESHOLD:
            if now - last_clap_time < CLAP_DEBOUNCE:
                continue
            last_clap_time = now
            if clap_count == 0:
                clap_count, first_clap_time = 1, now
            else:
                elapsed = now - first_clap_time
                if elapsed < CLAP_MIN_INTERVAL:
                    continue
                if elapsed <= CLAP_MAX_INTERVAL:
                    print("Double clap confirmed!")
                    clap_trigger_event.set()
                    close_stream()
                    return
                clap_count, first_clap_time = 1, now      # too slow: treat as a new first clap
        elif clap_count == 1 and (now - first_clap_time) > CLAP_MAX_INTERVAL:
            clap_count = 0


# ==================== MAIN ==================== #
def is_wake_phrase(text):
    t = re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", text.lower())).strip()
    return any(p in t for p in WAKE_PHRASES) or t in ("vision", "hey vision")


def start_vision():
    global initial_clap_activation_done

    calibrate_microphone_once(force=True)

    if check_clap_device_available():
        print("Vision is idle. Waiting for a double clap (or press Ctrl+C to quit)...")
        threading.Thread(target=detect_double_clap, daemon=True).start()
        while not initial_clap_activation_done:
            if clap_trigger_event.wait(timeout=0.3):
                clap_trigger_event.clear()
                initial_clap_activation_done = True
                time.sleep(0.5)                       # let the clap thread release the mic
                open_interface()
                time.sleep(1.0)
                speak(f"{time_greeting()} Sir. Vision online. All systems running at optimal performance. What is the plan?")
                enter_active_mode()
                run_active_listening_loop()
    else:
        print("Clap detection unavailable. Say 'Hey Vision' to activate.")
        initial_clap_activation_done = True

    print("[Idle] Say 'Hey Vision' to resume listening.")
    while True:
        try:
            word = listen_for_wake_word()
            if word and is_wake_phrase(word):
                speak("Yes Sir. I am listening.")
                enter_active_mode()
                run_active_listening_loop()
        except SystemExit:
            break
        except Exception as e:
            print("Main loop error:", e)
            reset_audio_pipeline(f"main loop error: {e}")
            time.sleep(0.5)


def main():
    try:
        start_vision()
    except (SystemExit, KeyboardInterrupt):
        pass
    finally:
        shutdown_app()


if __name__ == "__main__":
    main()
