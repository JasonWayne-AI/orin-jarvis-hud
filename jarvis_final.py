import os
import sys
import time
import math
import queue
import threading
import subprocess
import requests
import tkinter as tk
from tkinter import ttk

# Optional local/cloud model client fallbacks
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

# Audio & Speech dependencies (ensure whisper, sounddevice/pyaudio, kokoro are installed)
import sounddevice as sd
import numpy as np

# ----------------- CONFIGURATION -----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "llama3.2:latest"
LOCATION = "Old Westbury, NY"
SAMPLE_RATE = 16000

# ----------------- SYSTEM LOG & STATE QUEUES -----------------
log_queue = queue.Queue()
audio_queue = queue.Queue()
music_process = None

def log_event(message: str):
    timestamp = time.strftime("[%H:%M:%S]")
    log_queue.put(f"{timestamp} {message}")
    print(f"{timestamp} {message}")

# ----------------- HARDWARE & EXTERNAL TOOLS -----------------
def fetch_weather(location=LOCATION):
    try:
        url = f"https://wttr.in/{location.replace(' ', '+')}?format=%C+%t+(feels+%f)+Humidity:+%h"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            return resp.text.strip()
    except Exception as e:
        log_event(f"Weather lookup error: {e}")
    return "Weather unavailable"

def play_youtube_audio(query: str):
    global music_process
    stop_audio()
    log_event(f"Streaming audio for: {query}")
    cmd = f'yt-dlp -f bestaudio "ytsearch1:{query}" -o - | mpv --no-video -'
    music_process = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)

def stop_audio():
    global music_process
    if music_process:
        try:
            os.killpg(os.getpgid(music_process.pid), 9)
            music_process = None
            log_event("Audio playback stopped.")
        except Exception as e:
            log_event(f"Stop audio error: {e}")

# ----------------- LLM / INTELLIGENCE ROUTER -----------------
def query_llm(prompt: str) -> str:
    # Check for music command
    if prompt.lower().startswith("play "):
        song = prompt[5:].strip()
        play_youtube_audio(song)
        return f"Playing {song}."
    elif prompt.lower() in ["stop music", "stop audio", "quiet"]:
        stop_audio()
        return "Audio stopped."

    # Gemini online integration
    if GEMINI_API_KEY and genai:
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            system_instruction = (
                "You are JARVIS, an edge-computed onboard AI butler. "
                "Provide concise, intelligent, and refined responses. Support both English and Mandarin."
            )
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.7,
                )
            )
            return response.text.strip()
        except Exception as e:
            log_event(f"Gemini API offline/failed: {e}. Falling back to Ollama.")

    # Local Ollama fallback
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }
        resp = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=20)
        if resp.status_code == 200:
            return resp.json().get("response", "").strip()
    except Exception as e:
        log_event(f"Ollama inference error: {e}")

    return "System offline. Unable to process command."

# ----------------- SPEECH SYNTHESIS (TTS) -----------------
def speak(text: str):
    log_event(f"JARVIS: {text}")
    # Kokoro-82M / Piper / espeak fallback
    try:
        # Example using Kokoro CLI or local pipe:
        # kokoro-tts --text "{text}" | aplay
        subprocess.run(["espeak", text], check=False)
    except Exception as e:
        log_event(f"TTS Error: {e}")

# ----------------- TKINTER HUD INTERFACE -----------------
class JarvisHUD:
    def __init__(self, root):
        self.root = root
        self.root.title("JARVIS HUD - Jetson Orin NX")
        self.root.geometry("1024x600")
        self.root.configure(bg="#050811")

        self.canvas_width = 380
        self.canvas_height = 380
        self.angle = 0

        # Top Header Bar
        self.header_frame = tk.Frame(root, bg="#0b1220", height=40)
        self.header_frame.pack(fill=tk.X, side=tk.TOP)

        self.title_label = tk.Label(
            self.header_frame, text="JETSON ORIN NX // ONBOARD TELEMETRY",
            font=("Helvetica", 11, "bold"), fg="#38bdf8", bg="#0b1220"
        )
        self.title_label.pack(side=tk.LEFT, padx=15, pady=8)

        self.weather_label = tk.Label(
            self.header_frame, text="Loading Weather...",
            font=("Helvetica", 10), fg="#94a3b8", bg="#0b1220"
        )
        self.weather_label.pack(side=tk.RIGHT, padx=15, pady=8)

        # Main Workspace
        self.main_frame = tk.Frame(root, bg="#050811")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Left Column: Particle Orb Canvas
        self.canvas = tk.Canvas(
            self.main_frame, width=self.canvas_width, height=self.canvas_height,
            bg="#050811", highlightthickness=0
        )
        self.canvas.pack(side=tk.LEFT, padx=20)

        # Right Column: Terminal Logs & Input
        self.right_frame = tk.Frame(self.main_frame, bg="#050811")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            self.right_frame, bg="#0b1220", fg="#38bdf8",
            insertbackground="#38bdf8", font=("Courier", 10),
            borderwidth=1, relief="flat", height=18
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Bottom Interaction Box
        self.entry_frame = tk.Frame(self.right_frame, bg="#050811")
        self.entry_frame.pack(fill=tk.X)

        self.cmd_entry = tk.Entry(
            self.entry_frame, bg="#0b1220", fg="#f8fafc",
            insertbackground="#38bdf8", font=("Helvetica", 11),
            borderwidth=1, relief="flat"
        )
        self.cmd_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10), ipady=4)
        self.cmd_entry.bind("<Return>", self.handle_input)

        self.send_btn = tk.Button(
            self.entry_frame, text="TRANSMIT", command=self.handle_input,
            bg="#0284c7", fg="#ffffff", activebackground="#0369a1",
            font=("Helvetica", 10, "bold"), relief="flat", padx=15
        )
        self.send_btn.pack(side=tk.RIGHT)

        # Start loops
        self.update_weather()
        self.render_orb()
        self.poll_logs()

    def render_orb(self):
        self.canvas.delete("all")
        cx, cy = self.canvas_width // 2, self.canvas_height // 2
        radius = 110
        num_particles = 40

        # Draw 3D-projected rotating rings
        for i in range(num_particles):
            theta = (2 * math.pi / num_particles) * i + self.angle
            phi = (2 * math.pi / num_particles) * i * 2

            x = cx + radius * math.cos(theta)
            y = cy + (radius * 0.4) * math.sin(theta)

            depth = math.sin(theta)
            size = 2 + (depth + 1) * 2
            alpha_color = "#38bdf8" if depth > 0 else "#1e3a8a"

            self.canvas.create_oval(
                x - size, y - size, x + size, y + size,
                fill=alpha_color, outline=""
            )

        self.angle += 0.04
        self.root.after(33, self.render_orb)

    def handle_input(self, event=None):
        query = self.cmd_entry.get().strip()
        if not query:
            return
        self.cmd_entry.delete(0, tk.END)
        log_event(f"User: {query}")

        def worker():
            reply = query_llm(query)
            speak(reply)

        threading.Thread(target=worker, daemon=True).start()

    def update_weather(self):
        def worker():
            w_text = fetch_weather()
            self.weather_label.config(text=w_text)
        threading.Thread(target=worker, daemon=True).start()
        self.root.after(300000, self.update_weather)  # Refresh every 5 min

    def poll_logs(self):
        while not log_queue.empty():
            msg = log_queue.get_nowait()
            self.log_text.insert(tk.END, msg + "\n")
            self.log_text.see(tk.END)
        self.root.after(100, self.poll_logs)

# ----------------- MAIN EXECUTION -----------------
if __name__ == "__main__":
    log_event("JARVIS Core Initialized on Jetson Orin NX.")
    root = tk.Tk()
    app = JarvisHUD(root)
    root.mainloop()
