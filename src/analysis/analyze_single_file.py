# Utility: Find a non-silent segment for FFT analysis
def get_non_silent_segment(samples, segment_length=1024, threshold=0.01):
	total_len = len(samples)
	for start in range(0, total_len - segment_length, segment_length // 2):
		seg = samples[start:start+segment_length]
		rms = np.sqrt(np.mean(seg**2))
		if rms > threshold:
			return seg
	# Fallback: return last segment
	return samples[-segment_length:]
# === FFmpeg Audio Playback Function ===

import subprocess
ffmpeg_proc = None
def play_audio_ffmpeg(file_path):
	global ffmpeg_proc
	ffmpeg_proc = subprocess.Popen([
		'ffplay', '-nodisp', '-autoexit', file_path
	], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

from src.analysis.time_domain import TimeDomainAnalysis
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import yaml

from pydub import AudioSegment
# === 1. Set your audio file path here ===

# Allow user to select the audio file interactively
import tkinter as tk
from tkinter import filedialog

# Hide the main Tkinter window
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
	title="Select an audio file to analyze",
	filetypes=[("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a"), ("All Files", "*.*")]
)

if not file_path:
	print("No file selected. Exiting.")
	exit(1)

# === Load config.yaml ===
with open("../../configs/config.yaml", "r") as f:
	cfg = yaml.safe_load(f)

# Start ffmpeg playback
play_audio_ffmpeg(file_path)
# === 2. Time-Domain Analysis and Plot ===
tda = TimeDomainAnalysis(cfg)
tda.load_audio(file_path)
results = tda.run_all(print_results=True)

# Plot waveform
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(tda.signal)) / tda.sample_rate, tda.signal, linewidth=0.7)
plt.title(f"Waveform: {file_path.split('/')[-1]}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()

# ...existing code...
