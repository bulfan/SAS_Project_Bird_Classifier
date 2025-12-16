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
from src.analysis.spectral_analysis import SpectralAnalysisPipeline
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
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

cfg_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'configs', 'config.yaml'))
if not os.path.exists(cfg_path):
	cfg_path = os.path.normpath(os.path.join(os.getcwd(), 'configs', 'config.yaml'))
with open(cfg_path, 'r') as f:
	cfg = yaml.safe_load(f)

# Start ffmpeg playback
play_audio_ffmpeg(file_path)
# === 2. Time-Domain Analysis and Plot ===
tda = TimeDomainAnalysis()
tda.load_audio(file_path)
results = tda.run_all(print_results=True)

# Plot waveform
fig = plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(tda.signal)) / tda.sample_rate, tda.signal, linewidth=0.7)
plt.title(f"Waveform: {file_path.split('/')[-1]}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()
try:
	plt.close(fig)
except Exception:
	plt.close('all')

# === 3. Spectrogram (whole file) — optionally downsample for speed ===
import librosa
try:
	# choose a target sample rate for faster STFT (only if original is higher)
	target_sr = min(tda.sample_rate, 22050)
	if target_sr != tda.sample_rate:
		y = librosa.resample(tda.signal.astype(np.float32), orig_sr=tda.sample_rate, target_sr=target_sr)
	else:
		y = tda.signal

	# use a more moderate FFT size for speed (can be adjusted)
	n_fft = 2048
	hop_length = 512
	sap_full = SpectralAnalysisPipeline({'analysis': {'spectral': {'n_fft': n_fft, 'hop_length': hop_length, 'window': 'hann'}}, 'data': {'sample_rate': target_sr}})
	times, frequencies, spectrogram = sap_full.compute_spectrogram(y)
	# convert to dB
	eps = 1e-10
	S_db = 20 * np.log10(spectrogram + eps)
	fig = plt.figure(figsize=(12, 6))
	ax = fig.add_subplot(1, 1, 1)
	im = ax.pcolormesh(times, frequencies, S_db, shading='gouraud', cmap='magma')
	cbar = fig.colorbar(im, ax=ax)
	cbar.set_label('Magnitude (dB)')
	ax.set_xlabel('Time (s)')
	ax.set_ylabel('Frequency (Hz)')
	ax.set_ylim(0, target_sr / 2)
	ax.set_title(f'Spectrogram: {os.path.basename(file_path)} (sr={target_sr})')
	plt.tight_layout()
	plt.show()
	try:
		plt.close(fig)
	except Exception:
		plt.close('all')
except Exception:
	pass


