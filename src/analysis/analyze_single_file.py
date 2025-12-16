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

# === 3. Spectrum Plot ===
segment = get_non_silent_segment(tda.signal, segment_length=4096, threshold=0.01)
sap = SpectralAnalysisPipeline({'analysis': {'spectral': {'n_fft': 4096, 'window': 'hann'}}, 'data': {'sample_rate': tda.sample_rate}})
freqs, mags = sap.compute_magnitude_spectrum(segment)
mags = mags / (mags.max() if mags.max() > 0 else 1.0)
fig = plt.figure(figsize=(10, 4))
# Use binned bars to visualize the magnitude spectrum (reduces visual clutter)
num_bins = 200
freq_min = 0
freq_max = tda.sample_rate / 2
bins = np.linspace(freq_min, freq_max, num_bins + 1)
bin_idx = np.digitize(freqs, bins) - 1
bin_mags = np.zeros(num_bins)
bin_counts = np.zeros(num_bins)
for i, m in enumerate(mags):
	idx = bin_idx[i]
	if 0 <= idx < num_bins:
		bin_mags[idx] += m
		bin_counts[idx] += 1
# avoid divide-by-zero
bin_counts[bin_counts == 0] = 1
bin_mags = bin_mags / bin_counts
bin_centers = 0.5 * (bins[:-1] + bins[1:])
plt.bar(bin_centers, bin_mags, width=(bins[1] - bins[0]) * 0.9, align='center')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Mean normalized magnitude')
plt.title(f"Spectrum (binned): {os.path.basename(file_path)}")
plt.xlim(freq_min, freq_max)
plt.tight_layout()
plt.show()
try:
	plt.close(fig)
except Exception:
	plt.close('all')


