import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import glob
from src.analysis.spectral_analysis import SpectralAnalysisPipeline
from collections import defaultdict

# --- Auto-select first 10 samples from all class folders in data/raw/ ---
data_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')
class_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
file_paths = []
for class_folder in class_folders:
	class_path = os.path.join(data_dir, class_folder)
	class_files = sorted(glob.glob(os.path.join(class_path, '*')))[:20]
	file_paths.extend(class_files)

if not file_paths:
	print("No files found in any class folder. Exiting.")
	exit(1)

spectra = []
labels = []
features = []
class_labels = []

for file_path in file_paths:
	audio = AudioSegment.from_file(file_path)
	samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
	if audio.channels > 1:
		samples = samples.reshape((-1, audio.channels))
		samples = samples.mean(axis=1)
	sample_rate = audio.frame_rate
	n_fft = 4096
	# Use a non-silent segment for fair comparison
	def get_non_silent_segment(samples, segment_length=4096, threshold=0.01):
		total_len = len(samples)
		for start in range(0, total_len - segment_length, segment_length // 2):
			seg = samples[start:start+segment_length]
			rms = np.sqrt(np.mean(seg**2))
			if rms > threshold:
				return seg
		return samples[-segment_length:]
	segment = get_non_silent_segment(samples, segment_length=n_fft, threshold=0.01)
	# Use SpectralAnalysisPipeline for custom FFT
	sap = SpectralAnalysisPipeline({'analysis': {'spectral': {'n_fft': n_fft, 'window': 'hann'}}, 'data': {'sample_rate': sample_rate}})
	freqs, magnitudes = sap.compute_magnitude_spectrum(segment)
	# Normalize magnitudes to match numpy's scale
	magnitudes = magnitudes / np.max(magnitudes) if np.max(magnitudes) > 0 else magnitudes
	spectra.append(magnitudes)
	labels.append(os.path.basename(file_path))
	# --- Feature extraction ---
	dom_idx = np.argmax(magnitudes)
	dom_freq = freqs[dom_idx]
	centroid = np.sum(freqs * magnitudes) / np.sum(magnitudes) if np.sum(magnitudes) > 0 else 0
	bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitudes) / np.sum(magnitudes)) if np.sum(magnitudes) > 0 else 0
	mean_freq = np.mean(freqs)
	median_freq = np.median(freqs)
	features.append({
		'file': os.path.basename(file_path),
		'class': os.path.basename(file_path).split('1')[0].split('2')[0].split('3')[0].split('4')[0].split('5')[0].split('6')[0].split('7')[0].split('8')[0].split('9')[0],
		'dominant_freq': dom_freq,
		'centroid': centroid,
		'bandwidth': bandwidth,
		'mean_freq': mean_freq,
		'median_freq': median_freq
	})
	class_labels.append(features[-1]['class'])

# ...existing code...
