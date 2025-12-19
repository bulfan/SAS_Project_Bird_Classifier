"""
Feature Extractor for Bird Sound Classification.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List


class FeatureExtractor:
    """
    Extract features from preprocessed audio files.
    Features extracted:
    - Time domain: RMS energy, zero-crossing rate
    - Frequency domain: spectral centroid, bandwidth, peak frequency
    """
    
    def __init__(self, cfg=None, sample_rate: int = 22050):
        """Initialize feature extractor."""
        if cfg is not None:
            self.sample_rate = cfg.preprocessing.sample_rate
        else:
            self.sample_rate = sample_rate
    
    def extract_single(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract features from a single audio file."""
        features = []
        y = audio_data.astype(np.float64)
        
        # --- Time Domain Features ---
        
        # 1. RMS Energy (Volume/Loudness indicator)
        rms = np.sqrt(np.mean(y ** 2))
        features.append(rms)
        
        # 2. Zero Crossing Rate (Rough frequency/noisiness indicator)
        zcr = np.sum(np.abs(np.diff(np.sign(y)))) / (2 * len(y))
        features.append(zcr)
        
        # --- Frequency Domain Features ---
        
        # Compute magnitude spectrum only for positive frequencies
        spectrum = np.abs(np.fft.rfft(y))
        freqs = np.fft.rfftfreq(len(y), d=1.0 / self.sample_rate)
        
        # Normalize spectrum as probability distribution
        spec_sum = spectrum.sum()
        if spec_sum > 0:
            spec_prob = spectrum / spec_sum
        else:
            spec_prob = np.ones_like(spectrum) / len(spectrum)
        
        # 3. Spectral Centroid (Center of mass of the spectrum)
        centroid = np.sum(freqs * spec_prob)
        features.append(centroid)
        
        # 4. Spectral Bandwidth (Spread of frequencies around centroid)
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spec_prob))
        features.append(bandwidth)
        
        # 5. Peak Frequency (Dominant frequency)
        peak_freq = freqs[np.argmax(spectrum)]
        features.append(peak_freq)
        
        return np.array(features, dtype=np.float32)
    
    def get_feature_names(self) -> List[str]:
        """Return names of extracted features."""
        return [
            "rms_energy",
            "zero_crossing_rate", 
            "spectral_centroid",
            "spectral_bandwidth",
            "peak_frequency"]
    
    def process_split(self, dataset, processed_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from all files in a dataset split."""
        X = []
        y = []
        processed_dir = Path(processed_dir)
        for original_path, label_idx in dataset.samples:
            original_path = Path(original_path)
            class_name = dataset.get_class_name(label_idx)
            
            # Construct path to processed file
            filename = original_path.stem + ".npy"
            processed_path = processed_dir / class_name / filename
            
            if not processed_path.exists():
                print(f"   Warning: File not found: {processed_path}")
                continue
            
            try:
                audio_data = np.load(processed_path)
                features = self.extract_single(audio_data)
                X.append(features)
                y.append(label_idx)
                
            except Exception as e:
                print(f"   Error processing {processed_path}: {e}")
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
