"""
Time-Domain Analysis for audio signals.

This module provides a class to compute various time-domain features:
- Mean (μ): DC offset / average amplitude
- Standard Deviation (σ): Variation in the signal
- RMS: Root Mean Square (effective amplitude)
- RMS in dB: RMS converted to decibels (dBFS)
- Average amplitude, min/max amplitude, crest factor

Formulas:
    Mean: μ = (1/N) * Σx[i]
    Std Dev: σ = sqrt( (1/(N-1)) * Σ(x[i] - μ)² )
    RMS: sqrt( (1/N) * Σx[i]² )
"""

import numpy as np
from typing import Dict, Optional
import librosa


class TimeDomainAnalysis:
        
    def __init__(self, signal: Optional[np.ndarray] = None, sample_rate: int = 22050):
        self.signal = signal
        self.sample_rate = sample_rate
        self.results: Dict = {}

    def compute_avg_amplitude(self) -> float:
        """
        Compute the average absolute amplitude of the signal.
        Returns:
            The average amplitude
        """
        if self.signal is None:
            raise ValueError("No signal loaded. Call load_audio() or pass signal to constructor.")
        return float(np.mean(np.abs(self.signal)))

    def compute_min_amplitude(self) -> float:
        """
        Compute the minimum amplitude of the signal.
        Returns:
            The minimum amplitude
        """
        if self.signal is None:
            raise ValueError("No signal loaded. Call load_audio() or pass signal to constructor.")
        return float(np.min(self.signal))

    def compute_max_amplitude(self) -> float:
        """
        Compute the maximum amplitude of the signal.
        Returns:
            The maximum amplitude
        """
        if self.signal is None:
            raise ValueError("No signal loaded. Call load_audio() or pass signal to constructor.")
        return float(np.max(self.signal))

    def compute_crest_factor(self) -> float:
        """
        Compute the crest factor of the signal (max amplitude / RMS).
        Returns:
            The crest factor
        """
        if self.signal is None:
            raise ValueError("No signal loaded. Call load_audio() or pass signal to constructor.")
        rms = self.compute_rms()
        if rms == 0:
            return float('inf')
        return float(np.max(np.abs(self.signal)) / rms)

    def load_audio(self, file_path: str, sr: Optional[int] = None) -> np.ndarray:
        """
        Load an audio file.
        Args:
            file_path: Path to the audio file
            sr: Target sample rate (uses instance sample_rate if None)
        Returns:
            The loaded audio signal
        """
        if sr is None:
            sr = self.sample_rate
        self.signal, self.sample_rate = librosa.load(file_path, sr=sr, mono=True)
        return self.signal

    def compute_mean(self) -> float:
        """
        Compute the mean (μ) of the signal.
        μ = (1/N) * Σx[i] for i = 0 to N-1
        Returns:
            The mean value μ
        """
        if self.signal is None:
            raise ValueError("No signal loaded. Call load_audio() or pass signal to constructor.")
        N = len(self.signal)
        return float(np.sum(self.signal) / N)

    def compute_standard_deviation(self) -> float:
        """
        Compute the Standard Deviation (σ) of the signal.
        σ = sqrt( (1/(N-1)) * Σ(x[i] - μ)² )
        Returns:
            The standard deviation σ
        """
        if self.signal is None:
            raise ValueError("No signal loaded. Call load_audio() or pass signal to constructor.")
        N = len(self.signal)
        if N <= 1:
            return 0.0
        mu = self.compute_mean()
        squared_diff_sum = np.sum((self.signal - mu) ** 2)
        variance = squared_diff_sum / (N - 1)
        return float(np.sqrt(variance))

    def compute_rms(self) -> float:
        """
        Compute the Root Mean Square (RMS) of the signal.
        RMS = sqrt( (1/N) * Σx[i]² )
        Returns:
            The RMS value
        """
        if self.signal is None:
            raise ValueError("No signal loaded. Call load_audio() or pass signal to constructor.")
        return float(np.sqrt(np.mean(self.signal ** 2)))

    def compute_rms_db(self, eps: float = 1e-12) -> float:
        """
        Compute RMS in decibels (dBFS).
        RMS_dB = 20 * log10(RMS)
        Args:
            eps: Small value to avoid log(0)
        Returns:
            RMS in dBFS (0 dB = full scale)
        """
        rms = self.compute_rms()
        return float(20 * np.log10(rms + eps))

    def run_all(self, print_results: bool = True) -> Dict:
        if self.signal is None:
            raise ValueError("No signal loaded. Call load_audio() or pass signal to constructor.")
        self.results = {
            'mean': self.compute_mean(),
            'std': self.compute_standard_deviation(),
            'rms': self.compute_rms(),
            'rms_db': self.compute_rms_db(),
            'avg_amplitude': self.compute_avg_amplitude(),
            'min_amplitude': self.compute_min_amplitude(),
            'max_amplitude': self.compute_max_amplitude(),
            'crest_factor': self.compute_crest_factor(),
            'N': len(self.signal),
            'duration': len(self.signal) / self.sample_rate
        }
        if print_results:
            print(f"   - Mean (μ): {self.results['mean']:.6f}")
            print(f"   - Standard Deviation (σ): {self.results['std']:.6f}")
            print(f"   - RMS: {self.results['rms']:.6f}")
            print(f"   - RMS (dB): {self.results['rms_db']:.2f} dBFS")
            print(f"   - Avg Amplitude: {self.results['avg_amplitude']:.6f}")
            print(f"   - Min Amplitude: {self.results['min_amplitude']:.6f}")
            print(f"   - Max Amplitude: {self.results['max_amplitude']:.6f}")
            print(f"   - Crest Factor: {self.results['crest_factor']:.6f}")
        return self.results

    def compute_call_length(self, mode: str = 'total', top_db: float = 40.0, min_call_duration: float = 0.02,
                            low_freq: float | None = 1000.0, high_freq: float | None = 8000.0):
        """
        Detect non-silent segments (calls) using librosa and return durations.

        Args:
            mode: 'total' returns total non-silent duration, 'longest' returns longest segment,
                  'segments' returns a list of segment durations (seconds).
            top_db: the threshold (in dB) below reference to consider as silence for librosa.effects.split.
            min_call_duration: minimum duration (seconds) to consider a detected segment a call.

        Returns:
            float or list: depending on `mode`.
        """
        if self.signal is None:
            raise ValueError("No signal loaded. Call load_audio() or pass signal to constructor.")
        if mode not in ('total', 'longest', 'segments'):
            raise ValueError("mode must be one of 'total', 'longest', 'segments'")

        # Prepare signal
        y = np.asarray(self.signal, dtype=float)
        # Optional bandpass filter to reduce background noise and focus on call band
        if low_freq is not None and high_freq is not None and low_freq < high_freq:
            try:
                from scipy.signal import butter, filtfilt
                nyq = 0.5 * self.sample_rate
                low = float(low_freq) / nyq
                high = float(high_freq) / nyq
                b, a = butter(4, [low, high], btype='band')
                y = filtfilt(b, a, y)
            except Exception:
                # If scipy not available or filtering fails, continue with raw signal
                pass

        # Use librosa.effects.split which works in dB relative to maximum
        intervals = librosa.effects.split(y, top_db=float(top_db), frame_length=4096, hop_length=1024)
        durations = []
        for s, e in intervals:
            dur = (e - s) / float(self.sample_rate)
            if dur >= min_call_duration:
                durations.append(dur)

        if len(durations) == 0:
            if mode == 'segments':
                return []
            return 0.0

        if mode == 'segments':
            return durations
        if mode == 'total':
            return float(np.sum(durations))
        return float(np.max(durations))
        

# Standalone functions for feature extraction
def compute_rms(samples: np.ndarray) -> float:
    return np.sqrt(np.mean(samples**2))

def compute_std(samples: np.ndarray) -> float:
    return np.std(samples)

def compute_crest_factor(samples: np.ndarray) -> float:
    rms = compute_rms(samples)
    peak = np.max(np.abs(samples))
    return peak / rms if rms > 0 else 0

def compute_avg_amplitude(samples: np.ndarray) -> float:
    return np.mean(np.abs(samples))

def compute_min_amplitude(samples: np.ndarray) -> float:
    return np.min(np.abs(samples))

def compute_max_amplitude(samples: np.ndarray) -> float:
    return np.max(np.abs(samples))


    
   
