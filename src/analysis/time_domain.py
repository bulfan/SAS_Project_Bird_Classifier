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

        






    
   