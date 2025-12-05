"""
Time-Domain Analysis for audio signals.

This module provides a class to compute various time-domain features:
- Mean (μ): DC offset / average amplitude
- Standard Deviation (σ): Variation in the signal
- RMS: Root Mean Square (effective amplitude)
- RMS in dB: RMS converted to decibels (dBFS)

Formulas:
    Mean: μ = (1/N) * Σx[i]
    Std Dev: σ = sqrt( (1/(N-1)) * Σ(x[i] - μ)² )
    RMS: sqrt( (1/N) * Σx[i]² )
"""

import numpy as np
from typing import Dict, Optional
import librosa


class TimeDomainAnalysis:
    """
    A class to perform time-domain analysis on audio signals.
    
    Attributes:
        signal: The audio signal as a numpy array
        sample_rate: The sample rate of the audio
        results: Dictionary containing computed metrics
    """
    
    def __init__(self, signal: Optional[np.ndarray] = None, sample_rate: int = 22050):
        """
        Initialize the TimeDomainAnalysis.
        
        Args:
            signal: 1D numpy array representing the audio signal (optional)
            sample_rate: Sample rate of the audio (default: 22050)
        """
        self.signal = signal
        self.sample_rate = sample_rate
        self.results: Dict = {}
    
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
        """
        Run all time-domain analyses and return results.
        
        Args:
            print_results: If True, print the results to console
            
        Returns:
            Dictionary containing all computed metrics:
                - mean: The mean value μ
                - std: The standard deviation σ
                - rms: The RMS value
                - rms_db: RMS in dBFS
                - N: Number of samples
                - duration: Duration in seconds
        """
        if self.signal is None:
            raise ValueError("No signal loaded. Call load_audio() or pass signal to constructor.")
        
        self.results = {
            'mean': self.compute_mean(),
            'std': self.compute_standard_deviation(),
            'rms': self.compute_rms(),
            'rms_db': self.compute_rms_db(),
            'N': len(self.signal),
            'duration': len(self.signal) / self.sample_rate
        }
        
        if print_results:
            print(f"   - Mean (μ): {self.results['mean']:.6f}")
            print(f"   - Standard Deviation (σ): {self.results['std']:.6f}")
            print(f"   - RMS: {self.results['rms']:.6f}")
            print(f"   - RMS (dB): {self.results['rms_db']:.2f} dBFS")
        
        return self.results
