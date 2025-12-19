"""
Audio Processor for Bird Sound Classification - Noise Reduction.
Filter Bank Design (based on spectral analysis):
- Highpass filter: Removes low-frequency rumble (wind, traffic)
- Lowpass filter: Removes high-frequency hiss above bird vocalization range 
- Spectral gate: Removes broadband noise in frequency domain
- Noise gate: Time-domain gating for silence periods (no longer used in default pipeline)
"""

from pathlib import Path
from typing import Optional, Tuple, Any
import numpy as np
import librosa
from scipy.signal import firwin, filtfilt


# =============================================================================
# AudioProcessor Class
# =============================================================================

class AudioProcessor:
    """Audio preprocessing class with FIR filter bank for noise reduction."""
    
    def __init__(self, 
                 cfg: Optional[Any] = None,
                 sample_rate: int = 22050,
                 highpass_cutoff: float = 500,
                 lowpass_cutoff: float = 7000,
                 gate_threshold_db: float = -40):
        """Initialize the audio processor."""
        if cfg is not None:
            self.sample_rate = getattr(cfg, 'sample_rate', sample_rate)
            self.highpass_cutoff = getattr(cfg, 'highpass_cutoff', highpass_cutoff)
            self.lowpass_cutoff = getattr(cfg, 'lowpass_cutoff', lowpass_cutoff)
            self.gate_threshold_db = getattr(cfg, 'gate_threshold_db', gate_threshold_db)
        else:
            self.sample_rate = sample_rate
            self.highpass_cutoff = highpass_cutoff
            self.lowpass_cutoff = lowpass_cutoff
            self.gate_threshold_db = gate_threshold_db

    # =========================================================================
    # Audio Loading
    # =========================================================================
    
    def load_audio(self, file_path: str, normalize: bool = True,
                   max_duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Load an audio file using librosa."""
        y, sr = librosa.load(file_path, sr=self.sample_rate, duration=max_duration, mono=True)
        return y, sr
    
    # =========================================================================
    # Full Processing Pipeline
    # =========================================================================
    
    def process(self, y: np.ndarray, skip_filters: bool = False, skip_spectral_gate: bool = False) -> np.ndarray:
        """
        Apply noise reduction pipeline to audio signal.
        Pipeline order:
        1. Remove DC offset
        2. Spectral gate
        3. FIR bandpass filters
         """
        # Remove DC offset
        y = self.remove_dc(y)
        
        # Apply spectral gate to reduce broadband noise
        if not skip_spectral_gate:
            y = self.spectral_gate(y, threshold_db=self.gate_threshold_db)

        # Apply FIR bandpass filters
        if not skip_filters:
            y = self.highpass_filter(y, self.highpass_cutoff)
            y = self.lowpass_filter(y, self.lowpass_cutoff)
        return y

    # =========================================================================
    # FIR Filter Bank
    # =========================================================================

    def highpass_filter(self, y: np.ndarray, cutoff_freq: float, numtaps: int = 201) -> np.ndarray:
        """Apply FIR high-pass filter to remove low-frequency noise."""
        if len(y) == 0:
            return y

        nyq = 0.5 * self.sample_rate
        normal_cutoff = cutoff_freq / nyq
        
        if normal_cutoff <= 0 or normal_cutoff >= 1:
            return y

        if numtaps % 2 == 0:
            numtaps += 1

        # Design FIR highpass filter using Hamming window
        b = firwin(numtaps, normal_cutoff, pass_zero=False, window='hamming')
        
        # Apply zero-phase filtering 
        padlen = min(3 * numtaps, len(y) - 1)
        y_filtered = filtfilt(b, [1.0], y, padlen=padlen)
        
        return y_filtered.astype(y.dtype)
        
    def lowpass_filter(self, y: np.ndarray, cutoff_freq: float, 
                       numtaps: int = 201) -> np.ndarray:
        """Apply FIR low-pass filter to remove high-frequency noise."""
        if len(y) == 0:
            return y

        nyq = 0.5 * self.sample_rate
        normal_cutoff = cutoff_freq / nyq
        
        if normal_cutoff <= 0 or normal_cutoff >= 1:
            return y

        if numtaps % 2 == 0:
            numtaps += 1

        # Design FIR lowpass filter using Hamming window
        b = firwin(numtaps, normal_cutoff, pass_zero=True, window='hamming')
        
        # Apply zero-phase filtering
        padlen = min(3 * numtaps, len(y) - 1)
        y_filtered = filtfilt(b, [1.0], y, padlen=padlen)
        
        return y_filtered.astype(y.dtype)

    def hard_threshold_gate(self, y: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """Apply hard amplitude threshold."""
        if len(y) == 0:
            return y
        threshold_linear = 10 ** (threshold_db / 20)
        y_gated = np.where(np.abs(y) < threshold_linear, 0.0, y)
        return y_gated.astype(y.dtype)

    # =========================================================================
    # Spectral Noise Reduction (Using librosa)
    # =========================================================================
        
    def spectral_gate(self, y: np.ndarray, threshold_db: float = -40,
                      n_fft: int = 2048, hop_length: int = 512) -> np.ndarray:
        """Apply spectral gating to reduce broadband noise."""
        if len(y) == 0:
            return y
        
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        
        # Convert magnitude to dB
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Create mask: 1 where signal is above threshold, 0 elsewhere
        mask = (D_db > threshold_db).astype(np.float32)
        D_clean = D * mask
        
        # Reconstruct time-domain signal
        y_clean = librosa.istft(D_clean, hop_length=hop_length, length=len(y))
        
        return y_clean.astype(y.dtype)

    # =========================================================================
    # Time-Domain Noise Gate    !!!!NO LONGER USED!!!!
    # =========================================================================
    
    def noise_gate(self, y: np.ndarray, threshold_db: float = -40,
                   attack_ms: float = 10, release_ms: float = 100,
                   hold_ms: float = 200, max_on_ms: float = 2000,
                   ratio: float = 10) -> np.ndarray:
        """Apply smooth noise gate with attack/release envelope."""
        if len(y) == 0:
            return y
        
        threshold_linear = 10 ** (threshold_db / 20)
        attack_coeff = 1.0 - np.exp(-1.0 / (attack_ms * self.sample_rate / 1000))
        release_coeff = 1.0 - np.exp(-1.0 / (release_ms * self.sample_rate / 1000))
        hold_samples = int(hold_ms * self.sample_rate / 1000)
        max_on_samples = int(max_on_ms * self.sample_rate / 1000)
        n = len(y)
        envelope = np.zeros(n, dtype=np.float32)
        gain = np.zeros(n, dtype=np.float32)
        hold_counter = 0
        on_counter = 0
        
        for i in range(1, n):
            abs_sample = abs(y[i])
            if abs_sample > envelope[i-1]:
                envelope[i] = envelope[i-1] + attack_coeff * (abs_sample - envelope[i-1])
            else:
                envelope[i] = envelope[i-1] + release_coeff * (abs_sample - envelope[i-1])
            if envelope[i] > threshold_linear:
                if on_counter < max_on_samples:
                    target_gain = 1.0
                    on_counter += 1
                else:
                    target_gain = 0.0
                hold_counter = 0
                coeff = attack_coeff
            else:
                on_counter = 0
                if hold_counter < hold_samples:
                    target_gain = 1.0
                    hold_counter += 1
                else:
                    target_gain = 0.0
                coeff = release_coeff
            
            gain[i] = gain[i-1] + coeff * (target_gain - gain[i-1])
            gain[i] = np.clip(gain[i], 0.0, 1.0)
        
        return (y * gain).astype(y.dtype)

    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def remove_dc(self, y: np.ndarray) -> np.ndarray:
        """Remove DC offset from signal."""
        return (y - np.mean(y)).astype(y.dtype)
    
    def normalize(self, y: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """Normalize audio to target peak level."""
        if len(y) == 0:
            return y
        
        peak = np.max(np.abs(y))
        if peak < 1e-10:
            return y
        
        target_linear = 10 ** (target_db / 20)
        return (y * target_linear / peak).astype(y.dtype)