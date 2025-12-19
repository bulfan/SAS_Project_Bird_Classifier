"""
Audio Processor for Bird Sound Classification - Noise Reduction.

Provides FIR filter bank for reducing environmental noise in bird recordings.
Uses numpy/scipy/librosa for efficient implementation (libraries allowed in preprocessing).

Filter Bank Design (based on spectral analysis):
- Highpass filter: Removes low-frequency rumble (wind, traffic) below ~500 Hz
- Lowpass filter: Removes high-frequency hiss above bird vocalization range (~8 kHz)
- Spectral gate: Removes broadband noise in frequency domain
- Noise gate: Time-domain gating for silence periods

Note: The analysis module uses from-scratch implementations, but preprocessing
is allowed to use libraries for efficiency since it runs once per file.
"""

from pathlib import Path
from typing import Optional, Tuple, Any
import numpy as np

# Optional audio dependencies
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# Optional signal processing dependencies
try:
    from scipy.signal import firwin, filtfilt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False


# =============================================================================
# AudioProcessor Class
# =============================================================================

class AudioProcessor:
    """
    Audio preprocessing class with FIR filter bank for noise reduction.
    
    Uses library implementations (numpy, scipy, librosa) for efficiency.
    Can be initialized with a config object for easy parameter management.
    
    Example usage:
        # With config (from Hydra)
        ap = AudioProcessor(cfg=cfg.preprocessing)
        
        # Without config (use defaults)
        ap = AudioProcessor()
        
        # Manual parameters
        ap = AudioProcessor(sample_rate=44100, highpass_cutoff=300)
    """
    
    def __init__(self, 
                 cfg: Optional[Any] = None,
                 sample_rate: int = 22050,
                 highpass_cutoff: float = 500,
                 lowpass_cutoff: float = 7000,
                 gate_threshold_db: float = -40):
        """
        Initialize the audio processor.

        Args:
            cfg: Optional config object (e.g., from Hydra). If provided,
                 overrides individual parameters.
            sample_rate: Target sample rate for audio files.
            highpass_cutoff: Cutoff frequency for highpass filter (Hz).
            lowpass_cutoff: Cutoff frequency for lowpass filter (Hz).
            gate_threshold_db: Threshold in dB for noise/spectral gates.
        """
        # Load from config if provided, otherwise use passed parameters
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
        """
        Load an audio file using librosa or pydub.

        Args:
            file_path: Path to the audio file.
            normalize: If True, normalize to [-1, 1] range.
            max_duration: Maximum duration in seconds (None = full file).

        Returns:
            Tuple of (samples, sample_rate).
        """
        if LIBROSA_AVAILABLE:
            # Use librosa for loading (handles resampling automatically)
            y, sr = librosa.load(file_path, sr=self.sample_rate,
                                duration=max_duration, mono=True)
            return y, sr
        elif PYDUB_AVAILABLE:
            # Fallback to pydub
            seg = AudioSegment.from_file(file_path)
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            
            # Convert to mono
            if seg.channels > 1:
                samples = samples.reshape((-1, seg.channels)).mean(axis=1)
            
            # Normalize
            if normalize:
                max_val = float(2 ** (seg.sample_width * 8 - 1))
                samples = samples / max_val
            
            # Limit duration
            if max_duration is not None:
                max_samples = int(seg.frame_rate * max_duration)
                samples = samples[:max_samples]
            
            return samples, seg.frame_rate
        else:
            raise ImportError("Neither librosa nor pydub is available for audio loading")
    
    # =========================================================================
    # Full Processing Pipeline
    # =========================================================================
    
    def process(self, y: np.ndarray, 
                skip_filters: bool = False,
                skip_spectral_gate: bool = False) -> np.ndarray:
        """
        Apply noise reduction pipeline to audio signal.
        
        Pipeline order:
        1. Remove DC offset
        2. Spectral gate (attenuate low-energy frequency bins = noise)
        3. FIR bandpass filters (remove frequencies outside bird range)
        
        
        Note: Time-domain noise gate removed from default pipeline.
        It can still be called separately via noise_gate() if needed.
        
        Args:
            y: Input audio signal (1D numpy array).
            skip_filters: Skip highpass/lowpass FIR filters.
            skip_spectral_gate: Skip frequency-domain spectral gate.
        
        Returns:
            Processed audio signal.
        """
        # Remove DC offset
        y = self.remove_dc(y)
        
        # Apply spectral gate to reduce broadband noise
        if not skip_spectral_gate:
            y = self.spectral_gate(y, threshold_db=self.gate_threshold_db)

        # Apply FIR bandpass filters FIRST
        # This removes frequencies outside bird vocalization range
        if not skip_filters:
            y = self.highpass_filter(y, self.highpass_cutoff)
            y = self.lowpass_filter(y, self.lowpass_cutoff)
        
        
        
        return y

    # =========================================================================
    # FIR Filter Bank (Using scipy.signal)
    # =========================================================================

    def highpass_filter(self, y: np.ndarray, cutoff_freq: float, 
                        numtaps: int = 201) -> np.ndarray:
        """
        Apply FIR high-pass filter to remove low-frequency noise.
        
        Purpose: Remove rumble, wind noise, and low-frequency environmental sounds
        that are below the typical bird vocalization range.
        
        How it works:
        1. Design FIR filter coefficients using scipy.signal.firwin
        2. Apply filter using scipy.signal.filtfilt (zero-phase, forward-backward)

        Args:
            y: Input signal (1-D numpy array).
            cutoff_freq: Cutoff frequency in Hz (e.g., 500 Hz for birds).
            numtaps: Number of filter taps (higher = sharper cutoff).

        Returns:
            Filtered signal. Returns original if scipy unavailable.
        """
        if not SCIPY_AVAILABLE:
            print("Warning: scipy not available, skipping highpass filter")
            return y
        
        if len(y) == 0:
            return y

        nyq = 0.5 * self.sample_rate
        normal_cutoff = cutoff_freq / nyq
        
        # Validate cutoff frequency
        if normal_cutoff <= 0 or normal_cutoff >= 1:
            return y

        # Ensure numtaps is odd for Type I FIR filter
        if numtaps % 2 == 0:
            numtaps += 1

        # Design FIR highpass filter using Hamming window
        # pass_zero=False means highpass (blocks DC/low frequencies)
        b = firwin(numtaps, normal_cutoff, pass_zero=False, window='hamming')
        
        # Apply zero-phase filtering (forward + backward pass)
        # This provides better attenuation and no phase distortion
        # padlen handles edge effects
        padlen = min(3 * numtaps, len(y) - 1)
        y_filtered = filtfilt(b, [1.0], y, padlen=padlen)
        
        return y_filtered.astype(y.dtype)
        
    def lowpass_filter(self, y: np.ndarray, cutoff_freq: float, 
                       numtaps: int = 201) -> np.ndarray:
        """
        Apply FIR low-pass filter to remove high-frequency noise.
        
        Purpose: Remove hiss, high-frequency artifacts, and sounds above
        the typical bird vocalization range (most birds: 1-8 kHz).
        
        How it works:
        1. Design FIR filter coefficients using scipy.signal.firwin
        2. Apply filter using scipy.signal.filtfilt (zero-phase, forward-backward)

        Args:
            y: Input signal (1-D numpy array).
            cutoff_freq: Cutoff frequency in Hz (e.g., 8000 Hz for birds).
            numtaps: Number of filter taps.

        Returns:
            Filtered signal. Returns original if scipy unavailable.
        """
        if not SCIPY_AVAILABLE:
            print("Warning: scipy not available, skipping lowpass filter")
            return y
        
        if len(y) == 0:
            return y

        nyq = 0.5 * self.sample_rate
        normal_cutoff = cutoff_freq / nyq
        
        # Validate cutoff frequency
        if normal_cutoff <= 0 or normal_cutoff >= 1:
            return y

        # Ensure numtaps is odd for Type I FIR filter
        if numtaps % 2 == 0:
            numtaps += 1

        # Design FIR lowpass filter using Hamming window
        # pass_zero=True means lowpass (passes DC/low frequencies)
        b = firwin(numtaps, normal_cutoff, pass_zero=True, window='hamming')
        
        # Apply zero-phase filtering (forward + backward pass)
        # This provides better attenuation and no phase distortion
        padlen = min(3 * numtaps, len(y) - 1)
        y_filtered = filtfilt(b, [1.0], y, padlen=padlen)
        
        return y_filtered.astype(y.dtype)
        
    def hard_threshold_gate(self, y: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """
        Apply hard amplitude threshold (simple gating).
        
        Purpose: Zero out samples below a threshold. Simple but can cause
        audible artifacts (clicks) at gate transitions.
        
        Args:
            y: Input audio signal (1D numpy array).
            threshold_db: Threshold in dB (default: -40).
        
        Returns:
            Gated audio signal.
        """
        if len(y) == 0:
            return y
        
        # Convert threshold from dB to linear scale
        # 0 dB = 1.0, -20 dB = 0.1, -40 dB = 0.01, -60 dB = 0.001
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Apply hard threshold: zero out samples below threshold
        y_gated = np.where(np.abs(y) < threshold_linear, 0.0, y)
        
        return y_gated.astype(y.dtype)

    # =========================================================================
    # Spectral Noise Reduction (Using librosa/numpy)
    # =========================================================================
        
    def spectral_gate(self, y: np.ndarray, threshold_db: float = -40,
                      n_fft: int = 2048, hop_length: int = 512,
                      percentile: float = 10) -> np.ndarray:
        """
        Apply spectral gating to reduce broadband noise.
        
        Purpose: Attenuate frequency bins below a threshold in each time frame,
        effectively removing constant background noise while preserving
        bird vocalizations.
        
        How it works:
        1. Compute STFT (Short-Time Fourier Transform)
        2. Convert magnitude to dB scale
        3. Create mask: keep bins above threshold
        4. Apply mask and reconstruct signal with ISTFT
        
        Note: Bandpass filtering should be done separately with FIR filters
        BEFORE calling spectral_gate to avoid ISTFT reintroducing frequencies.
        
        Args:
            y: Input audio signal (1D numpy array).
            threshold_db: Magnitude threshold in dB relative to max (default: -40).
            n_fft: FFT window size.
            hop_length: Hop length for STFT.
            percentile: Fallback percentile threshold if librosa unavailable.
        
        Returns:
            Noise-reduced audio signal.
        """
        if len(y) == 0:
            return y
        
        if LIBROSA_AVAILABLE:
            # Use librosa for robust STFT/ISTFT
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            
            # Convert magnitude to dB (relative to max)
            D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            # Create mask: 1 where signal is above threshold, 0 elsewhere
            mask = (D_db > threshold_db).astype(np.float32)
            
            # Apply mask to complex STFT
            D_clean = D * mask
            
            # Reconstruct time-domain signal
            y_clean = librosa.istft(D_clean, hop_length=hop_length, length=len(y))
            
            return y_clean.astype(y.dtype)
        else:
            # Fallback to numpy FFT (simpler, less robust)
            Y = np.fft.rfft(y)
            magnitude = np.abs(Y)
            
            # Use percentile-based threshold
            threshold = np.percentile(magnitude, percentile)
            mask = (magnitude >= threshold).astype(np.float32)
            
            # Apply mask and reconstruct
            Y_clean = Y * mask
            y_clean = np.fft.irfft(Y_clean, n=len(y))
            
            return y_clean.astype(y.dtype)

    # =========================================================================
    # Time-Domain Noise Gate    !!!!NO LONGER USED!!!!
    # =========================================================================
    
    def noise_gate(self, y: np.ndarray, threshold_db: float = -40,
                   attack_ms: float = 10, release_ms: float = 100,
                   hold_ms: float = 200, max_on_ms: float = 2000,
                   ratio: float = 10) -> np.ndarray:
        """
        Apply smooth noise gate with attack/release envelope.
        
        Purpose: Attenuate signal during quiet periods while avoiding
        audible artifacts. Uses smooth gain transitions.
        
        How it works:
        1. Track signal envelope (peak follower)
        2. When envelope > threshold: open gate (with attack time)
        3. When envelope < threshold: close gate (with release time)
        4. Hold time keeps gate open during short pauses (between bird notes)
        5. Max on time prevents long continuous noise from passing through
        
        Args:
            y: Input audio signal (1D numpy array).
            threshold_db: Threshold in dB below which signal is attenuated.
            attack_ms: Time in ms for gate to fully open.
            release_ms: Time in ms for gate to fully close.
            hold_ms: Time in ms to keep gate open during short silences.
            max_on_ms: Maximum time in ms to keep gate open continuously.
            ratio: Compression ratio (unused, kept for compatibility).
        
        Returns:
            Gated audio signal with smooth transitions.
        """
        if len(y) == 0:
            return y
        
        # Convert threshold to linear scale
        threshold_linear = 10 ** (threshold_db / 20)
        
        # Calculate time constants (exponential smoothing coefficients)
        attack_coeff = 1.0 - np.exp(-1.0 / (attack_ms * self.sample_rate / 1000))
        release_coeff = 1.0 - np.exp(-1.0 / (release_ms * self.sample_rate / 1000))
        hold_samples = int(hold_ms * self.sample_rate / 1000)
        max_on_samples = int(max_on_ms * self.sample_rate / 1000)
        
        # Initialize arrays
        n = len(y)
        envelope = np.zeros(n, dtype=np.float32)
        gain = np.zeros(n, dtype=np.float32)
        
        # Track envelope and compute gain
        hold_counter = 0
        on_counter = 0
        
        for i in range(1, n):
            # Update envelope (peak follower with attack/release)
            abs_sample = abs(y[i])
            if abs_sample > envelope[i-1]:
                # Attack: envelope rises quickly
                envelope[i] = envelope[i-1] + attack_coeff * (abs_sample - envelope[i-1])
            else:
                # Release: envelope falls slowly
                envelope[i] = envelope[i-1] + release_coeff * (abs_sample - envelope[i-1])
            
            # Determine target gain based on envelope vs threshold
            if envelope[i] > threshold_linear:
                # Signal above threshold - open gate
                if on_counter < max_on_samples:
                    target_gain = 1.0
                    on_counter += 1
                else:
                    # Been on too long - probably noise, close gate
                    target_gain = 0.0
                hold_counter = 0
                coeff = attack_coeff
            else:
                # Signal below threshold
                on_counter = 0
                if hold_counter < hold_samples:
                    # In hold period - keep gate open
                    target_gain = 1.0
                    hold_counter += 1
                else:
                    # Close gate
                    target_gain = 0.0
                coeff = release_coeff
            
            # Smooth gain transition
            gain[i] = gain[i-1] + coeff * (target_gain - gain[i-1])
            gain[i] = np.clip(gain[i], 0.0, 1.0)
        
        # Apply gain
        return (y * gain).astype(y.dtype)

    # =========================================================================
    # Utility Methods
    # =========================================================================
    
    def remove_dc(self, y: np.ndarray) -> np.ndarray:
        """
        Remove DC offset from signal.

        Args:
            y: Input audio signal.

        Returns:
            Signal with DC offset removed.
        """
        return (y - np.mean(y)).astype(y.dtype)
    
    def normalize(self, y: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """
        Normalize audio to target peak level.

        Args:
            y: Input audio signal.
            target_db: Target peak level in dB (default: -3 dB).

        Returns:
            Normalized audio signal.
        """
        if len(y) == 0:
            return y
        
        peak = np.max(np.abs(y))
        if peak < 1e-10:
            return y
        
        target_linear = 10 ** (target_db / 20)
        return (y * target_linear / peak).astype(y.dtype)