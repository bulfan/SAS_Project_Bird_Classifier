# Restore process_and_save for batch_preprocess.py import
from pathlib import Path
import numpy as np
def process_and_save(filepath: str, out_dir: str, target_sr: int = 22050, dtype=np.float32, out_format: str = 'npy') -> str:
    """
    Load `filepath`, convert to mono, resample+denoise and save result as a .npy file
    under `out_dir/<class_name>/<stem>.npy`.

    Returns the saved file path as string.
    """
    try:
        from pydub import AudioSegment
    except Exception:
        raise RuntimeError("pydub is required for process_and_save")
    import numpy as _np

    audio = AudioSegment.from_file(filepath)
    samples = _np.array(audio.get_array_of_samples(), dtype=_np.float32)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)

    # Resample and denoise (reuse your previous logic or add your own)
    y_proc = samples
    sr_out = audio.frame_rate
    if sr_out != target_sr:
        try:
            import librosa
            y_proc = librosa.resample(y_proc, orig_sr=sr_out, target_sr=target_sr)
            sr_out = target_sr
        except Exception:
            pass

    out_path = Path(out_dir) / Path(filepath).parent.name
    out_path.mkdir(parents=True, exist_ok=True)
    save_fp = out_path / (Path(filepath).stem + ".npy")
    _np.save(save_fp, _np.asarray(y_proc, dtype=dtype))
    return str(save_fp)
"""
Audio preprocessing utilities for bird sound classification.
"""


def fir_filter_convolution(x: list, h: list) -> list:
    """
    Perform convolution of signal x[n] with filter h[n]
    and return the output y[n]
    where x[n] and h[n] overlap.
    Parameters:
        x (list): Input discrete signal.
        h (list): FIR filter kernel.

    Returns:
        list: Convolved output signal where x[n] and h[n] overlap.
    """

    # Length of input signal and filter
    L_x = len(x)
    L_h = len(h)

    # Length of the output signal
    L_y = L_x + L_h - 1

    # Initialize output signal y[n] with zeros
    y = [0] * L_y

    # Perform convolution: sliding window dot product
    for n in range(L_y):
        y[n] = sum(
            x[i] * h[n - i]
            for i in range(max(0, n - L_h + 1), min(L_x, n + 1))
        )

    start = 0
    end = L_y
    return y[start:end]


class AudioProcessor:
    """Class for preprocessing audio data."""
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the audio processor.

        Args:
            sample_rate: Target sample rate for audio files.
        """
        self.sample_rate = sample_rate
        # Cache for windows and FFT twiddle factors to speed up batch processing
        self._window_cache = {}
        self._twiddle_cache = {}
        self._bitrev_cache = {}

    def load_audio(self, file_path: str):
        """
        Load an audio file.

        Args:
            file_path: Path to the audio file.

        Returns:
            Audio data as a numpy array.
        """
        raise NotImplementedError("Audio loading not implemented")

    def extract_features(self, audio_data):
        """
        Extract features from audio data (e.g., mel spectrogram).

        Args:
            audio_data: Raw audio data.

        Returns:
            Extracted features.
        """
        raise NotImplementedError("Feature extraction not implemented")

    def preprocess(self, file_path: str):
        """
        Load and preprocess an audio file.

        Args:
            file_path: Path to the audio file.

        Returns:
            Preprocessed audio features.
        """
        audio_data = self.load_audio(file_path)
        features = self.extract_features(audio_data)
        return features

    def highpass_filter(self, y: np.ndarray, cutoff_freq: float, numtaps: int = 201) -> np.ndarray:
        """
        Apply a custom FIR high-pass filter using our own design and convolution.

        Args:
            y: Input signal (1-D numpy array).
            cutoff_freq: Cutoff frequency (Hz).
            numtaps: Number of filter taps (default: 201).

        Returns:
            Filtered signal as numpy array.
        """
        if cutoff_freq is None or cutoff_freq <= 0 or cutoff_freq >= self.sample_rate / 2:
            return y

        try:
            b = self.design_highpass_fir(cutoff_freq, numtaps)
        except Exception:
            return y

        try:
            y_f = fir_filter_convolution(list(y), list(b))
            return np.asarray(y_f, dtype=y.dtype)
        except Exception:
            return y
        
    def lowpass_filter(self, y: np.ndarray, cutoff_freq: float, numtaps: int = 1001) -> np.ndarray:
        """
        Apply a custom FIR low-pass filter using our own design and convolution.

        Args:
            y: Input signal (1-D numpy array).
            cutoff_freq: Cutoff frequency (Hz).
            numtaps: Number of filter taps (default: 1001).

        Returns:
            Filtered signal as numpy array.
        """
        if cutoff_freq is None or cutoff_freq <= 0 or cutoff_freq >= self.sample_rate / 2:
            return y

        try:
            b = self.design_lowpass_fir(cutoff_freq, numtaps)
        except Exception:
            return y

        try:
            y_f = fir_filter_convolution(list(y), list(b))
            return np.asarray(y_f, dtype=y.dtype)
        except Exception:
            return y
        
    def hard_threshold_gate(self, y: np.ndarray, threshold_db: float = -40) -> np.ndarray:
        """
        Apply a hard amplitude threshold to gate out signals below a certain dB level.
        This is a non-dynamic (static) method: samples below the threshold are set to zero.
        
        Args:
            y: Input audio signal (1D numpy array).
            threshold_db: Threshold in dB (default: -40). Signals below this are gated out.
        
        Returns:
            Thresholded audio signal as numpy array.
        """
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise ValueError("Input must be a 1D numpy array.")
        
        # Convert threshold to linear scale
        threshold = 10 ** (threshold_db / 20)
        
        # Apply hard threshold: set samples below threshold to zero
        y_thresholded = np.where(np.abs(y) < threshold, 0, y)
        
        return y_thresholded
    
    # =========================================================================
    # Custom FFT and Window Functions (Implemented from Scratch)
    # =========================================================================
    
    def _create_window(self, size: int, window_type: str = 'hann') -> np.ndarray:
        """
        Create a window function from scratch (with caching for speed).
        
        Args:
            size: Window size.
            window_type: Type of window ('hann', 'hamming', or 'none').
        
        Returns:
            Window array.
        """
        if size <= 0:
            raise ValueError("Window size must be positive")
        
        # Cache key
        cache_key = (size, window_type)
        if cache_key in self._window_cache:
            return self._window_cache[cache_key]
        
        if window_type == 'none' or window_type is None:
            window = np.ones(size, dtype=np.float32)
        elif size == 1:
            window = np.array([1.0], dtype=np.float32)
        else:
            window = np.zeros(size, dtype=np.float32)
            n_arr = np.arange(size, dtype=np.float32)
            
            if window_type == 'hann':
                window[:] = 0.5 * (1.0 - np.cos(2.0 * np.pi * n_arr / (size - 1)))
            elif window_type == 'hamming':
                window[:] = 0.54 - 0.46 * np.cos(2.0 * np.pi * n_arr / (size - 1))
            else:
                window[:] = 1.0
        
        self._window_cache[cache_key] = window
        return window
    
    def _bit_reverse(self, n: int) -> np.ndarray:
        """Fast bit reversal using bit manipulation (cached)."""
        if n in self._bitrev_cache:
            return self._bitrev_cache[n]
        
        levels = int(np.log2(n))
        indices = np.zeros(n, dtype=int)
        for i in range(n):
            # Fast bit reversal using bit manipulation
            rev = 0
            temp = i
            for _ in range(levels):
                rev = (rev << 1) | (temp & 1)
                temp >>= 1
            indices[i] = rev
        
        self._bitrev_cache[n] = indices
        return indices
    
    def _get_twiddle_factors(self, n: int) -> np.ndarray:
        """Precompute twiddle factors for FFT (cached)."""
        if n in self._twiddle_cache:
            return self._twiddle_cache[n]
        
        # Precompute all twiddle factors we might need
        k_max = n // 2
        twiddles = np.zeros(k_max, dtype=np.complex128)
        for k in range(k_max):
            twiddles[k] = np.exp(-2j * np.pi * k / n)
        
        self._twiddle_cache[n] = twiddles
        return twiddles
    
    def _fft(self, signal: np.ndarray) -> np.ndarray:
        """
        Compute the Fast Fourier Transform (FFT) of a 1D signal using a custom radix-2 Decimation in Time (DIT) Cooley-Tukey algorithm.

        This implementation is optimized for efficiency with caching of bit-reversal indices and twiddle factors to avoid recomputation
        in batch processing scenarios. It pads the input signal to the next power of 2 length if necessary for radix-2 compatibility.

        Algorithm Overview:
        - Bit-reversal permutation: Rearranges the input signal indices to align with the DIT structure.
        - Iterative butterfly operations: Combines smaller DFTs into larger ones using twiddle factors (complex exponentials).
        - Time complexity: O(N log N), where N is the padded signal length.
        - Space complexity: O(N) for the signal array and cached factors.

        Args:
            signal (np.ndarray): Input time-domain signal as a 1D real or complex numpy array. Length should ideally be a power of 2
                                 for optimal performance; otherwise, it will be zero-padded.

        Returns:
            np.ndarray: Complex-valued frequency-domain representation of the input signal, with length equal to the padded N.

        Notes:
            - Twiddle factors and bit-reversal indices are cached per length N to speed up repeated calls.
            - This is a from-scratch implementation without external FFT libraries for educational and portability purposes.
            - For inverse FFT, use _ifft() method.
        """
        n = len(signal)
        if n & (n - 1) != 0:
            # Pad to next power of 2
            next_pow2 = 1
            while next_pow2 < n:
                next_pow2 <<= 1
            padded = np.zeros(next_pow2, dtype=np.float64)
            padded[:n] = signal
            signal = padded
            n = next_pow2

        x = np.asarray(signal, dtype=np.complex128)
        
        # Fast bit reversal (cached)
        bit_rev_indices = self._bit_reverse(n)
        x = x[bit_rev_indices]

        # Precompute twiddle factors (cached)
        twiddles = self._get_twiddle_factors(n)

        # FFT butterfly operations
        size = 2
        while size <= n:
            half_size = size // 2
            table_step = n // size
            for i in range(0, n, size):
                for j in range(half_size):
                    k = j * table_step
                    twiddle = twiddles[k] if k < len(twiddles) else np.exp(-2j * np.pi * k / n)
                    temp = x[i + j + half_size] * twiddle
                    x[i + j + half_size] = x[i + j] - temp
                    x[i + j] = x[i + j] + temp
            size *= 2
        return x
    
    def _ifft(self, spectrum: np.ndarray, n: int = None) -> np.ndarray:
        """
        Inverse FFT using custom implementation.
        
        Args:
            spectrum: Complex frequency-domain signal.
            n: Desired output length (for inverse rfft compatibility).
        
        Returns:
            Time-domain signal.
        """
        # Take conjugate, apply FFT, then conjugate and scale
        result = np.conj(self._fft(np.conj(spectrum)))
        result = result / len(spectrum)
        
        if n is not None and n < len(result):
            result = result[:n]
        elif n is not None and n > len(result):
            # Pad with zeros
            padded = np.zeros(n, dtype=result.dtype)
            padded[:len(result)] = result
            result = padded
        
        return result.real
    
    def _stft(self, y: np.ndarray, n_fft: int = 2048, hop_length: int = 512, 
              window: str = 'hann') -> np.ndarray:
        """
        Optimized Short-Time Fourier Transform using custom FFT implementation.
        
        Args:
            y: Input audio signal (1D numpy array).
            n_fft: FFT window size.
            hop_length: Number of samples between successive frames.
            window: Window type ('hann', 'hamming', or 'none').
        
        Returns:
            Complex-valued spectrogram (n_freqs, n_frames).
        """
        n_samples = len(y)
        window_func = self._create_window(n_fft, window)  # Cached
        
        n_frames = 1 + (n_samples - n_fft) // hop_length
        if n_frames < 1:
            n_frames = 1
        
        n_freqs = n_fft // 2 + 1
        spectrogram = np.zeros((n_freqs, n_frames), dtype=np.complex128)
        
        # Pre-allocate frame buffer to avoid repeated allocation
        frame = np.zeros(n_fft, dtype=y.dtype)
        
        for frame_idx in range(n_frames):
            start = frame_idx * hop_length
            end = start + n_fft
            
            if end <= n_samples:
                # Fast path: no padding needed
                np.multiply(y[start:end], window_func, out=frame)
            else:
                # Zero-pad if needed
                frame.fill(0)
                available = n_samples - start
                if available > 0:
                    frame[:available] = y[start:start + available]
                np.multiply(frame, window_func, out=frame)
            
            # Compute FFT
            spectrum = self._fft(frame)
            
            # Take only positive frequencies (rfft equivalent)
            spectrogram[:, frame_idx] = spectrum[:n_freqs]
        
        return spectrogram
    
    def _istft(self, stft_matrix: np.ndarray, hop_length: int = 512, 
               window: str = 'hann', length: int = None) -> np.ndarray:
        """
        Inverse Short-Time Fourier Transform using custom IFFT implementation.
        
        Args:
            stft_matrix: Complex-valued spectrogram (n_freqs, n_frames).
            hop_length: Number of samples between successive frames.
            window: Window type (must match STFT window).
            length: Desired output length. If None, inferred from hop_length.
        
        Returns:
            Reconstructed time-domain signal.
        """
        n_freqs, n_frames = stft_matrix.shape
        n_fft = 2 * (n_freqs - 1)
        
        window_func = self._create_window(n_fft, window)
        
        # Estimate output length if not provided
        if length is None:
            length = (n_frames - 1) * hop_length + n_fft
        
        # Initialize output
        y = np.zeros(length, dtype=np.float64)
        
        # Reconstruct each frame
        for frame_idx in range(n_frames):
            start = frame_idx * hop_length
            
            # Reconstruct full spectrum from positive frequencies
            spectrum = np.zeros(n_fft, dtype=np.complex128)
            spectrum[:n_freqs] = stft_matrix[:, frame_idx]
            # Mirror negative frequencies (for real-valued output)
            # Skip DC (index 0) and Nyquist (index n_freqs-1 if n_fft is even)
            if n_fft % 2 == 0:
                # Even n_fft: Nyquist is at n_freqs-1, mirror from n_freqs-2 down to 1
                spectrum[n_freqs:] = np.conj(stft_matrix[n_freqs-2:0:-1, frame_idx])
            else:
                # Odd n_fft: no Nyquist, mirror from n_freqs-1 down to 1
                spectrum[n_freqs:] = np.conj(stft_matrix[n_freqs-1:0:-1, frame_idx])
            
            # Inverse FFT
            frame = self._ifft(spectrum, n=n_fft)
            
            # Apply window and overlap-add
            frame_windowed = frame * window_func
            end = start + n_fft
            if end <= length:
                y[start:end] += frame_windowed
            else:
                available = length - start
                y[start:] += frame_windowed[:available]
        
        # Normalize by window overlap
        # Compute normalization factor based on window overlap
        # This accounts for the fact that samples are added multiple times due to overlap
        window_squared_sum = np.sum(window_func ** 2)
        if window_squared_sum > 0:
            # Normalize by the expected overlap-add gain
            # For perfect reconstruction, we need to divide by the sum of squared windows
            # at each sample position, but a simpler approximation is to normalize by hop_length
            # when using standard windows with 50% overlap
            normalization = hop_length / window_squared_sum
            y = y * normalization
        
        return y.astype(np.float32)
    
    def _amplitude_to_db(self, magnitude: np.ndarray, ref: float = None) -> np.ndarray:
        """
        Convert magnitude to decibels.
        
        Args:
            magnitude: Magnitude values.
            ref: Reference value for dB calculation. If None, uses max value.
        
        Returns:
            Magnitude in dB.
        """
        if ref is None:
            ref = np.max(magnitude)
        
        # Avoid log of zero
        eps = 1e-10
        magnitude = np.maximum(magnitude, eps)
        
        if ref > 0:
            db = 20.0 * np.log10(magnitude / ref)
        else:
            db = 20.0 * np.log10(magnitude + eps)
        
        return db
        
    def spectral_gate(self, y: np.ndarray, threshold_db: float = -40, percentile: float = 10) -> np.ndarray:
        """
        Apply a spectral gate to the audio signal in the frequency domain using librosa for better noise reduction.
        
        Args:
            y: Input audio signal (1D numpy array).
            threshold_db: Magnitude threshold in dB (default: -40).
            percentile: Not used; kept for compatibility.
        
        Returns:
            Gated audio signal as numpy array (time-domain).
        """
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise ValueError("Input must be a 1D numpy array.")
        
        if len(y) == 0:
            return y
        
        try:
            import librosa
        except ImportError:
            # Fallback to simple FFT-based gating if librosa not available
            Y = np.fft.rfft(y)
            mag = np.abs(Y)
            threshold = np.percentile(mag, percentile)
            mask = mag >= threshold
            Y_gated = Y * mask
            return np.fft.irfft(Y_gated, n=len(y))
        
        # Use original librosa defaults for better noise reduction
        # n_fft=2048, hop_length=512 (n_fft // 4)
        D = librosa.stft(y)
        
        # Convert to dB
        D_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        
        # Create mask
        mask = D_db > threshold_db
        
        # Apply mask
        D_clean = D * mask
        
        # Reconstruct time-domain signal using librosa ISTFT
        y_clean = librosa.istft(D_clean, length=len(y))
        
        return y_clean
    
    def noise_gate(self, y: np.ndarray, threshold_db: float = -40, attack_ms: float = 10, release_ms: float = 100, hold_ms: float = 200, max_on_ms: float = 2000, ratio: float = 10) -> np.ndarray:
        """
        Apply a noise gate to the audio signal to reduce low-level noise, with hold time for intra-call silences and max on-time to avoid gating long noise.
        
        Args:
            y: Input audio signal (1D numpy array).
            threshold_db: Threshold in dB below which signal is attenuated (default: -40).
            attack_ms: Attack time in milliseconds (default: 10).
            release_ms: Release time in milliseconds (default: 100).
            hold_ms: Hold time in milliseconds to keep gate open during short silences (default: 200).
            max_on_ms: Maximum time in milliseconds to keep gate open for continuous activity (default: 2000).
            ratio: Compression ratio for signals below threshold (default: 10 for soft gating).
        
        Returns:
            Gated audio signal as numpy array.
        """
        if not isinstance(y, np.ndarray) or y.ndim != 1:
            raise ValueError("Input must be a 1D numpy array.")
        
        # Convert threshold to linear scale
        threshold = 10 ** (threshold_db / 20)
        
        # Calculate coefficients
        attack_coeff = 1 - np.exp(-1 / (attack_ms * self.sample_rate / 1000))
        release_coeff = 1 - np.exp(-1 / (release_ms * self.sample_rate / 1000))
        hold_samples = int(hold_ms * self.sample_rate / 1000)
        max_on_samples = int(max_on_ms * self.sample_rate / 1000)
        
        # Initialize
        envelope = np.zeros_like(y, dtype=np.float32)
        gain = np.zeros_like(y, dtype=np.float32)
        gain[0] = 0  # Start closed
        hold_counter = 0
        on_counter = 0
        
        for i in range(1, len(y)):
            # Update envelope
            abs_y = abs(y[i])
            if abs_y > envelope[i-1]:
                envelope[i] = attack_coeff * (envelope[i-1] - abs_y) + abs_y
            else:
                envelope[i] = release_coeff * (envelope[i-1] - abs_y) + abs_y
            
            # Determine target gain
            if envelope[i] > threshold:
                if on_counter < max_on_samples:
                    gain_target = 1.0
                    on_counter += 1
                else:
                    gain_target = 0.0  # Long activity assumed noise
                hold_counter = 0  # Reset hold
                coeff = attack_coeff
            else:
                on_counter = 0  # Reset on counter
                if hold_counter < hold_samples:
                    gain_target = 1.0  # Hold open for short silences
                    hold_counter += 1
                else:
                    gain_target = 0.0
                coeff = release_coeff
            
            # Smooth gain
            gain[i] = gain[i-1] + (gain_target - gain[i-1]) * coeff
            
            # Clip gain to prevent overflow
            gain[i] = np.clip(gain[i], 0, 1)
        
        # Apply gain to signal
        return y * gain
    
    