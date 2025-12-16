"""
Audio preprocessing utilities for bird sound classification.
"""
import numpy as np


class AudioProcessor:
    """Class for preprocessing audio data."""

    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the audio processor.

        Args:
            sample_rate: Target sample rate for audio files.
        """
        self.sample_rate = sample_rate

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

    def bandpass_filter(self, y: np.ndarray, low_freq: float, high_freq: float) -> np.ndarray:
        """
        Apply a Butterworth bandpass filter to a 1-D signal.

        Args:
            y: Input signal (1-D numpy array).
            low_freq: Low cutoff frequency (Hz).
            high_freq: High cutoff frequency (Hz).

        Returns:
            Filtered signal as numpy array. If scipy is not available, returns original signal.
        """
        try:
            from scipy.signal import butter, filtfilt
        except Exception:
            # scipy not available; return original signal
            return y

        sr = getattr(self, 'sample_rate', None)
        if sr is None or low_freq is None or high_freq is None:
            return y

        nyq = 0.5 * sr
        low = max(0.0001, float(low_freq) / nyq)
        high = min(0.9999, float(high_freq) / nyq)
        if low >= high:
            return y
        b, a = butter(4, [low, high], btype='band')
        try:
            y_f = filtfilt(b, a, y)
            return y_f
        except Exception:
            return y


def bandpass_filter(y: np.ndarray, sr: int, low_freq: float, high_freq: float) -> np.ndarray:
    """
    Module-level bandpass filter helper (uses scipy if available).

    Args:
        y: Input signal
        sr: Sample rate
        low_freq: low cutoff (Hz)
        high_freq: high cutoff (Hz)

    Returns:
        Filtered signal or original if filtering fails.
    """
    try:
        from scipy.signal import butter, filtfilt
    except Exception:
        return y
    nyq = 0.5 * sr
    low = max(0.0001, float(low_freq) / nyq)
    high = min(0.9999, float(high_freq) / nyq)
    if low >= high:
        return y
    b, a = butter(4, [low, high], btype='band')
    try:
        return filtfilt(b, a, y)
    except Exception:
        return y


def normalize_audio(y: np.ndarray, method: str = 'rms', target_rms: float = 0.1) -> np.ndarray:
    """
    Normalize a 1-D audio signal.

    Args:
        y: input signal (numpy array)
        method: 'peak' or 'rms'
        target_rms: target RMS level when method == 'rms'

    Returns:
        Normalized signal as numpy array (float32)
    """
    import numpy as _np

    if y is None:
        return y
    arr = _np.asarray(y, dtype=_np.float32)
    if arr.size == 0:
        return arr

    if method == 'peak':
        peak = _np.max(_np.abs(arr))
        if peak > 0:
            return (arr / peak).astype(_np.float32)
        return arr
    if method == 'rms':
        rms = _np.sqrt(_np.mean(arr ** 2))
        if rms <= 0:
            return arr
        factor = float(target_rms) / (rms + 1e-12)
        return (arr * factor).astype(_np.float32)
    raise ValueError("method must be 'peak' or 'rms'")
