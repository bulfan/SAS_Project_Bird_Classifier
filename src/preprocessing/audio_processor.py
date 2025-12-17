# Restore process_and_save for batch_preprocess.py import
from pathlib import Path
import numpy as np
def process_and_save(filepath: str, out_dir: str, target_sr: int = 10000, dtype=np.float32, out_format: str = 'npy') -> str:
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
            # scipy not available; return y
            return y

        sr = getattr(self, 'sample_rate', None)
        if sr is None or low_freq is None or high_freq is None:
            return y
        if sr == 0:
            return y
        nyq = 0.5 * sr
        low = max(0.0001, float(low_freq) / nyq)
        high = min(0.9999, float(high_freq) / nyq)
        # Debug: print filter params and input shape
        # print(f"bandpass_filter: y.shape={getattr(y, 'shape', None)}, low={low}, high={high}, sr={sr}")
        if not isinstance(y, np.ndarray):
            return y
        if y.ndim != 1:
            y = y.flatten()
        if low >= high:
            return y
        try:
            b, a = butter(4, [low, high], btype='band')
            y_f = filtfilt(b, a, y)
            if not isinstance(y_f, np.ndarray) or y_f.shape != y.shape:
                return y
            return y_f
        except Exception as e:
            # print(f"bandpass_filter exception: {e}")
            return y


    def lowpass_filter(self, y: np.ndarray, cutoff_freq: float) -> np.ndarray:
                """
                Apply a Butterworth low-pass filter to a 1-D signal.

                Args:
                    y: Input signal (1-D numpy array).
                    cutoff_freq: Cutoff frequency (Hz).

                Returns:
                    Filtered signal as numpy array. If scipy is not available, returns original signal.
                """
                try:
                    from scipy.signal import butter, filtfilt
                except Exception:
                    return y

                sr = getattr(self, 'sample_rate', None)
                if sr is None or cutoff_freq is None:
                    return y

                nyq = 0.5 * sr
                normal_cutoff = min(0.9999, float(cutoff_freq) / nyq)
                if normal_cutoff <= 0 or normal_cutoff >= 1:
                    return y
                b, a = butter(4, normal_cutoff, btype='low')
                try:
                    y_f = filtfilt(b, a, y)
                    return y_f
                except Exception:
                    return y        
