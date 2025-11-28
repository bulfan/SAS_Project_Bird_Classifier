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
