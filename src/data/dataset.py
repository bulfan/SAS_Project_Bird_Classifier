"""
Dataset loading and handling for bird sound classification.
"""


class BirdSoundDataset:
    """Dataset class for loading and handling bird sound data."""

    def __init__(self, data_dir: str):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to the directory containing audio files.
        """
        self.data_dir = data_dir

    def load(self):
        """Load the dataset from the data directory."""
        raise NotImplementedError("Subclasses must implement load method")

    def __len__(self):
        """Return the number of samples in the dataset."""
        raise NotImplementedError("Subclasses must implement __len__ method")

    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        raise NotImplementedError("Subclasses must implement __getitem__ method")
