"""
Dataset loading and handling for bioacoustic classification.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import random
import re


class BirdSoundDataset:
    """Dataset class for loading and handling bioacoustic data."""

    def __init__(self, data_dir: str, processed_dir: Optional[str] = None):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to the directory containing raw audio files.
            processed_dir: Path to the directory for processed data (optional).
        """
        self.data_dir = Path(data_dir)
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.samples: List[Tuple[str, int]] = []  # List of (file_path, label)
        self.class_names: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}
        self._loaded = False

    def load(self) -> "BirdSoundDataset":
        """
        Load the dataset from the data directory.
        
        Supports either directory structure:
            data_dir/
                class1/
                    audio1.mp3
                    audio2.mp3
                class2/
                    audio1.mp3
                    ...

        Or flat-file structure where each file name encodes class:
            data_dir/
                species-a-1.mp3
                species-a-2.mp3
                species-b.mp3
        
        Returns:
            self for method chaining.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data directory not found: {self.data_dir}"
            )

        self.samples = []
        self.class_names = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Get all subdirectories (each represents a class/species)
        class_dirs = sorted([
            d for d in self.data_dir.iterdir()
            if d.is_dir()
        ])

        audio_extensions = {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}

        if class_dirs:
            # Standard class-subfolder layout
            for idx, class_dir in enumerate(class_dirs):
                class_name = class_dir.name
                self.class_names.append(class_name)
                self.class_to_idx[class_name] = idx
                self.idx_to_class[idx] = class_name

                audio_files = [
                    f for f in class_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in audio_extensions
                ]

                for audio_file in audio_files:
                    self.samples.append((str(audio_file), idx))
        else:
            # Flat-file layout: infer class name from file stem
            audio_files = [
                f for f in self.data_dir.iterdir()
                if f.is_file() and f.suffix.lower() in audio_extensions
            ]
            if not audio_files:
                raise ValueError(
                    f"No class directories or audio files found in {self.data_dir}"
                )

            class_to_files: Dict[str, List[Path]] = {}
            for audio_file in sorted(audio_files):
                class_name = self._infer_flat_file_class_name(audio_file.stem)
                class_to_files.setdefault(class_name, []).append(audio_file)

            for idx, class_name in enumerate(sorted(class_to_files.keys())):
                self.class_names.append(class_name)
                self.class_to_idx[class_name] = idx
                self.idx_to_class[idx] = class_name

                for audio_file in class_to_files[class_name]:
                    self.samples.append((str(audio_file), idx))
        
        self._loaded = True
        return self

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if not self._loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        """
        Return a sample from the dataset.
        
        Args:
            idx: Index of the sample.
            
        Returns:
            Tuple of (file_path, label_index).
        """
        if not self._loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} out of range for dataset "
                f"with {len(self.samples)} samples"
            )
        return self.samples[idx]

    def get_class_name(self, idx: int) -> str:
        """
        Get the class name for a given label index.
        
        Args:
            idx: Label index.
            
        Returns:
            Class name string.
        """
        return self.idx_to_class.get(idx, "Unknown")

    def get_class_index(self, class_name: str) -> int:
        """
        Get the label index for a given class name.
        
        Args:
            class_name: Name of the class.
            
        Returns:
            Label index.
        """
        return self.class_to_idx.get(class_name, -1)

    def get_num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        return len(self.class_names)

    def get_samples_by_class(self, class_idx: int) -> List[Tuple[str, int]]:
        """
        Get all samples belonging to a specific class.
        
        Args:
            class_idx: Index of the class.
            
        Returns:
            List of (file_path, label) tuples for the class.
        """
        return [
            (path, label) for path, label in self.samples
            if label == class_idx
        ]

    def _create_subset(
        self,
        samples: List[Tuple[str, int]]
    ) -> "BirdSoundDataset":
        """
        Create a new dataset subset with the given samples.

        Args:
            samples: List of (file_path, label) tuples.

        Returns:
            New BirdSoundDataset instance with the subset.
        """
        processed_path = str(self.processed_dir) if self.processed_dir else None
        subset = BirdSoundDataset(str(self.data_dir), processed_path)
        subset.samples = samples
        subset.class_names = self.class_names.copy()
        subset.class_to_idx = self.class_to_idx.copy()
        subset.idx_to_class = self.idx_to_class.copy()
        subset._loaded = True
        return subset

    def split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = True,
        seed: Optional[int] = None
    ) -> Tuple["BirdSoundDataset", "BirdSoundDataset", "BirdSoundDataset"]:
        """
        Split the dataset into training, validation, and test sets.

        Uses stratified splitting to maintain class distribution across splits.

        Args:
            train_ratio: Ratio of samples for training (default: 0.7).
            val_ratio: Ratio of samples for validation (default: 0.15).
            test_ratio: Ratio of samples for testing (default: 0.15).
            shuffle: Whether to shuffle before splitting.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset).

        Raises:
            RuntimeError: If dataset is not loaded.
            ValueError: If ratios don't sum to 1.0 (within tolerance).
        """
        if not self._loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        # Validate ratios
        total_ratio = train_ratio + val_ratio + test_ratio
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(
                f"Ratios must sum to 1.0, got {total_ratio:.2f} "
                f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
            )

        if seed is not None:
            random.seed(seed)

        # Stratified split: process each class separately
        train_samples = []
        val_samples = []
        test_samples = []

        for class_idx in range(len(self.class_names)):
            class_samples = self.get_samples_by_class(class_idx)

            if shuffle:
                random.shuffle(class_samples)

            n_samples = len(class_samples)
            n_train = int(n_samples * train_ratio)
            n_val = int(n_samples * val_ratio)
            # Remaining samples go to test to avoid rounding issues
            n_test = n_samples - n_train - n_val

            # Keep at least one training sample for very small classes.
            if n_samples > 0 and n_train == 0:
                n_train = 1
                if n_val > 0:
                    n_val -= 1
                elif n_test > 0:
                    n_test -= 1

            train_samples.extend(class_samples[:n_train])
            val_samples.extend(class_samples[n_train:n_train + n_val])
            test_samples.extend(class_samples[n_train + n_val:])

        # Shuffle the final lists
        if shuffle:
            random.shuffle(train_samples)
            random.shuffle(val_samples)
            random.shuffle(test_samples)

        # Create dataset subsets
        train_dataset = self._create_subset(train_samples)
        val_dataset = self._create_subset(val_samples)
        test_dataset = self._create_subset(test_samples)

        return train_dataset, val_dataset, test_dataset

    @staticmethod
    def _infer_flat_file_class_name(stem: str) -> str:
        """
        Infer class label from a flat-file stem.

        - Lowercases and normalizes separators
        - Removes trailing numeric clip suffixes like "-1" / "_2"
        """
        normalized = stem.strip().lower().replace(" ", "-").replace("_", "-")
        normalized = re.sub(r'-{2,}', '-', normalized)
        normalized = re.sub(r'-\d+$', '', normalized)
        return normalized

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the dataset.
        
        Returns:
            Dictionary with dataset statistics.
        """
        if not self._loaded:
            return {"loaded": False}
        
        class_distribution = {}
        for class_name in self.class_names:
            class_idx = self.class_to_idx[class_name]
            count = len(self.get_samples_by_class(class_idx))
            class_distribution[class_name] = count
        
        return {
            "loaded": True,
            "total_samples": len(self.samples),
            "num_classes": len(self.class_names),
            "class_names": self.class_names,
            "class_distribution": class_distribution,
            "data_dir": str(self.data_dir)
        }

    def print_summary(self, title: str = "Dataset Summary") -> None:
        """
        Print a formatted summary of the dataset.
        
        Args:
            title: Title to display at the top of the summary.
        """
        stats = self.summary()
        
        print(f"\n   {title}")
        print(f"   {'=' * 50}")
        
        if not stats.get("loaded", False):
            print("   Dataset not loaded.")
            return
        
        print(f"   Data Directory: {stats['data_dir']}")
        print(f"   Total Samples: {stats['total_samples']}")
        print(f"   Number of Classes: {stats['num_classes']}")
        print(f"\n   Class Distribution:")
        print(f"   {'-' * 30}")
        
        for class_name, count in stats['class_distribution'].items():
            percentage = (
                (count / stats['total_samples']) * 100
                if stats['total_samples'] > 0 else 0.0
            )
            bar = '#' * int(percentage / 5)
            print(f"   {class_name:12} | {count:4} samples | {percentage:5.1f}% | {bar}")
        
        print(f"   {'-' * 30}")

    def __repr__(self) -> str:
        """Return string representation of the dataset."""
        if not self._loaded:
            return (
                f"BirdSoundDataset(data_dir='{self.data_dir}', "
                f"loaded=False)"
            )
        return (
            f"BirdSoundDataset(data_dir='{self.data_dir}', "
            f"samples={len(self.samples)}, classes={len(self.class_names)})"
        )
