"""
Bird Sound Classifier - Main Entry Point

This script demonstrates the basic pipeline for bird sound classification:
1. Load dataset
2. Preprocess audio
3. Train/use classifier
"""

from src.data.dataset import BirdSoundDataset
from src.preprocessing.audio_processor import AudioProcessor
from src.models.classifier import BirdClassifier
from src.training.trainer import Trainer


def main():
    """Main function to run the bird sound classifier pipeline."""
    print("Bird Sound Classifier")
    print("=" * 40)

    # Initialize components
    print("\n1. Initializing components...")
    dataset = BirdSoundDataset(data_dir="data/raw")
    processor = AudioProcessor(sample_rate=22050)
    classifier = BirdClassifier(num_classes=10)

    print("   - Dataset initialized")
    print("   - Audio processor initialized")
    print("   - Classifier initialized")

    print("\n2. Pipeline ready for implementation")
    print("   - Implement data loading in src/data/dataset.py")
    print("   - Implement preprocessing in src/preprocessing/audio_processor.py")
    print("   - Implement model in src/models/classifier.py")
    print("   - Implement training in src/training/trainer.py")


if __name__ == "__main__":
    main()
