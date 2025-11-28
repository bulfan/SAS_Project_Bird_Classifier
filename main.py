"""
Bird Sound Classifier - Main Entry Point

This script demonstrates the basic pipeline for bird sound classification:
1. Load configuration (via Hydra)
2. Load dataset
3. Preprocess audio
4. Train/use classifier
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import BirdSoundDataset
from src.preprocessing.audio_processor import AudioProcessor
from src.models.classifier import BirdClassifier
# from src.training.trainer import Trainer  # Will be used later


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to run the bird sound classifier pipeline."""
    print("Bird Sound Classifier")
    print("=" * 40)

    # Print configuration
    print("\n1. Configuration loaded via Hydra:")
    print(OmegaConf.to_yaml(cfg))

    # Extract configuration values
    raw_dir = cfg.data.raw_dir
    processed_dir = cfg.data.processed_dir
    sample_rate = cfg.data.sample_rate
    train_ratio = cfg.data.train_ratio
    val_ratio = cfg.data.val_ratio
    test_ratio = cfg.data.test_ratio
    num_classes = cfg.model.num_classes

    print(f"   - Raw data directory: {raw_dir}")
    print(f"   - Processed directory: {processed_dir}")
    print(f"   - Sample rate: {sample_rate}")

    # Initialize components
    print("\n2. Initializing components...")
    dataset = BirdSoundDataset(data_dir=raw_dir, processed_dir=processed_dir)
    processor = AudioProcessor(sample_rate=sample_rate)
    classifier = BirdClassifier(num_classes=num_classes)

    print("   - Dataset initialized")
    print("   - Audio processor initialized")
    print("   - Classifier initialized")

    # Load and explore dataset
    print("\n3. Loading dataset...")
    dataset.load()

    summary = dataset.summary()
    print(f"   - Total samples: {summary['total_samples']}")
    print(f"   - Number of classes: {summary['num_classes']}")
    print(f"   - Classes: {summary['class_names']}")
    print("\n   Class distribution:")
    for class_name, count in summary['class_distribution'].items():
        print(f"      {class_name}: {count} samples")

    # Split dataset into train/validation/test
    print("\n4. Splitting dataset (stratified)...")
    train_dataset, val_dataset, test_dataset = dataset.split(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shuffle=True,
        seed=42
    )
    print(f"   - Training samples: {len(train_dataset)}")
    print(f"   - Validation samples: {len(val_dataset)}")
    print(f"   - Test samples: {len(test_dataset)}")

    print("\n5. Pipeline ready!")
    print("   Next steps:")
    print("   - Implement preprocessing in src/preprocessing/audio_processor.py")
    print("   - Implement model in src/models/classifier.py")
    print("   - Implement training in src/training/trainer.py")


if __name__ == "__main__":
    main()
