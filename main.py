"""
Bird Sound Classifier - Main Entry Point

This script demonstrates the basic pipeline for bird sound classification:
1. Load configuration (via Hydra)
2. Load dataset
3. Split dataset (BEFORE any analysis/preprocessing)
4. Run data exploration on TRAINING data only
5. Run spectral analysis on TRAINING data only
6. Preprocess audio (fit on train, apply to all)
7. Train/use classifier
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data.dataset import BirdSoundDataset
from src.preprocessing.audio_processor import AudioProcessor
from src.models.classifier import BirdClassifier
from src.utils.plotting import plot_waveform
from src.analysis.time_domain import TimeDomainAnalysis
from src.analysis.data_exploration import  DataExplorationPipeline
from src.analysis.spectral_analysis import SpectralAnalysisPipeline
# from src.training.trainer import Trainer  # Will be used later
import os


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to run the bird sound classifier pipeline."""
    print("Bird Sound Classifier")
    print("=" * 40)

    # Print configuration
    print("\n1. Configuration loaded via Hydra:")
    print(OmegaConf.to_yaml(cfg))

    # Convert to dict for pipeline initialization
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

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
    print("   - Dataset initialized")

    processor = AudioProcessor(sample_rate=sample_rate)
    print("   - Audio processor initialized")

    classifier = BirdClassifier(num_classes=num_classes)
    print("   - Classifier initialized")

    # Initialize analysis pipelines with config
    exploration_pipeline = DataExplorationPipeline(cfg_dict)
    print("   - Data Exploration Pipeline initialized")

    spectral_pipeline = SpectralAnalysisPipeline(cfg_dict)
    print("   - Spectral Analysis Pipeline initialized")


    # Load and explore dataset
    print("\n3. Loading dataset...")
    dataset.load()
    
    # Print dataset summary if enabled
    dataset.print_summary("Full Dataset Summary")

    # Split dataset into train/validation/test
    print("\n4. Splitting dataset (stratified)...")
    train_dataset, val_dataset, test_dataset = dataset.split(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        shuffle=True,
        seed=cfg.data.seed
    )
    
    # Print split summaries
    train_dataset.print_summary("Training Dataset")
    val_dataset.print_summary("Validation Dataset")
    test_dataset.print_summary("Test Dataset")

    # Run analysis pipelines based on configuration
    # Pipeline modes: "single", "folder", "train", "dataset"
    print("\n5. Running Analysis Pipelines...")
    
    # Data Exploration Pipeline (time-domain)
    exploration_results = exploration_pipeline.run(
        dataset=dataset,
        train_dataset=train_dataset
    )

    # Spectral Analysis Pipeline (frequency-domain)
    spectral_results = spectral_pipeline.run(
        dataset=dataset,
        train_dataset=train_dataset
    )

    # TODO: Next steps in pipeline
    print("\n6. Pipeline ready!")
    print("   Next steps:")
    print("   - Implement preprocessing in src/preprocessing/audio_processor.py")
    print("   - Implement model in src/models/classifier.py")
    print("   - Implement training in src/training/trainer.py")

    # Plot a sample waveform
    print("\n6. Plotting sample waveform...")
    audio_dir = raw_dir
    if os.path.exists(audio_dir):
        # Get first class subdirectory
        subdirs = [d for d in os.listdir(audio_dir) if os.path.isdir(os.path.join(audio_dir, d))]
        if subdirs:
            class_dir = os.path.join(audio_dir, subdirs[0])
            audio_files = [f for f in os.listdir(class_dir) if f.endswith(('.wav', '.mp3', '.ogg'))]
            if audio_files:
                sample_file = os.path.join(class_dir, audio_files[0])
                print(f"   File: {sample_file}")
                
                # Compute time-domain analysis
                print("\n7. Time-domain analysis...")
                tda = TimeDomainAnalysis(sample_rate=sample_rate)
                tda.load_audio(sample_file)
                tda.run_all()
                
                # Now plot the waveform
                print("\n8. Displaying waveform plot...")
                plot_waveform(sample_file, sr=sample_rate)
            else:
                print("   No audio files found in class directory")
        else:
            print("   No class subdirectories found")
    else:
        print(f"   Audio directory not found: {audio_dir}")


if __name__ == "__main__":
    main()
