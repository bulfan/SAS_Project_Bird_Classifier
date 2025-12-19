"""
Bird Sound Classifier - Main Entry Point

This script demonstrates the basic pipeline for bird sound classification:
1. Load configuration (via Hydra)
2. Load dataset
3. Split dataset (BEFORE any analysis/preprocessing)
4. Run data exploration on TRAINING data only
5. Run spectral analysis on TRAINING data only
6. Preprocess audio (using config parameters)
7. Test preprocessing on a sample and visualize
"""

import time
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from src.models.KNN import KNNClassifier
from src.models.random_forest import RFClassifier
from src.data.dataset import BirdSoundDataset
from src.preprocessing.audio_processor import AudioProcessor
from src.models.classifier import BirdClassifier
from src.analysis.time_analysis import TimeAnalysisPipeline
from src.analysis.spectral_analysis import SpectralAnalysisPipeline
from src.features.scaler import StandardScaler


def test_preprocessing_on_sample(cfg: DictConfig, train_dataset, spectral_pipeline) -> None:
    """
    Test preprocessing on samples from training set and generate spectrograms.
    
    Files to test are specified via cfg.preprocessing.test_files.
    Supports both file stems (strings) and indices (ints).
    
    Args:
        cfg: Hydra config object
        train_dataset: Training dataset to sample from
        spectral_pipeline: SpectralAnalysisPipeline for spectrogram generation
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING TEST: Samples from Training Set")
    print("=" * 60)
    
    if len(train_dataset) == 0:
        print("   Error: Training dataset is empty!")
        return
    
    # Get test files from config
    test_files_config = getattr(cfg.preprocessing, 'test_files', [])
    
    # If no files specified, default to first file
    if not test_files_config:
        test_indices = [0]
    else:
        test_indices = []
        for item in test_files_config:
            if isinstance(item, int):
                # Direct index
                test_indices.append(item)
            else:
                # File stem - search for matching file
                for idx, (file_path, _) in enumerate(train_dataset.samples):
                    if Path(file_path).stem == item:
                        test_indices.append(idx)
                        break
    
    # Remove duplicates and sort
    test_indices = sorted(set(test_indices))
    
    print(f"\n   Testing {len(test_indices)} file(s): indices {test_indices}")
    
    # Initialize processor with config
    processor = AudioProcessor(cfg=cfg.preprocessing)
    print(f"\n   Processor settings:")
    print(f"     - Sample rate: {processor.sample_rate} Hz")
    print(f"     - Highpass cutoff: {processor.highpass_cutoff} Hz")
    print(f"     - Lowpass cutoff: {processor.lowpass_cutoff} Hz")
    print(f"     - Gate threshold: {processor.gate_threshold_db} dB")
    
    # Setup output directory for test
    test_output_dir = Path(cfg.analysis.spectral.output_dir) / "preprocessing_test"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Temporarily modify spectral pipeline for test output
    original_output_dir = spectral_pipeline.output_dir
    original_save = spectral_pipeline.save_plots
    spectral_pipeline.output_dir = test_output_dir
    spectral_pipeline.save_plots = True
    
    # Get skip flags from config
    skip_filters = getattr(cfg.preprocessing, 'skip_filters', False)
    skip_spectral_gate = getattr(cfg.preprocessing, 'skip_spectral_gate', False)
    
    # Process each test file
    for test_idx in test_indices:
        if test_idx >= len(train_dataset.samples):
            print(f"\n   ✗ Index {test_idx} out of range (dataset has {len(train_dataset.samples)} files)")
            continue
        
        file_path, label_idx = train_dataset.samples[test_idx]
        class_name = train_dataset.get_class_name(label_idx)
        
        print(f"\n   File {test_idx}: {Path(file_path).name} ({class_name})")
        
        try:
            # Load original audio
            original_samples, sr = processor.load_audio(file_path)
            print(f"     - Duration: {len(original_samples) / sr:.2f}s, Samples: {len(original_samples):,}")
            print(f"     - Original range: [{original_samples.min():.4f}, {original_samples.max():.4f}]")
            
            # Apply preprocessing
            start_time = time.time()
            processed_samples = processor.process(
                original_samples,
                skip_filters=skip_filters,
                skip_spectral_gate=skip_spectral_gate
            )
            elapsed = time.time() - start_time
            
            print(f"     - Processing time: {elapsed:.2f}s")
            print(f"     - Processed range: [{processed_samples.min():.4f}, {processed_samples.max():.4f}]")
            
            # Calculate energy reduction
            original_energy = np.sum(original_samples ** 2)
            processed_energy = np.sum(processed_samples ** 2)
            energy_reduction = (1 - processed_energy / original_energy) * 100 if original_energy > 0 else 0
            print(f"     - Energy reduction: {energy_reduction:.1f}%")
            
            # Set sample rate for spectral pipeline
            spectral_pipeline.sample_rate = sr
            
            # Plot original spectrogram
            _plot_spectrogram_from_samples(
                spectral_pipeline, original_samples, sr,
                title=f"ORIGINAL: {Path(file_path).name}",
                save_path=test_output_dir / f"{Path(file_path).stem}_original_spectrogram.png"
            )
            
            # Plot processed spectrogram
            _plot_spectrogram_from_samples(
                spectral_pipeline, processed_samples, sr,
                title=f"PROCESSED: {Path(file_path).name}",
                save_path=test_output_dir / f"{Path(file_path).stem}_processed_spectrogram.png"
            )
            
            print(f"     ✓ Spectrograms saved")
            
        except Exception as e:
            print(f"     ✗ Error processing: {e}")
    
    # Restore original settings
    spectral_pipeline.output_dir = original_output_dir
    spectral_pipeline.save_plots = original_save
    
    print(f"\n   Output directory: {test_output_dir}")
    print("=" * 60)


def _plot_spectrogram_from_samples(
    pipeline: SpectralAnalysisPipeline,
    samples: np.ndarray,
    sample_rate: int,
    title: str,
    save_path: Path
) -> None:
    """
    Helper function to plot spectrogram directly from samples array.
    
    Args:
        pipeline: SpectralAnalysisPipeline instance
        samples: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        title: Plot title
        save_path: Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    # Set sample rate for pipeline
    pipeline.sample_rate = sample_rate
    
    # Compute spectrogram
    times, frequencies, spectrogram = pipeline.compute_spectrogram(samples)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    eps = 1e-10
    spectrogram_db = 20 * np.log10(spectrogram + eps)
    im = ax.pcolormesh(times, frequencies, spectrogram_db,
                       shading='gouraud', cmap='viridis')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnitude (dB)')
    
    # Log scale settings
    ax.set_yscale('log')
    max_freq = sample_rate / 2
    band_ticks = [t for t in [100, 500, 1000, 5000, 10000, max_freq] if t <= max_freq]
    ax.set_yticks(band_ticks)
    ax.set_yticklabels([str(int(t)) for t in band_ticks])
    ax.set_ylim(20, max_freq)
    ax.set_ylabel('Frequency (Hz, log scale)')
    
    ax.set_xlabel('Time (seconds)')
    ax.set_title(title)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_preprocessing_on_splits(cfg: DictConfig, 
                                 train_dataset, 
                                 val_dataset, 
                                 test_dataset) -> None:
    """
    Run preprocessing on train, validation, and test splits separately.
    
    Args:
        cfg: Hydra config object
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
    """
    print("\n" + "=" * 60)
    print("PREPROCESSING: Running on Train/Val/Test Splits")
    print("=" * 60)
    
    # Get preprocessing config
    preproc_cfg = cfg.preprocessing
    base_output_dir = Path(cfg.data.processed_dir)
    
    print(f"\n   Output base directory: {base_output_dir}")
    print(f"   Preprocessing config:")
    print(f"     - Sample rate: {preproc_cfg.sample_rate} Hz")
    print(f"     - Highpass cutoff: {preproc_cfg.highpass_cutoff} Hz")
    print(f"     - Lowpass cutoff: {preproc_cfg.lowpass_cutoff} Hz")
    print(f"     - Gate threshold: {preproc_cfg.gate_threshold_db} dB")
    print(f"     - Workers: {preproc_cfg.n_workers}")
    
    # Process each split
    splits = [
        ('train', train_dataset),
        ('val', val_dataset),
        ('test', test_dataset)
    ]
    
    for split_name, split_dataset in splits:
        if len(split_dataset) == 0:
            print(f"\n   Skipping {split_name} split (empty)")
            continue
        
        print(f"\n   Processing {split_name} split ({len(split_dataset)} files)...")
        
        # Create temporary directory structure for this split
        split_output_dir = base_output_dir / split_name
        
        # Get unique class directories from the split
        class_files = {}
        for file_path, label_idx in split_dataset.samples:
            class_name = split_dataset.get_class_name(label_idx)
            if class_name not in class_files:
                class_files[class_name] = []
            class_files[class_name].append(file_path)
        
        # Process files for each class
        processor = AudioProcessor(cfg=preproc_cfg)
        total_processed = 0
        total_failed = 0
        
        for class_name, files in class_files.items():
            class_output_dir = split_output_dir / class_name
            class_output_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in files:
                try:
                    # Load audio
                    samples, sr = processor.load_audio(file_path)
                    
                    # Apply preprocessing
                    processed = processor.process(
                        samples,
                        skip_filters=getattr(preproc_cfg, 'skip_filters', False),
                        skip_spectral_gate=getattr(preproc_cfg, 'skip_spectral_gate', False)
                    )
                    
                    # Save processed audio
                    output_path = class_output_dir / (Path(file_path).stem + ".npy")
                    np.save(output_path, processed.astype(np.float32))
                    total_processed += 1
                    
                except Exception as e:
                    print(f"     ✗ Failed {Path(file_path).name}: {e}")
                    total_failed += 1
        
        print(f"     ✓ {split_name}: {total_processed} processed, {total_failed} failed")
        print(f"     Output: {split_output_dir}")
    
    print("\n" + "=" * 60)


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
    train_ratio = cfg.data.train_ratio
    val_ratio = cfg.data.val_ratio
    test_ratio = cfg.data.test_ratio
    num_classes = cfg.model.num_classes

    print(f"   - Raw data directory: {raw_dir}")
    print(f"   - Processed directory: {processed_dir}")

    # Initialize components
    print("\n2. Initializing components...")
    dataset = BirdSoundDataset(data_dir=raw_dir, processed_dir=processed_dir)
    print("   - Dataset initialized")

    processor = AudioProcessor(cfg=cfg.preprocessing)
    print(f"   - Audio processor initialized (hp={processor.highpass_cutoff}Hz, lp={processor.lowpass_cutoff}Hz)")

    classifier = BirdClassifier(num_classes=num_classes)
    print("   - Classifier initialized")

    # Initialize analysis pipelines with config
    time_pipeline = TimeAnalysisPipeline(cfg_dict)
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
    time_results = time_pipeline.run(
        dataset=dataset,
        train_dataset=train_dataset
    )

    # Spectral Analysis Pipeline (frequency-domain)
    spectral_results = spectral_pipeline.run(
        dataset=dataset,
        train_dataset=train_dataset
    )

    # Test preprocessing on a sample first
    print("\n6. Testing Preprocessing...")
    if cfg.preprocessing.test_mode:
        test_preprocessing_on_sample(cfg, dataset, spectral_pipeline)
    else:
        print("   Skipping preprocessing test (test_mode disabled)")
    
    # Run preprocessing on all splits
    print("\n7. Running Preprocessing on All Splits...")
    if cfg.preprocessing.enabled:
        run_preprocessing_on_splits(cfg, train_dataset, val_dataset, test_dataset)
    else:
        print("   Preprocessing pipeline is disabled in config. Skipping.")

    # ==========================================
    # 8. Feature Extraction
    # ==========================================
    print("\n8. Feature Extraction (From Processed Data)...")
    from src.features.feature_extractor import FeatureExtractor
    
    # Initialize Extractor
    extractor = FeatureExtractor(cfg)
    processed_base = Path(cfg.data.processed_dir)
    
    # Extract features for each split
    print("   Extracting Training Features...")
    X_train, y_train = extractor.process_split(train_dataset, processed_base / "train")
    
    print("   Extracting Validation Features...")
    X_val, y_val = extractor.process_split(val_dataset, processed_base / "val")
    
    print("   Extracting Test Features...")
    X_test, y_test = extractor.process_split(test_dataset, processed_base / "test")
    
    print(f"   Feature Matrix Shapes: Train {X_train.shape}, Val {X_val.shape}, Test {X_test.shape}")
   
    # ==========================================
    # 9. Scaling & Model Selection (The Fix)
    # ==========================================
    print("\n9. Model Optimization...")    
    # 1. SCALE ALL DATA CORRECTLY
    # Initialize scaler
    temp_scaler = StandardScaler()
    X_train_temp = temp_scaler.fit_transform(X_train)
    X_val_temp = temp_scaler.transform(X_val)
    
    # 2. HYPERPARAMETER TUNING (Find best k)
    print("   Tuning KNN k-value...")
    best_k = 5
    best_acc = 0
    
    # Try odd numbers from 1 to 15
    for k in [1, 3, 5, 7, 9, 11, 13, 15]:
        knn = KNNClassifier(k=k)
        knn.fit(X_train_scaled, y_train)
        
        # Evaluate on Validation Set
        val_preds = knn.predict(X_val_scaled)
        acc = accuracy_score(y_val, val_preds) # Assuming sklearn import or manual calc
        
        print(f"     k={k}: Validation Acc = {acc*100:.1f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_k = k

    print(f"   ✓ Best k found: {best_k} (Acc: {best_acc*100:.1f}%)")

    # ==========================================
    # 10. Final Training (Merging Train + Val)
    # ==========================================
    print(f"\n10. Final Training (Train + Val) with k={best_k}...")

    # Get model type from config
    model_type = getattr(cfg.model, 'type', 'knn')
    use_validation = getattr(cfg.training, 'use_validation', True)

    # MERGE Train and Validation sets for maximum power
    X_final = np.concatenate((X_train_scaled, X_val_scaled))
    y_final = np.concatenate((y_train, y_val))

    # Determine which models to train
    models_to_train = []
    if model_type in ['knn', 'both']:
        models_to_train.append(('KNN', KNNClassifier(k=best_k)))
    if model_type in ['random_forest', 'both']:
        models_to_train.append(('RandomForest', RFClassifier(cfg=cfg.model.random_forest)))
    
    # Store results for model selection
    results = {}
    
    for model_name, model in models_to_train:
        print(f"\n   --- {model_name} ---")
        
        # Train on training set
        model.fit(X_final, y_final)
        
        # Evaluate on validation set (for model selection)
        if use_validation and len(X_val) > 0:
            val_preds = model.predict(X_val_scaled)
            val_acc = accuracy_score(y_val, val_preds)
            print(f"   Validation Accuracy: {val_acc*100:.2f}%")
            results[model_name] = {'model': model, 'val_acc': val_acc}
        else:
            results[model_name] = {'model': model, 'val_acc': None}
    
    # Select best model based on validation accuracy
    if use_validation and len(results) > 1:
        best_model_name = max(results, key=lambda x: results[x]['val_acc'] or 0)
        print(f"\n   Best model (by validation): {best_model_name}")
    else:
        best_model_name = list(results.keys())[0]
    
    best_model = results[best_model_name]['model']
    
    # ==========================================
    # 11. Final Evaluation on Test Set
    # ==========================================
    print(f"\n11. Final Evaluation ({best_model_name} on Test Set)...")
    
    # Predict on test set
    predictions = best_model.predict(X_test_scaled)
    
    # Calculate metrics using sklearn
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    
    print(f"\n   Overall Metrics:")
    print(f"   {'='*40}")
    print(f"   Accuracy:  {accuracy*100:.2f}%")
    print(f"   Precision: {precision*100:.2f}% (weighted)")
    print(f"   Recall:    {recall*100:.2f}% (weighted)")
    print(f"   F1-Score:  {f1*100:.2f}% (weighted)")
    
    # Detailed classification report
    print(f"\n   Per-Class Report:")
    print(f"   {'='*40}")
    class_names = dataset.class_names
    print(classification_report(y_test, predictions, target_names=class_names, zero_division=0))
    
    print("\n" + "="*40)
    print("Pipeline Complete!")

if __name__ == "__main__":
    main()
