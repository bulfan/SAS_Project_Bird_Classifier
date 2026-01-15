"""
Bird Sound Classifier - Main Entry Point

Pipeline Steps:
    1. Load configuration (Hydra)
    2. Initialize components (dataset, processor, analysis pipelines)
    3. Load and explore raw audio data
    4. Split into train/val/test sets
    5. Run time-domain and spectral analysis
    6. Test preprocessing on sample files
    7. Preprocess all audio splits
    8. Extract features from processed data
    9. Scale features and tune hyperparameters
    10. Final training on combined train+val
    11. Evaluate on test set
    12. Display feature importance
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
from src.analysis.time_analysis import TimeAnalysisPipeline
from src.analysis.spectral_analysis import SpectralAnalysisPipeline
from src.features.scaler import StandardScaler
from src.features.feature_extractor import FeatureExtractor


# =============================================================================
# PREPROCESSING UTILITIES
# =============================================================================

def test_preprocessing_on_sample(cfg: DictConfig, train_dataset, spectral_pipeline) -> None:
    """Test preprocessing on samples from training set and generate spectrograms."""
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
    
    test_indices = sorted(set(test_indices))
    print(f"\n   Testing {len(test_indices)} file(s): indices {test_indices}")
    
    processor = AudioProcessor(cfg=cfg.preprocessing)
    print(f"\n   Processor settings:")
    print(f"     - Sample rate: {processor.sample_rate} Hz")
    print(f"     - Highpass cutoff: {processor.highpass_cutoff} Hz")
    print(f"     - Lowpass cutoff: {processor.lowpass_cutoff} Hz")
    print(f"     - Gate threshold: {processor.gate_threshold_db} dB")
    
    # Setup test output directory and temporarily redirect spectral pipeline output
    test_output_dir = Path(cfg.analysis.spectral.output_dir) / "preprocessing_test"
    test_output_dir.mkdir(parents=True, exist_ok=True)
    original_output_dir = spectral_pipeline.output_dir
    original_save = spectral_pipeline.save_plots
    spectral_pipeline.output_dir = test_output_dir
    spectral_pipeline.save_plots = True
    
    skip_filters = getattr(cfg.preprocessing, 'skip_filters', False)
    skip_spectral_gate = getattr(cfg.preprocessing, 'skip_spectral_gate', False)
    
    for test_idx in test_indices:
        if test_idx >= len(train_dataset.samples):
            print(f"\n   ✗ Index {test_idx} out of range (dataset has {len(train_dataset.samples)} files)")
            continue
        
        file_path, label_idx = train_dataset.samples[test_idx]
        class_name = train_dataset.get_class_name(label_idx)
        
        print(f"\n   File {test_idx}: {Path(file_path).name} ({class_name})")
        
        try:
            original_samples, sr = processor.load_audio(file_path)
            print(f"     - Duration: {len(original_samples) / sr:.2f}s, Samples: {len(original_samples):,}")
            print(f"     - Original range: [{original_samples.min():.4f}, {original_samples.max():.4f}]")
            
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
            
            spectral_pipeline.sample_rate = sr
            
            # Generate before/after spectrograms
            _plot_spectrogram_from_samples(
                spectral_pipeline, original_samples, sr,
                title=f"ORIGINAL: {Path(file_path).name}",
                save_path=test_output_dir / f"{Path(file_path).stem}_original_spectrogram.png"
            )
            _plot_spectrogram_from_samples(
                spectral_pipeline, processed_samples, sr,
                title=f"PROCESSED: {Path(file_path).name}",
                save_path=test_output_dir / f"{Path(file_path).stem}_processed_spectrogram.png"
            )
            
            print(f"     ✓ Spectrograms saved")
            
        except Exception as e:
            print(f"     ✗ Error processing: {e}")
    
    # Restore spectral pipeline to original state
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
    """Helper to plot spectrogram directly from audio samples."""
    import matplotlib.pyplot as plt
    
    pipeline.sample_rate = sample_rate
    times, frequencies, spectrogram = pipeline.compute_spectrogram(samples)
    
    # Convert to dB scale and plot
    fig, ax = plt.subplots(figsize=(12, 6))
    eps = 1e-10
    spectrogram_db = 20 * np.log10(spectrogram + eps)
    im = ax.pcolormesh(times, frequencies, spectrogram_db,
                       shading='gouraud', cmap='viridis')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Magnitude (dB)')
    
    # Use log frequency scale for better visualization
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


# =============================================================================
# BATCH PREPROCESSING
# =============================================================================

def run_preprocessing_on_splits(cfg: DictConfig, 
                                 train_dataset, 
                                 val_dataset, 
                                 test_dataset) -> None:
    """Run preprocessing on train, validation, and test splits separately."""
    print("\n" + "=" * 60)
    print("PREPROCESSING: Running on Train/Val/Test Splits")
    print("=" * 60)
    
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
        ('test', test_dataset)]
    
    for split_name, split_dataset in splits:
        if len(split_dataset) == 0:
            print(f"\n   Skipping {split_name} split (empty)")
            continue
        
        print(f"\n   Processing {split_name} split ({len(split_dataset)} files)...")
        
        split_output_dir = base_output_dir / split_name
        
        # Group files by class for organized output
        class_files = {}
        for file_path, label_idx in split_dataset.samples:
            class_name = split_dataset.get_class_name(label_idx)
            if class_name not in class_files:
                class_files[class_name] = []
            class_files[class_name].append(file_path)
        

        processor = AudioProcessor(cfg=preproc_cfg)
        total_processed = 0
        total_failed = 0
        
        for class_name, files in class_files.items():
            class_output_dir = split_output_dir / class_name
            class_output_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in files:
                try:
                    samples, sr = processor.load_audio(file_path)
                    
                    processed = processor.process(
                        samples,
                        skip_filters=getattr(preproc_cfg, 'skip_filters', False),
                        skip_spectral_gate=getattr(preproc_cfg, 'skip_spectral_gate', False)
                    )
                    
                    output_path = class_output_dir / (Path(file_path).stem + ".npy")
                    np.save(output_path, processed.astype(np.float32))
                    total_processed += 1
                    
                except Exception as e:
                    print(f"     ✗ Failed {Path(file_path).name}: {e}")
                    total_failed += 1
        
        print(f"     ✓ {split_name}: {total_processed} processed, {total_failed} failed")
        print(f"     Output: {split_output_dir}")
    
    print("\n" + "=" * 60)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function to run the bird sound classifier pipeline."""
    print("Bird Sound Classifier")
    print("=" * 40)

    # -------------------------------------------------------------------------
    # Step 1: Load Configuration
    # -------------------------------------------------------------------------
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

    print(f"   - Raw data directory: {raw_dir}")
    print(f"   - Processed directory: {processed_dir}")

    # -------------------------------------------------------------------------
    # Step 2: Initialize Components
    # -------------------------------------------------------------------------
    print("\n2. Initializing components...")
    dataset = BirdSoundDataset(data_dir=raw_dir, processed_dir=processed_dir)
    print("   - Dataset initialized")

    processor = AudioProcessor(cfg=cfg.preprocessing)
    print(f"   - Audio processor initialized (hp={processor.highpass_cutoff}Hz, lp={processor.lowpass_cutoff}Hz)")

    time_pipeline = TimeAnalysisPipeline(cfg_dict)
    print("   - Data Exploration Pipeline initialized")

    spectral_pipeline = SpectralAnalysisPipeline(cfg_dict)
    print("   - Spectral Analysis Pipeline initialized")

    # -------------------------------------------------------------------------
    # Step 3: Load Dataset
    # -------------------------------------------------------------------------
    print("\n3. Loading dataset...")
    dataset.load()
    
    dataset.print_summary("Full Dataset Summary")

    # -------------------------------------------------------------------------
    # Step 4: Train/Val/Test Split
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # Step 5: Run Analysis Pipelines (time & frequency domain)
    # -------------------------------------------------------------------------
    print("\n5. Running Analysis Pipelines...")
    
    time_results = time_pipeline.run(
        dataset=dataset,
        train_dataset=train_dataset)
    spectral_results = spectral_pipeline.run(
        dataset=dataset,
        train_dataset=train_dataset)

    # -------------------------------------------------------------------------
    # Step 6: Test Preprocessing (visual verification for debugging)
    # -------------------------------------------------------------------------
    print("\n6. Testing Preprocessing...")
    if cfg.preprocessing.test_mode:
        test_preprocessing_on_sample(cfg, dataset, spectral_pipeline)
    else:
        print("   Skipping preprocessing test (test_mode disabled)")
    
    # -------------------------------------------------------------------------
    # Step 7: Batch Preprocessing (filter, denoise, save as .npy)
    # -------------------------------------------------------------------------
    print("\n7. Running Preprocessing on All Splits...")
    if cfg.preprocessing.enabled:
        run_preprocessing_on_splits(cfg, train_dataset, val_dataset, test_dataset)
    else:
        print("   Preprocessing pipeline is disabled in config. Skipping.")

    # -------------------------------------------------------------------------
    # Step 8: Feature Extraction (spectral features from processed audio)
    # -------------------------------------------------------------------------
    print("\n8. Feature Extraction (From Processed Data)...")
    
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
   
    # -------------------------------------------------------------------------
    # Step 9: Hyperparameter Tuning (use validation set to find best k)
    # -------------------------------------------------------------------------
    print("\n9. Model Optimization...")
    
    # Scale train/val for tuning (test set stays untouched until final eval)
    temp_scaler = StandardScaler()
    X_train_temp = temp_scaler.fit_transform(X_train)
    X_val_temp = temp_scaler.transform(X_val)
    
    print("   Tuning KNN k-value...")
    best_k = 5
    best_acc = 0
    for k in [1, 3, 5, 7, 9, 11, 13, 15]:
        knn = KNNClassifier(k=k)
        knn.fit(X_train_temp, y_train)
        val_preds = knn.predict(X_val_temp)
        acc = accuracy_score(y_val, val_preds)
        
        print(f"     k={k}: Validation Acc = {acc*100:.1f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_k = k

    print(f"   ✓ Best k found: {best_k} (Acc: {best_acc*100:.1f}%)")

    # -------------------------------------------------------------------------
    # Step 10: Final Training (merge train+val for maximum data)
    # -------------------------------------------------------------------------
    print(f"\n10. Final Training (Train + Val) with k={best_k}...")

    model_type = getattr(cfg.model, 'type', 'knn')
    
    # Merge train and validation for final model
    X_final = np.concatenate((X_train, X_val))
    y_final = np.concatenate((y_train, y_val))
    print(f"   Combined Training Set: {len(X_final)} samples")
    
    # Re-fit scaler on combined data, then scale test set
    final_scaler = StandardScaler()
    X_final_scaled = final_scaler.fit_transform(X_final)
    X_test_scaled = final_scaler.transform(X_test)

    # Train selected model(s)
    if model_type in ['knn', 'both']:
        knn_final = KNNClassifier(k=best_k)
        knn_final.fit(X_final_scaled, y_final)
    if model_type in ['random_forest', 'both']:
        rf_final = RFClassifier(cfg=cfg.model.random_forest)
        rf_final.fit(X_final_scaled, y_final)
    
    # -------------------------------------------------------------------------
    # Step 11: Final Evaluation on unseen test set
    # -------------------------------------------------------------------------
    print(f"\n11. Final Evaluation on Test Set...")
    
    class_names = dataset.class_names
    
    if model_type in ['knn', 'both']:
        print(f"\n   --- Evaluation: KNN (k={best_k}) ---")
        knn_predictions = knn_final.predict(X_test_scaled)
        
        # Calculate metrics
        knn_accuracy = accuracy_score(y_test, knn_predictions)
        knn_precision = precision_score(y_test, knn_predictions, average='weighted', zero_division=0)
        knn_recall = recall_score(y_test, knn_predictions, average='weighted', zero_division=0)
        knn_f1 = f1_score(y_test, knn_predictions, average='weighted', zero_division=0)
        
        print(f"\n   Metrics:")
        print(f"   {'='*40}")
        print(f"   Accuracy:  {knn_accuracy*100:.2f}%")
        print(f"   Precision: {knn_precision*100:.2f}% (weighted)")
        print(f"   Recall:    {knn_recall*100:.2f}% (weighted)")
        print(f"   F1-Score:  {knn_f1*100:.2f}% (weighted)")
        
        print(f"\n   Per-Class Report:")
        print(f"   {'='*40}")
        print(classification_report(y_test, knn_predictions, target_names=class_names, zero_division=0))

    if model_type in ['random_forest', 'both']:
        print(f"\n   --- Evaluation: Random Forest ---")
        rf_predictions = rf_final.predict(X_test_scaled)
        
        # Calculate metrics
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        rf_precision = precision_score(y_test, rf_predictions, average='weighted', zero_division=0)
        rf_recall = recall_score(y_test, rf_predictions, average='weighted', zero_division=0)
        rf_f1 = f1_score(y_test, rf_predictions, average='weighted', zero_division=0)
        
        print(f"\n   Metrics:")
        print(f"   {'='*40}")
        print(f"   Accuracy:  {rf_accuracy*100:.2f}%")
        print(f"   Precision: {rf_precision*100:.2f}% (weighted)")
        print(f"   Recall:    {rf_recall*100:.2f}% (weighted)")
        print(f"   F1-Score:  {rf_f1*100:.2f}% (weighted)")
        
        print(f"\n   Per-Class Report:")
        print(f"   {'='*40}")
        print(classification_report(y_test, rf_predictions, target_names=class_names, zero_division=0))

    # -------------------------------------------------------------------------
    # Step 12: Explainability
    # -------------------------------------------------------------------------
    print(f"\n12. Explainability (Feature Importance)...")
    
    importances = rf_final.get_feature_importance()
    feature_names = extractor.get_feature_names()
    
    if importances is not None and feature_names is not None:
        indices = np.argsort(importances)[::-1]  # Sort descending
        
        print(f"   Which signal features mattered most?")
        print(f"   {'='*40}")
        print(f"   {'Rank':<5} | {'Feature':<20} | {'Importance':<10}")
        print(f"   {'-'*40}")
        
        for f in range(len(feature_names)):
            idx = indices[f]
            print(f"   {f+1:<5} | {feature_names[idx]:<20} | {importances[idx]*100:.1f}%")
            
        print(f"\n   Observation: The model relies mostly on '{feature_names[indices[0]]}'")
        print(f"   to distinguish between bird species.")

    print("\n" + "="*40)
    print("Pipeline Complete!")

if __name__ == "__main__":
    main()
