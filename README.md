# Bird Sound Classifier

A machine learning project for classifying bird species from audio recordings.

## Project Structure

```
SAS_Project_Bird_Classifier/
├── main.py                         # Main entry point
├── configs/
│   └── config.yaml                 # Configuration file
├── data/
│   ├── raw/                        # Raw audio files (organized by species)
│   │   ├── aldfly/
│   │   ├── amecro/
│   │   └── ...
│   └── processed/                  # Preprocessed features (.npy files)
│       ├── train/
│       ├── val/
│       └── test/
├── src/
│   ├── data/
│   │   └── dataset.py              # Dataset loading and splitting
│   ├── preprocessing/
│   │   └── audio_processor.py      # Audio noise reduction (filters, spectral gate)
│   ├── features/
│   │   ├── feature_extractor.py    # MFCC and spectral feature extraction
│   │   └── scaler.py               # Feature normalization
│   ├── analysis/
│   │   ├── time_analysis.py        # Time-domain analysis
│   │   └── spectral_analysis.py    # Frequency-domain analysis
│   └── models/
│       ├── KNN.py                  # K-Nearest Neighbors classifier
│       └── random_forest.py        # Random Forest classifier
├── outputs/                        # Generated outputs (plots, logs)
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

All settings are controlled via `configs/config.yaml`. Key sections:

### Data
```yaml
data:
  raw_dir: "data/raw"           # Directory with raw audio files
  processed_dir: "data/processed"
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  seed: 42
```

### Preprocessing
```yaml
preprocessing:
  enabled: false                # Set to true to run preprocessing
  sample_rate: 22050
  highpass_cutoff: 1000         # Hz - removes low-frequency noise
  lowpass_cutoff: 8000          # Hz - removes high-frequency noise
  gate_threshold_db: -40        # Spectral gate threshold
```

### Analysis
```yaml
analysis:
  exploration:
    enabled: false              # Time-domain analysis
    run_mode: "train"           # Options: "single", "folder", "train", "dataset"
  spectral:
    enabled: false              # Frequency-domain analysis
    run_mode: "train"
```

### Model
```yaml
model:
  type: "both"                  # Options: "knn", "random_forest", "both"
  knn:
    k: 5
  random_forest:
    n_estimators: 100
    max_depth: null
```

## Usage

1. **Prepare data**: Place audio files in `data/raw/`, organized by species (one folder per class)

2. **Configure**: Edit `configs/config.yaml` to enable the desired pipelines:
   - Set `preprocessing.enabled: true` to preprocess audio
   - Set `analysis.exploration.enabled: true` for time-domain analysis
   - Set `analysis.spectral.enabled: true` for frequency-domain analysis

3. **Run**:
   ```bash
   python main.py
   ```

   Override config values from command line:
   ```bash
   python main.py preprocessing.enabled=true model.type=knn
   ```