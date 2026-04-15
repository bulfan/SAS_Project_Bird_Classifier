# SAS_Project_Whale_Classifier

Project for the second try of SAS. This is a whale sound classifier.

## Overview

This project implements a whale sound classifier that:
1. Loads whale sound audio data
2. Preprocesses the audio (feature extraction)
3. Classifies whale species using a machine learning model

## Project Structure

```
SAS_Project_Bird_Classifier/
├── configs/                    # Configuration files
│   └── config.yaml            # Default training configuration
├── data/                       # Dataset storage
│   ├── raw/                   # Raw audio files
│   └── processed/             # Preprocessed features
├── notebooks/                  # Jupyter notebooks for exploration
├── src/                        # Source code
│   ├── data/                  # Data loading utilities
│   │   └── dataset.py         # Dataset class
│   ├── preprocessing/         # Audio preprocessing
│   │   └── audio_processor.py # Audio feature extraction
│   ├── models/                # Model definitions
│   │   └── classifier.py      # Bird species classifier
│   ├── training/              # Training utilities
│   │   └── trainer.py         # Training pipeline
│   └── utils/                 # Helper functions
│       └── helpers.py         # Utility functions
├── main.py                     # Main entry point
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Layout

The loader supports two input styles in `data/raw/`:

1. **Class subfolders** (recommended):
   - `data/raw/<species_name>/<clip>.mp3`
2. **Flat files**:
   - `data/raw/<species-name>-<clip-number>.mp3`
   - Example: `north-atlantic-right-whale-eubalaena-glacialis-1.mp3`
   - The trailing `-<number>` is treated as clip index, and the rest becomes the class name.

## Usage

```bash
python main.py
```

When you run `python main.py`, output folders are populated automatically:

- If `data/processed/train` already has `.npy` files, existing processed data is reused.
- If processed data is missing, preprocessing auto-runs and creates:
  - `data/processed/train/<species>/*.npy`
  - `data/processed/val/<species>/*.npy`
  - `data/processed/test/<species>/*.npy`
- Analysis plots are created under `outputs/` based on config (`analysis.*.output_dir`) when enabled.

## Development

1. Place your audio dataset in `data/raw/`
2. Implement the preprocessing pipeline in `src/preprocessing/`
3. Define your model architecture in `src/models/`
4. Train the model using `src/training/`


## Future steps

1. Load dataset (file paths + labels)
2. Split into train/val/test (just paths, no audio loaded yet)
3. Analyze training data only (exploration, spectral) → understand YOUR training data
4. Fit preprocessing on training data only (e.g., normalization stats)
5. Apply preprocessing to all sets (train, val, test) using training stats
6. Train model on training set
7. Validate/tune on validation set
8. Final evaluation on test set
