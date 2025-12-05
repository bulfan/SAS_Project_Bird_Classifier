# SAS_Project_Bird_Classifier

Project for the second try of SAS. This is a Bird sound classifier.

## Overview

This project implements a bird sound classifier that:
1. Loads bird sound audio data
2. Preprocesses the audio (feature extraction)
3. Classifies bird species using a machine learning model

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

## Usage

```bash
python main.py
```

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