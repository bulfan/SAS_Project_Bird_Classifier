"""Batch preprocessing script to denoise raw audio files.

Applies FIR filter bank and noise gates to reduce environmental noise.
Saves processed numpy arrays under `data/processed/<class>/<file>.npy`.

Usage:
    # From command line
    python -m src.preprocessing.batch_preprocess --raw data/raw --out data/processed
    
    # With Hydra config
    python -m src.preprocessing.batch_preprocess --config-path=../../configs --config-name=config
    
    # On Habrok (with multiple workers)
    python -m src.preprocessing.batch_preprocess --workers 8
"""

# =============================================================================
# Imports
# =============================================================================

import numpy as np
import os
import glob
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from src.preprocessing.audio_processor import AudioProcessor, LIBROSA_AVAILABLE, PYDUB_AVAILABLE


# =============================================================================
# Single File Processing (for multiprocessing)
# =============================================================================

def _process_single_file(args: Tuple[str, str, dict]) -> Tuple[str, bool, str]:
    """
    Process a single audio file. Designed to be called by multiprocessing.
    
    Args:
        args: Tuple of (input_path, output_path, processor_config)
    
    Returns:
        Tuple of (output_path, success, error_message)
    """
    input_path, output_path, config = args
    
    try:
        # Create processor with config
        ap = AudioProcessor(
            sample_rate=config.get('sample_rate', 22050),
            highpass_cutoff=config.get('highpass_cutoff', 500),
            lowpass_cutoff=config.get('lowpass_cutoff', 7000),
            gate_threshold_db=config.get('gate_threshold_db', -40)
        )
        
        # Load audio
        samples, sr = ap.load_audio(input_path)
        
        # Apply full processing pipeline
        samples = ap.process(
            samples,
            skip_filters=config.get('skip_filters', False),
            skip_noise_gate=config.get('skip_noise_gate', False),
            skip_spectral_gate=config.get('skip_spectral_gate', False)
        )
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save processed audio
        np.save(output_path, samples.astype(np.float32))
        
        return (output_path, True, "")
    
    except Exception as e:
        return (input_path, False, str(e))


# =============================================================================
# Batch Processing
# =============================================================================

def run_batch(raw_dir: str, 
              out_dir: str,
              cfg: Optional[Any] = None,
              n_per_class: Optional[int] = None,
              n_workers: int = 0) -> List[str]:
    """
    Batch preprocess audio files with FIR filter bank for noise reduction.
    
    Supports parallel processing for Habrok/HPC clusters.
    
    Pipeline:
    1. Load audio file
    2. Remove DC offset
    3. Apply highpass filter (remove low-freq rumble)
    4. Apply lowpass filter (remove high-freq hiss)  
    5. Apply noise gate (attenuate quiet periods)
    6. Apply spectral gate (remove broadband noise)
    7. Save as .npy file
    
    Args:
        raw_dir: Input directory with class subdirectories.
        out_dir: Output directory for processed files.
        cfg: Optional config object (from Hydra). If provided, uses config values.
        n_per_class: Number of files per class to process (None = all).
        n_workers: Number of parallel workers (0 = sequential processing).
    
    Returns:
        List of successfully processed file paths.
    """
    raw_dir = os.path.abspath(raw_dir)
    out_dir = os.path.abspath(out_dir)

    if not os.path.isdir(raw_dir):
        raise RuntimeError(f"Raw directory not found: {raw_dir}")

    # Build config dict from cfg object or defaults
    if cfg is not None:
        config = {
            'sample_rate': getattr(cfg, 'sample_rate', 22050),
            'highpass_cutoff': getattr(cfg, 'highpass_cutoff', 500),
            'lowpass_cutoff': getattr(cfg, 'lowpass_cutoff', 7000),
            'gate_threshold_db': getattr(cfg, 'gate_threshold_db', -40),
            'skip_filters': getattr(cfg, 'skip_filters', False),
            'skip_noise_gate': getattr(cfg, 'skip_noise_gate', False),
            'skip_spectral_gate': getattr(cfg, 'skip_spectral_gate', False),
        }
        n_workers = getattr(cfg, 'n_workers', n_workers)
    else:
        config = {
            'sample_rate': 22050,
            'highpass_cutoff': 500,
            'lowpass_cutoff': 7000,
            'gate_threshold_db': -40,
            'skip_filters': False,
            'skip_noise_gate': False,
            'skip_spectral_gate': False,
        }

    # Collect all files to process
    tasks = []
    for cls in sorted(os.listdir(raw_dir)):
        cls_path = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        
        # Find audio files (mp3, wav, flac, ogg)
        files = []
        for ext in ['*.mp3', '*.wav', '*.flac', '*.ogg']:
            files.extend(glob.glob(os.path.join(cls_path, ext)))
        files = sorted(files)
        
        if n_per_class is not None:
            files = files[:n_per_class]
        
        for fp in files:
            out_path = os.path.join(out_dir, cls, Path(fp).stem + ".npy")
            tasks.append((fp, out_path, config))
    
    print(f"Processing {len(tasks)} files...")
    print(f"  Input:  {raw_dir}")
    print(f"  Output: {out_dir}")
    print(f"  Workers: {n_workers if n_workers > 0 else 'sequential'}")
    print(f"  Config: sr={config['sample_rate']}, hp={config['highpass_cutoff']}Hz, "
          f"lp={config['lowpass_cutoff']}Hz, threshold={config['gate_threshold_db']}dB")
    
    processed_files = []
    failed_files = []
    
    if n_workers > 0:
        # Parallel processing (for Habrok/HPC)
        # Use spawn method for better compatibility with CUDA
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = {executor.submit(_process_single_file, task): task for task in tasks}
            
            for i, future in enumerate(as_completed(futures)):
                output_path, success, error = future.result()
                if success:
                    processed_files.append(output_path)
                    print(f"  [{i+1}/{len(tasks)}] ✓ {Path(output_path).name}")
                else:
                    failed_files.append((output_path, error))
                    print(f"  [{i+1}/{len(tasks)}] ✗ {Path(output_path).name}: {error}")
    else:
        # Sequential processing
        for i, task in enumerate(tasks):
            output_path, success, error = _process_single_file(task)
            if success:
                processed_files.append(output_path)
                print(f"  [{i+1}/{len(tasks)}] ✓ {Path(output_path).name}")
            else:
                failed_files.append((output_path, error))
                print(f"  [{i+1}/{len(tasks)}] ✗ {Path(output_path).name}: {error}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Preprocessing complete!")
    print(f"  Successful: {len(processed_files)}/{len(tasks)}")
    if failed_files:
        print(f"  Failed: {len(failed_files)}")
        for fp, err in failed_files[:5]:  # Show first 5 errors
            print(f"    - {Path(fp).name}: {err}")
        if len(failed_files) > 5:
            print(f"    ... and {len(failed_files) - 5} more")
    print(f"  Output directory: {out_dir}")
    print(f"{'='*60}")
    
    return processed_files


# =============================================================================
# Command Line Interface
# =============================================================================

def _cli():
    """Command line interface for batch preprocessing."""
    p = argparse.ArgumentParser(
        description="Preprocess audio files with noise reduction filter bank.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Paths
    p.add_argument('--raw', type=str, 
                   default=str(Path(__file__).resolve().parents[2] / 'data' / 'raw'),
                   help='Input directory with class subdirectories')
    p.add_argument('--out', type=str,
                   default=str(Path(__file__).resolve().parents[2] / 'data' / 'processed'),
                   help='Output directory for processed .npy files')
    
    # Processing parameters
    p.add_argument('--sr', type=int, default=22050,
                   help='Target sample rate')
    p.add_argument('--highpass', type=float, default=500,
                   help='Highpass filter cutoff frequency (Hz)')
    p.add_argument('--lowpass', type=float, default=7000,
                   help='Lowpass filter cutoff frequency (Hz)')
    p.add_argument('--threshold', type=float, default=-40,
                   help='Gate threshold in dB')
    
    # Skip options
    p.add_argument('--skip-filters', action='store_true',
                   help='Skip highpass/lowpass filters')
    p.add_argument('--skip-noise-gate', action='store_true',
                   help='Skip time-domain noise gate')
    p.add_argument('--skip-spectral-gate', action='store_true',
                   help='Skip frequency-domain spectral gate')
    
    # Batch options
    p.add_argument('--n', type=int, default=None,
                   help='Number of files per class (None = all)')
    p.add_argument('--workers', type=int, default=0,
                   help='Number of parallel workers (0 = sequential)')
    
    args = p.parse_args()
    
    # Build config dict from CLI args
    config = {
        'sample_rate': args.sr,
        'highpass_cutoff': args.highpass,
        'lowpass_cutoff': args.lowpass,
        'gate_threshold_db': args.threshold,
        'skip_filters': args.skip_filters,
        'skip_noise_gate': args.skip_noise_gate,
        'skip_spectral_gate': args.skip_spectral_gate,
    }
    
    # Create a simple namespace to act as config
    class Config:
        pass
    cfg = Config()
    for k, v in config.items():
        setattr(cfg, k, v)
    cfg.n_workers = args.workers
    
    run_batch(args.raw, args.out, cfg=cfg, n_per_class=args.n, n_workers=args.workers)


if __name__ == '__main__':
    _cli()