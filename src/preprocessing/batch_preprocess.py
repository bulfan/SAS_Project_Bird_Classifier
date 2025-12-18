import numpy as np
import os
import glob
import argparse
from pathlib import Path
from pydub import AudioSegment
import librosa
from src.preprocessing.audio_processor import AudioProcessor

"""Batch preprocessing script to denoise raw audio files.

Saves processed numpy arrays under `data/processed/<class>/<file>.npy`.
"""

def run_batch(raw_dir: str, out_dir: str, target_sr: int = 22050, out_format: str = 'npy', threshold_db: float = -40, n_per_class: int = None, skip_spectral_gate: bool = False, skip_filters: bool = False, skip_noise_gate: bool = False):
    raw_dir = os.path.abspath(raw_dir)
    out_dir = os.path.abspath(out_dir)

    if not os.path.isdir(raw_dir):
        raise RuntimeError(f"Raw directory not found: {raw_dir}")

    ap = AudioProcessor(sample_rate=target_sr)
    processed_files = []
    total_samples = 0

    for cls in sorted(os.listdir(raw_dir)):
        cls_path = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        files = sorted(glob.glob(os.path.join(cls_path, '*')))
        if n_per_class is not None:
            files = files[:n_per_class]
        
        for fp in files:
            try:
                audio = AudioSegment.from_file(fp)
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                if audio.channels > 1:
                    samples = samples.reshape((-1, audio.channels)).mean(axis=1)
                total_samples += len(samples)
                
                # Denoise: mean normalization
                samples = samples - np.mean(samples)
                
                # Apply filters (optional)
                if not skip_filters:
                    try:
                        samples = ap.highpass_filter(samples, 500)
                        samples = ap.lowpass_filter(samples, 7000)
                    except Exception as e:
                        print(f"[ERROR] Filters failed for {fp}: {e}")
                
                # Apply noise gate (optional)
                if not skip_noise_gate:
                    try:
                        samples = ap.noise_gate(samples, threshold_db=threshold_db, attack_ms=5, release_ms=50, hold_ms=200, max_on_ms=2000)
                    except Exception as e:
                        print(f"[ERROR] Noise gate failed for {fp}: {e}")
                
                # Apply spectral gate (optional)
                if not skip_spectral_gate:
                    try:
                        samples = ap.spectral_gate(samples, percentile=10)
                    except Exception as e:
                        print(f"[ERROR] Spectral gate failed for {fp}: {e}")
                
                out_path = Path(out_dir) / cls
                out_path.mkdir(parents=True, exist_ok=True)
                save_fp = out_path / (Path(fp).stem + ".npy")
                np.save(save_fp, samples.astype(np.float32))
                processed_files.append(str(save_fp))
                print(f"Processed and saved: {save_fp}")
            except Exception as e:
                print(f"Failed to process {fp}: {e}")
    
    print(f"Processed {len(processed_files)} files with {total_samples} total samples. Output dir: {out_dir}")
    return processed_files

def _cli():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', default=str(Path(__file__).resolve().parents[2] / 'data' / 'raw'))
    p.add_argument('--out', default=str(Path(__file__).resolve().parents[2] / 'data' / 'processed'))
    p.add_argument('--sr', type=int, default=22050, help='Target sample rate (for AudioProcessor, not used for resampling)')
    p.add_argument('--format', choices=['npy', 'wav', 'mp3'], default='npy', help='Output format for processed files')
    p.add_argument('--threshold', type=float, default=-40, help='Threshold in dB for noise gate')
    p.add_argument('--n', type=int, default=None, help='Number of files per class to process (default: all)')
    p.add_argument('--skip-spectral-gate', action='store_true', help='Skip spectral gate for faster processing')
    p.add_argument('--skip-filters', action='store_true', help='Skip highpass/lowpass filters for faster processing')
    p.add_argument('--skip-noise-gate', action='store_true', help='Skip noise gate for faster processing')
    args = p.parse_args()
    run_batch(args.raw, args.out, target_sr=args.sr, out_format=args.format, threshold_db=args.threshold, n_per_class=args.n, skip_spectral_gate=args.skip_spectral_gate, skip_filters=args.skip_filters, skip_noise_gate=args.skip_noise_gate)

if __name__ == '__main__':
    _cli()
