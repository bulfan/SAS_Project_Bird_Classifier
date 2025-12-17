import numpy as np
from src.preprocessing.audio_processor import AudioProcessor
def apply_lowpass_to_preprocessed(processed_dir: str, cutoff_freq: float = 4000.0, sample_rate: int = 10000):
    """
    Apply a low pass filter to all .npy files in processed_dir/<class>/*.npy.
    Overwrites each file with the filtered version.
    """
    processed_dir = os.path.abspath(processed_dir)
    ap = AudioProcessor(sample_rate=sample_rate)
    for cls in sorted(os.listdir(processed_dir)):
        cls_path = os.path.join(processed_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        npy_files = sorted(glob.glob(os.path.join(cls_path, '*.npy')))
        for npy_fp in npy_files:
            try:
                arr = np.load(npy_fp)
                arr_filt = ap.lowpass_filter(arr, cutoff_freq)
                np.save(npy_fp, arr_filt.astype(np.float32))
                print(f"Lowpass filtered: {npy_fp}")
            except Exception as e:
                print(f"Failed to filter {npy_fp}: {e}")
"""Batch preprocessing script to resample and denoise raw audio files.

Saves processed numpy arrays under `data/processed/<class>/<file>.npy`.
"""
import os
import glob
import argparse
from pathlib import Path




def run_batch(raw_dir: str, out_dir: str, target_sr: int = 10000, out_format: str = 'npy'):
    raw_dir = os.path.abspath(raw_dir)
    out_dir = os.path.abspath(out_dir)

    if not os.path.isdir(raw_dir):
        raise RuntimeError(f"Raw directory not found: {raw_dir}")

    ap = AudioProcessor(sample_rate=target_sr)
    processed_files = []
    for cls in sorted(os.listdir(raw_dir)):
        cls_path = os.path.join(raw_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        for fp in sorted(glob.glob(os.path.join(cls_path, '*'))):
            try:
                from pydub import AudioSegment
                import numpy as _np
                audio = AudioSegment.from_file(fp)
                samples = _np.array(audio.get_array_of_samples(), dtype=_np.float32)
                if audio.channels > 1:
                    samples = samples.reshape((-1, audio.channels)).mean(axis=1)
                # Denoise: mean normalization
                samples = samples - _np.mean(samples)
                print(f"[DEBUG] Before filter: {fp}, samples type: {type(samples)}, shape: {getattr(samples, 'shape', None)}")
                try:
                    filtered = ap.bandpass_filter(samples, 250, ap.sample_rate/2-1)
                    print(f"[DEBUG] After filter: {fp}, filtered type: {type(filtered)}, shape: {getattr(filtered, 'shape', None)}")
                    if isinstance(filtered, _np.ndarray) and filtered.shape == samples.shape:
                        samples = filtered
                    else:
                        print(f"Warning: bandpass_filter failed for {fp}, using unfiltered samples.")
                except Exception as e:
                    print(f"[ERROR] Exception in bandpass_filter for {fp}: {e}")
                # Resample if needed
                sr_out = audio.frame_rate
                if sr_out != target_sr:
                    try:
                        import librosa
                        samples = librosa.resample(samples, orig_sr=sr_out, target_sr=target_sr)
                    except Exception:
                        pass
                out_path = Path(out_dir) / cls
                out_path.mkdir(parents=True, exist_ok=True)
                save_fp = out_path / (Path(fp).stem + ".npy")
                _np.save(save_fp, _np.asarray(samples, dtype=_np.float32))
                processed_files.append(str(save_fp))
                print(f"Processed and saved: {save_fp}")
            except Exception as e:
                print(f"Failed to process {fp}: {e}")
    print(f"Processed {len(processed_files)} files. Output dir: {out_dir}")
    return processed_files


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument('--raw', default=str(Path(__file__).resolve().parents[2] / 'data' / 'raw'))
    p.add_argument('--out', default=str(Path(__file__).resolve().parents[2] / 'data' / 'processed'))
    p.add_argument('--sr', type=int, default=10000)
    p.add_argument('--format', choices=['npy', 'wav', 'mp3'], default='npy', help='Output format for processed files')
    args = p.parse_args()
    run_batch(args.raw, args.out, target_sr=args.sr, out_format=args.format)


if __name__ == '__main__':
    _cli()
    # After batch processing, apply lowpass filter to all processed .npy files
    processed_dir = str(Path(__file__).resolve().parents[2] / 'data' / 'processed')
    apply_lowpass_to_preprocessed(processed_dir)
