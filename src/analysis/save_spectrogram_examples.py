"""
Save spectrogram examples per class (no GUI display).
Usage: python -m src.analysis.save_spectrogram_examples --n 2
Writes per-file spectrogram PNGs and a combined grid per class under outputs/frequency_examples/<class>/
"""
from pathlib import Path
import os
import argparse
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from src.preprocessing.call_detection import detect_calls


def save_spectrogram(y, sr, out_path, n_fft=2048, hop_length=512, cmap='magma'):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann'))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(6, 3))
    times = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)
    freqs = np.linspace(0, sr/2, S_db.shape[0])
    plt.pcolormesh(times, freqs, S_db, shading='gouraud', cmap=cmap)
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, sr/2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def combine_images_horizontally(image_paths, out_path):
    images = [Image.open(p) for p in image_paths]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height), (255, 255, 255))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    new_im.save(out_path)


def main(data_dir='data/raw', out_base='outputs/frequency_examples', n_per_class=2, target_sr=22050, n_fft=2048, hop_length=512, calls_per_file=5):
    data_dir = Path(data_dir)
    out_base = Path(out_base)
    out_base.mkdir(parents=True, exist_ok=True)
    classes = [p for p in sorted(data_dir.iterdir()) if p.is_dir()]
    summary = {}
    for cls in classes:
        files = sorted([f for f in cls.iterdir() if f.suffix.lower() in ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.npy')])[:n_per_class]
        if not files:
            continue
        class_out = out_base / cls.name
        class_out.mkdir(parents=True, exist_ok=True)
        saved = []
        for f in files:
            try:
                if f.suffix.lower() == '.npy':
                    # processed numpy arrays (saved by preprocessing) are assumed
                    # to be 1-D sample arrays at `target_sr` (no sr stored)
                    y = np.load(str(f))
                    sr = target_sr
                else:
                    y, sr = librosa.load(str(f), sr=target_sr, mono=True)
                # detect calls and save up to `calls_per_file` chops per file
                segments = detect_calls(y, sr)
                if segments:
                    for i, (start_t, end_t, dur) in enumerate(segments[:calls_per_file], start=1):
                        s_idx = int(max(0, round(start_t * sr)))
                        e_idx = int(min(len(y), round(end_t * sr)))
                        y_seg = y[s_idx:e_idx]
                        out_file = class_out / f'spectrogram_{f.stem}_call{i}.png'
                        save_spectrogram(y_seg, sr, str(out_file), n_fft=n_fft, hop_length=hop_length)
                        saved.append(str(out_file))
                else:
                    # fallback: save full-file spectrogram if no calls detected
                    out_file = class_out / f'spectrogram_{f.stem}.png'
                    save_spectrogram(y, sr, str(out_file), n_fft=n_fft, hop_length=hop_length)
                    saved.append(str(out_file))
            except Exception as e:
                print(f'Failed to process {f}: {e}')
        # combine into a grid (horizontal)
        if saved:
            grid_path = class_out / 'spectrogram_grid.png'
            try:
                combine_images_horizontally(saved, str(grid_path))
            except Exception as e:
                print(f'Failed to create grid for {cls.name}: {e}')
        summary[cls.name] = {'saved': saved, 'grid': str(class_out / 'spectrogram_grid.png')}
    # print summary
    for k, v in summary.items():
        print(f"Class {k}: saved {len(v['saved'])} spectrograms, grid: {v['grid']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=2, help='Examples per class')
    parser.add_argument('--data', type=str, default='data/raw', help='Raw data dir')
    parser.add_argument('--out', type=str, default='outputs/frequency_examples', help='Output base dir')
    parser.add_argument('--sr', type=int, default=22050, help='Target sample rate (resample higher files down)')
    args = parser.parse_args()
    main(data_dir=args.data, out_base=args.out, n_per_class=args.n, target_sr=args.sr)
