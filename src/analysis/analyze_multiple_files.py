def run_multiple_file_analysis():
    """
    Main script for analysing multiple files
    The Analysis pipeline is as follows:
    [0] Preprocess (filter)
    [1] Load first N_CLASS_SAMPLES from each class 
    [2] Segment N_SEGMENTS_PER_CLASS
    [3] Extract 5 Time based and 5 Spectral Features
    [4] Export Spectrograms for further inspection 
    """
    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    from pathlib import Path
    from src.analysis.data_exploration import extract_all_features


    def save_segment_spectrogram(segment, sr, out_path, n_fft=4096, hop_length=512, cmap='magma', use_mel=True, n_mels=128):
        if use_mel:
            S = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
            S_db = librosa.power_to_db(S, ref=np.max)
            plt.figure(figsize=(6, 3))
            librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap=cmap)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel Spectrogram')
        else:
            S = np.abs(librosa.stft(segment, n_fft=n_fft, hop_length=hop_length, window='hann'))
            S_db = librosa.amplitude_to_db(S, ref=np.max)
            plt.figure(figsize=(6, 3))
            librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='linear', cmap=cmap)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Spectrogram')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()

    import os
    import glob
    import numpy as np
    from pydub import AudioSegment

    # --- Auto-select first samples from all class folders in data/processed (fallback to data/raw) ---
    raw_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')
    proc_dir = os.path.join(os.path.dirname(__file__), '../../data/processed')
    class_folders = sorted([f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f))])
    N_CLASS_SAMPLES = 37
    # file_entries: list of dicts {'path': str, 'class': str, 'is_npy': bool}
    file_entries = []
    for class_folder in class_folders:
        # prefer .npy processed files if present
        proc_class_path = os.path.join(proc_dir, class_folder)
        if os.path.isdir(proc_class_path):
            proc_files = sorted(glob.glob(os.path.join(proc_class_path, '*.npy')))[:N_CLASS_SAMPLES]
            for pf in proc_files:
                file_entries.append({'path': pf, 'class': class_folder, 'is_npy': True})
            # if we found processed files, skip raw for this class
            if proc_files:
                continue
        # otherwise fall back to raw
        raw_class_path = os.path.join(raw_dir, class_folder)
        raw_files = sorted(glob.glob(os.path.join(raw_class_path, '*')))[:N_CLASS_SAMPLES]
        for rf in raw_files:
            file_entries.append({'path': rf, 'class': class_folder, 'is_npy': False})

    if not file_entries:
        print("No files found in any class folder. Exiting.")
        return []

    from src.analysis.data_exploration import extract_all_features
    features = []
    n_fft = 4096*16
    threshold = 0.01
    N_SEGMENTS_PER_CLASS = 50
    TARGET_SR = 22050

    # Helper to extract all non-overlapping, non-silent segments from a sample array
    def extract_non_silent_segments(samples, segment_length=4096, threshold=0.01):
        total_len = len(samples)
        segments = []
        for start in range(0, total_len - segment_length + 1, segment_length):
            seg = samples[start:start+segment_length]
            rms = np.sqrt(np.mean(seg**2))
            if rms > threshold:
                segments.append(seg)
        return segments

    from collections import defaultdict
    class_to_files = defaultdict(list)
    for entry in file_entries:
        class_to_files[entry['class']].append(entry)

    class_to_segments = {}
    for cls, entries in class_to_files.items():
        segments = []
        for entry in entries:
            if entry.get('is_npy'):
                samples = np.load(entry['path'])
                sample_rate = TARGET_SR
            else:
                audio = AudioSegment.from_file(entry['path'])
                samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
                if audio.channels > 1:
                    samples = samples.reshape((-1, audio.channels))
                    samples = samples.mean(axis=1)
                sample_rate = audio.frame_rate
            segs = extract_non_silent_segments(samples, segment_length=n_fft, threshold=threshold)
            segments.extend(segs)
            if len(segments) >= N_SEGMENTS_PER_CLASS:
                break
        if len(segments) < N_SEGMENTS_PER_CLASS:
            raise ValueError(f"Class '{cls}' has only {len(segments)} non-silent segments, fewer than required {N_SEGMENTS_PER_CLASS}.")
        class_to_segments[cls] = segments[:N_SEGMENTS_PER_CLASS]

    # Now, extract features for each segment, ensuring class balance
    for cls, seglist in class_to_segments.items():
        for i, segment in enumerate(seglist):
            # Add normalization here
            segment = segment / np.max(np.abs(segment)) if np.max(np.abs(segment)) > 0 else segment
            feats = extract_all_features(segment, TARGET_SR)
            feats['file'] = f'{cls}_segment{i+1}'
            feats['class'] = cls
            features.append(feats)
            print(f"Class {cls} segment {i+1}/{N_SEGMENTS_PER_CLASS}: features extracted.", flush=True)

    # Save per-file features CSV for downstream analysis
    from pathlib import Path
    import csv
    out_dir = Path('outputs/analysis')
    out_dir.mkdir(parents=True, exist_ok=True)
    file_csv = out_dir / 'features_per_file.csv'
    if features:
        keys = list(features[0].keys())
        with open(str(file_csv), 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for f in features:
                row = {k: f.get(k, '') for k in keys}
                writer.writerow(row)

    return features

def plot_feature_scatter(features, x_feature, y_feature, title=None):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    unique_classes = sorted(set(f['class'] for f in features))
    colors = cm.get_cmap('tab10', len(unique_classes))
    class_to_color = {cls: colors(i) for i, cls in enumerate(unique_classes)}
    plt.figure(figsize=(8, 6))
    for feat in features:
        plt.scatter(feat[x_feature], feat[y_feature], color=class_to_color[feat['class']], label=feat['class'])
    # Only show one legend entry per class
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.xlabel(f'{x_feature.replace("_", " ").title()}')
    plt.ylabel(f'{y_feature.replace("_", " ").title()}')
    plt.title(title or f'{x_feature} vs {y_feature} by Class')
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Running Analysis")
    features = run_multiple_file_analysis()
    print("\nSummary of Extracted Features (per class):")
    import numpy as np
    from collections import defaultdict
    # List of desired features
    time_features = ['rms', 'std', 'crest_factor', 'avg_amplitude', 'min_amplitude', 'max_amplitude']
    spectral_features = ['avg_power', 'max_frequency', 'min_frequency', 'min_magnitude', 'max_magnitude']
    class_to_feats = defaultdict(list)
    for f in features:
        class_to_feats[f['class']].append(f)
    for cls, feats in class_to_feats.items():
        print(f"\nClass: {cls}")
        for feat_name in time_features:
            vals = [f[feat_name] for f in feats if feat_name in f]
            print(f"  {feat_name}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")
        for feat_name in spectral_features:
            vals = [f[feat_name] for f in feats if feat_name in f]
            print(f"  {feat_name}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")
