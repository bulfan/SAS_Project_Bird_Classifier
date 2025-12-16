def run_multiple_file_analysis():

    import os
    import glob
    import numpy as np
    from pydub import AudioSegment

    # --- Auto-select first samples from all class folders in data/raw/ ---
    data_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')
    class_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    MAX_FILES_PER_CLASS = 30
    file_paths = []
    for class_folder in class_folders:
        class_path = os.path.join(data_dir, class_folder)
        class_files = sorted(glob.glob(os.path.join(class_path, '*')))[:MAX_FILES_PER_CLASS]
        file_paths.extend(class_files)

    if not file_paths:
        print("No files found in any class folder. Exiting.")
        return []

    features = []
    n_fft = 4096
    threshold = 0.01
    from src.analysis.spectral_analysis import SpectralAnalysisPipeline
    from src.preprocessing.call_detection import detect_calls
    # helper to get a non-silent segment (reusable)
    def get_non_silent_segment(samples, segment_length=4096, threshold=0.01):
        total_len = len(samples)
        for start in range(0, max(1, total_len - segment_length), max(1, segment_length // 2)):
            seg = samples[start:start+segment_length]
            rms = np.sqrt(np.mean(seg**2))
            if rms > threshold:
                return seg
        return samples[-segment_length:]
    total_files = len(file_paths)
    for idx, file_path in enumerate(file_paths):
        print(f"Processing {idx+1}/{total_files}: {file_path}", flush=True)
        audio = AudioSegment.from_file(file_path)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels))
            samples = samples.mean(axis=1)
        sample_rate = audio.frame_rate
        segment = get_non_silent_segment(samples, segment_length=n_fft, threshold=threshold)
        sap = SpectralAnalysisPipeline({'analysis': {'spectral': {'n_fft': n_fft, 'window': 'hann'}}, 'data': {'sample_rate': sample_rate}})
        freqs, magnitudes = sap.compute_magnitude_spectrum(segment)
        magnitudes = magnitudes / np.max(magnitudes) if np.max(magnitudes) > 0 else magnitudes
        dom_idx = np.argmax(magnitudes)
        dom_freq = freqs[dom_idx]
        centroid = np.sum(freqs * magnitudes) / np.sum(magnitudes) if np.sum(magnitudes) > 0 else 0
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitudes) / np.sum(magnitudes)) if np.sum(magnitudes) > 0 else 0
        mean_freq = np.mean(freqs)
        median_freq = np.median(freqs)
        class_label = os.path.basename(os.path.dirname(file_path))
        # compute call length metrics using detect_calls (preprocessing)
        segments = detect_calls(samples, sample_rate,
                    top_db=40.0,
                    frame_length=4096,
                    hop_length=1024,
                    min_call_duration=0.05,
                    min_silence_duration=0.05,
                    low_freq=1000.0,
                    high_freq=8000.0)
        call_durations = [d for (_, _, d) in segments]
        call_total = float(np.sum(call_durations)) if call_durations else 0.0
        call_longest = float(np.max(call_durations)) if call_durations else 0.0
        num_calls = len(call_durations)
        features.append({
            'file': os.path.basename(file_path),
            'class': class_label,
            'dominant_freq': dom_freq,
            'centroid': centroid,
            'bandwidth': bandwidth,
            'call_total': call_total,
            'call_longest': call_longest,
            'num_calls': num_calls,
            'call_durations': call_durations,
            'mean_freq': mean_freq,
            'median_freq': median_freq
        })
        print(f" Finished {idx+1}/{total_files}: call_total={call_total:.3f}s, call_longest={call_longest:.3f}s", flush=True)
    # Aggregate call-length statistics per class
    from collections import defaultdict
    # aggregate per-call durations across files so stats are about individual calls
    class_call_durations = defaultdict(list)
    for f in features:
        class_call_durations[f['class']].extend(f.get('call_durations', []))

    print('\nCall length summary by class (per-call):')
    for cls, calls in sorted(class_call_durations.items()):
        arr = np.array(calls) if len(calls) > 0 else np.array([0.0])
        print(f" - {cls}: calls={len(calls)}, total={arr.sum():.3f}s, mean={arr.mean():.3f}s, median={np.median(arr):.3f}s, max={arr.max():.3f}s")

    # --- Per-class spectral aggregation (take up to 5 samples per class) ---
    from collections import defaultdict as _dd
    class_to_files = _dd(list)
    for fp in file_paths:
        cls = os.path.basename(os.path.dirname(fp))
        class_to_files[cls].append(fp)

    import matplotlib.pyplot as plt
    num_bins = 200
    print('\nPer-class spectral aggregation (up to 30 samples/class):')
    for cls in sorted(class_to_files.keys()):
        sample_files = class_to_files[cls][:30]
        if not sample_files:
            continue
        # determine maximum sample_rate among selected files
        sr_list = []
        per_file_bin_mags = []
        per_file_dom = []
        for sf in sample_files:
            seg_audio = AudioSegment.from_file(sf)
            y = np.array(seg_audio.get_array_of_samples(), dtype=np.float32)
            if seg_audio.channels > 1:
                y = y.reshape((-1, seg_audio.channels)).mean(axis=1)
            sr = seg_audio.frame_rate
            sr_list.append(sr)
        max_sr = max(sr_list)
        freq_max = max_sr / 2.0
        bins = np.linspace(0, freq_max, num_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        for sf in sample_files:
            seg_audio = AudioSegment.from_file(sf)
            y = np.array(seg_audio.get_array_of_samples(), dtype=np.float32)
            if seg_audio.channels > 1:
                y = y.reshape((-1, seg_audio.channels)).mean(axis=1)
            sr = seg_audio.frame_rate
            seg = get_non_silent_segment(y, segment_length=n_fft, threshold=threshold)
            sap = SpectralAnalysisPipeline({'analysis': {'spectral': {'n_fft': n_fft, 'window': 'hann'}}, 'data': {'sample_rate': sr}})
            freqs, mags = sap.compute_magnitude_spectrum(seg)
            mags = mags / (np.max(mags) if np.max(mags) > 0 else 1.0)
            # digitize into common bins (scale freqs to the common axis if sr != max_sr)
            freqs_scaled = freqs * (sr / max_sr)
            idxs = np.digitize(freqs_scaled, bins) - 1
            bin_mags = np.zeros(num_bins)
            bin_counts = np.zeros(num_bins)
            for i, m in enumerate(mags):
                idx = idxs[i]
                if 0 <= idx < num_bins:
                    bin_mags[idx] += m
                    bin_counts[idx] += 1
            bin_counts[bin_counts == 0] = 1
            bin_mags = bin_mags / bin_counts
            per_file_bin_mags.append(bin_mags)
            # per-file dominant frequency by area
            dom_idx = int(np.argmax(bin_mags))
            per_file_dom.append(bin_centers[dom_idx])

        # class-level aggregation
        class_bin = np.mean(np.vstack(per_file_bin_mags), axis=0)
        class_dom_idx = int(np.argmax(class_bin))
        class_dom_freq = bin_centers[class_dom_idx]
        print(f" - {cls}: sampled_files={len(sample_files)}, per-file-dominant-mean={np.mean(per_file_dom):.1f}Hz, class-dominant={class_dom_freq:.1f}Hz")

        # plot class-averaged binned spectrum
        plt.figure(figsize=(8, 3))
        plt.bar(bin_centers, class_bin, width=(bins[1]-bins[0]) * 0.9)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Mean normalized magnitude')
        plt.title(f'Class-avg spectrum (binned): {cls} — dom {class_dom_freq:.1f} Hz')
        plt.xlim(0, freq_max)
        plt.tight_layout()
        plt.show()

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

def run_logistic_regression(features, feature_names=None):
    from src.models.MultiLogRegression import MultiLogRegression
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    if feature_names is None:
        feature_names = ['centroid', 'bandwidth', 'dominant_freq', 'mean_freq', 'median_freq']
    y = np.array([f['class'] for f in features])
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            f1, f2 = feature_names[i], feature_names[j]
            X = np.array([[f[f1], f[f2]] for f in features])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            MultiLog = MultiLogRegression(max_iter=500)
            MultiLog.fit(X_train_scaled, y_train)
            y_pred = MultiLog.predict(X_test_scaled)
            accuracy = MultiLog.score(X_test_scaled, y_test)
            print(f"Features: {f1}, {f2} | MultiLogRegression test accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    import matplotlib.cm as cm
    print("Running Analysis")
    features = run_multiple_file_analysis()
    print("Plotting")
    # Example plots
    plot_feature_scatter(features, 'centroid', 'bandwidth', title='Centroid vs Bandwidth by Class')
    plot_feature_scatter(features, 'dominant_freq', 'centroid', title='Dominant Frequency vs Centroid by Class')
    # Logistic regression
    run_logistic_regression(features)
    unique_classes = sorted(set(f['class'] for f in features))
    colors = cm.get_cmap('tab10', len(unique_classes))
