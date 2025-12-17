def run_multiple_file_analysis():

    import os
    import glob
    import numpy as np
    from pydub import AudioSegment

    # --- Auto-select first samples from all class folders in data/processed (fallback to data/raw) ---
    raw_dir = os.path.join(os.path.dirname(__file__), '../../data/raw')
    proc_dir = os.path.join(os.path.dirname(__file__), '../../data/processed')
    class_folders = sorted([f for f in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, f))])
    MAX_FILES_PER_CLASS = 30
    # file_entries: list of dicts {'path': str, 'class': str, 'is_npy': bool}
    file_entries = []
    for class_folder in class_folders:
        # prefer .npy processed files if present
        proc_class_path = os.path.join(proc_dir, class_folder)
        if os.path.isdir(proc_class_path):
            proc_files = sorted(glob.glob(os.path.join(proc_class_path, '*.npy')))[:MAX_FILES_PER_CLASS]
            for pf in proc_files:
                file_entries.append({'path': pf, 'class': class_folder, 'is_npy': True})
            # if we found processed files, skip raw for this class
            if proc_files:
                continue
        # otherwise fall back to raw
        raw_class_path = os.path.join(raw_dir, class_folder)
        raw_files = sorted(glob.glob(os.path.join(raw_class_path, '*')))[:MAX_FILES_PER_CLASS]
        for rf in raw_files:
            file_entries.append({'path': rf, 'class': class_folder, 'is_npy': False})

    if not file_entries:
        print("No files found in any class folder. Exiting.")
        return []

    features = []
    per_call_features = []
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
    TARGET_SR = 10000
    total_files = len(file_entries)
    for idx, entry in enumerate(file_entries):
        file_path = entry['path']
        print(f"Processing {idx+1}/{total_files}: {file_path}", flush=True)
        if entry.get('is_npy'):
            # load preprocessed numpy array (assume TARGET_SR)
            samples = np.load(file_path)
            sample_rate = TARGET_SR
        else:
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
        # spectral centroid (weighted mean) and spectral standard deviation (sqrt weighted variance)
        centroid = np.sum(freqs * magnitudes) / np.sum(magnitudes) if np.sum(magnitudes) > 0 else 0
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitudes) / np.sum(magnitudes)) if np.sum(magnitudes) > 0 else 0
        # Unweighted mean of frequency bins is constant given n_fft/sample_rate; use centroid instead
        mean_freq = centroid
        # We'll expose SD as `sd_freq` (spectral standard deviation = bandwidth)
        sd_freq = float(bandwidth)
        class_label = os.path.basename(os.path.dirname(file_path))
        # compute call length metrics using detect_calls (preprocessing)
        segments = detect_calls(samples, sample_rate,
                    top_db=40.0,
                    frame_length=4096,
                    hop_length=1024,
                    min_call_duration=0.05,
                    min_silence_duration=0.05,
                low_freq=1000.0,
                high_freq=min(8000.0, sample_rate/2.0))
        call_durations = [d for (_, _, d) in segments]
        call_total = float(np.sum(call_durations)) if call_durations else 0.0
        call_count = int(len(call_durations))
        call_sd = float(np.std(call_durations)) if call_durations else 0.0

        # extract per-call spectral features (dominant, centroid, sd)
        call_dominants = []
        for (start_t, end_t, dur) in segments:
            s_idx = int(max(0, round(start_t * sample_rate)))
            e_idx = int(min(len(samples), round(end_t * sample_rate)))
            y_seg = samples[s_idx:e_idx]
            if len(y_seg) == 0:
                continue
            sap_call = SpectralAnalysisPipeline({'analysis': {'spectral': {'n_fft': n_fft, 'window': 'hann'}}, 'data': {'sample_rate': sample_rate}})
            freqs_c, mags_c = sap_call.compute_magnitude_spectrum(y_seg)
            mags_c = mags_c / (np.max(mags_c) if np.max(mags_c) > 0 else 1.0)
            dom_idx_c = int(np.argmax(mags_c))
            dom_freq_c = float(freqs_c[dom_idx_c])
            centroid_c = float(np.sum(freqs_c * mags_c) / np.sum(mags_c)) if np.sum(mags_c) > 0 else 0.0
            sd_c = float(np.sqrt(np.sum(((freqs_c - centroid_c) ** 2) * mags_c) / np.sum(mags_c))) if np.sum(mags_c) > 0 else 0.0
            per_call_features.append({
                'file': os.path.basename(file_path),
                'class': class_label,
                'start_t': float(start_t),
                'end_t': float(end_t),
                'duration': float(dur),
                'dominant_freq': dom_freq_c,
                'centroid': centroid_c,
                'sd_freq': sd_c
            })
            call_dominants.append(dom_freq_c)

        # set file-level dominant to mean of per-call dominants when available
        file_dominant = float(np.mean(call_dominants)) if call_dominants else float(dom_freq)

        # Keep dominant frequency, spectral centroid (as `centroid`), spectral SD, and call_total
        features.append({
            'file': os.path.basename(file_path),
            'class': class_label,
            'dominant_freq': file_dominant,
            'centroid': float(centroid),
            'sd_freq': sd_freq,
            'call_total': call_total,
            'call_count': call_count,
            'call_sd': call_sd,
            'mean_call_length': float(call_total / call_count) if call_count > 0 else 0.0,
            'call_durations': call_durations
        })
        print(f" Finished {idx+1}/{total_files}: dominant={file_dominant:.1f}Hz, centroid={centroid:.1f}Hz, sd={sd_freq:.2f}Hz, call_total={call_total:.3f}s, calls={call_count}, call_sd={call_sd:.3f}s", flush=True)
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

    # Compute class-level call-length standard deviation (per-class)
    class_call_stats = {}
    for cls, calls in class_call_durations.items():
        if calls:
            arr = np.array(calls)
            class_call_stats[cls] = {
                'count': int(len(arr)),
                'total': float(arr.sum()),
                'mean': float(arr.mean()),
                'median': float(np.median(arr)),
                'max': float(arr.max()),
                'sd': float(np.std(arr))
            }
        else:
            class_call_stats[cls] = {'count': 0, 'total': 0.0, 'mean': 0.0, 'median': 0.0, 'max': 0.0, 'sd': 0.0}

    # Attach class-level call SD to each per-file feature record for downstream use
    for f in features:
        cls = f.get('class')
        f['class_call_sd'] = class_call_stats.get(cls, {}).get('sd', 0.0)

    # --- Per-class spectral aggregation (take up to 5 samples per class) ---
    from collections import defaultdict as _dd
    class_to_files = _dd(list)
    for entry in file_entries:
        cls = entry['class']
        class_to_files[cls].append(entry)

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
            if sf.get('is_npy'):
                y = np.load(sf['path'])
                sr = TARGET_SR
            else:
                seg_audio = AudioSegment.from_file(sf['path'])
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
            if sf.get('is_npy'):
                y = np.load(sf['path'])
                sr = TARGET_SR
            else:
                seg_audio = AudioSegment.from_file(sf['path'])
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
        from pathlib import Path
        out_dir = Path('outputs/analysis/plots')
        out_dir.mkdir(parents=True, exist_ok=True)
        save_fp = out_dir / f'class_avg_spectrum_{cls}.png'
        plt.savefig(str(save_fp), dpi=150, bbox_inches='tight')
        plt.close()

    # Save per-file and per-call features CSVs for downstream analysis
    from pathlib import Path
    import csv
    out_dir = Path('outputs/analysis')
    out_dir.mkdir(parents=True, exist_ok=True)
    file_csv = out_dir / 'features_per_file.csv'
    call_csv = out_dir / 'features_per_call.csv'
    # write per-file
    if features:
        keys = ['file', 'class', 'dominant_freq', 'centroid', 'sd_freq', 'call_total', 'call_count', 'call_sd']
        with open(str(file_csv), 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for f in features:
                row = {k: f.get(k, '') for k in keys}
                writer.writerow(row)
    # write per-call
    if per_call_features:
        keys = ['file', 'class', 'start_t', 'end_t', 'duration', 'dominant_freq', 'centroid', 'sd_freq']
        with open(str(call_csv), 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            for c in per_call_features:
                writer.writerow({k: c.get(k, '') for k in keys})

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
        feature_names = ['dominant_freq', 'call_total']
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


def run_regression_evaluate(features, feature_names=None, test_size=0.2, random_state=42):
    """Train MultiLogRegression on given features and report overall and per-class accuracy."""
    from src.models.MultiLogRegression import MultiLogRegression
    import numpy as _np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import confusion_matrix, classification_report

    if feature_names is None:
        feature_names = ['dominant_freq', 'call_total']

    # build X, y
    y = _np.array([f['class'] for f in features])
    X = _np.array([[f.get(feature_names[0], 0.0), f.get(feature_names[1], 0.0)] for f in features], dtype=_np.float64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MultiLogRegression(max_iter=500)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    overall_acc = float(_np.mean(y_pred == y_test))
    print(f"Overall test accuracy: {overall_acc:.3f}")

    classes = sorted(list(set(y)))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print('\nPer-class accuracy (recall on test set):')
    for i, cls in enumerate(classes):
        true_count = int(cm[i].sum())
        correct = int(cm[i, i])
        acc = correct / true_count if true_count > 0 else 0.0
        print(f" - {cls}: accuracy={acc:.3f} ({correct}/{true_count})")

    print('\nClassification report:')
    print(classification_report(y_test, y_pred, labels=classes, zero_division=0))

if __name__ == "__main__":
    import matplotlib.cm as cm
    print("Running Analysis")
    features = run_multiple_file_analysis()
    print("Plotting")
    # Example plots
    #plot_feature_scatter(features, 'centroid', 'bandwidth', title='Centroid vs Bandwidth by Class')
    #plot_feature_scatter(features, 'dominant_freq', 'centroid', title='Dominant Frequency vs Centroid by Class')
    # Logistic regression evaluations on requested feature pairs
    print('\nEvaluation: dominant_freq vs mean_call_length')
    run_regression_evaluate(features, feature_names=['dominant_freq', 'mean_call_length'])
    print('\nEvaluation: mean_call_length vs call_sd')
    run_regression_evaluate(features, feature_names=['mean_call_length', 'call_sd'])
    print('\nEvaluation: dominant_freq vs sd_freq')
    run_regression_evaluate(features, feature_names=['dominant_freq', 'sd_freq'])
    unique_classes = sorted(set(f['class'] for f in features))
    colors = cm.get_cmap('tab10', len(unique_classes))
