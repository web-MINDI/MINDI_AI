import numpy as np
import librosa
import librosa.display
import parselmouth
from parselmouth.praat import call


def extract_features(wav_path, segment_duration=5.0):
    sound = parselmouth.Sound(wav_path)
    total_duration = sound.get_total_duration()
    results = []
    num_segments = int(np.floor(total_duration / segment_duration))

    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = start_time + segment_duration
        segment = sound.extract_part(from_time=start_time, to_time=end_time, preserve_times=False)
        pitch = segment.to_pitch()
        f0 = pitch.selected_array['frequency']
        f0_voiced = f0[f0 > 0]
        if len(f0_voiced) == 0: continue

        f0_mean = np.mean(f0_voiced)
        f0_std = np.std(f0_voiced)
        f0_max = np.max(f0_voiced)
        f0_min = np.min(f0_voiced)

        point_process = call(segment, "To PointProcess (periodic, cc)", 75, 500)
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([segment, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        apq5 = call([segment, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        harmonicity = call(segment, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)
        hnr_std = call(harmonicity, "Get standard deviation", 0, 0)
        hnr_max = call(harmonicity, "Get maximum", 0, 0, "Parabolic")
        hnr_min = call(harmonicity, "Get minimum", 0, 0, "Parabolic")

        segment_np = segment.values[0]
        sr = 16000

        mfcc = librosa.feature.mfcc(y=segment_np, sr=sr, n_mfcc=12)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)

        mfcc_stats = []
        for i in range(12):
            mfcc_stats += [
                np.mean(mfcc[i]), np.std(mfcc[i]),
                np.min(mfcc[i]), np.max(mfcc[i])
            ]

        delta_stats = []
        for i in range(12):
            delta_stats += [
                np.mean(delta_mfcc[i]), np.std(delta_mfcc[i]),
                np.min(delta_mfcc[i]), np.max(delta_mfcc[i])
            ]

        delta2_stats = []
        for i in range(12):
            delta2_stats += [
                np.mean(delta2_mfcc[i]), np.std(delta2_mfcc[i]),
                np.min(delta2_mfcc[i]), np.max(delta2_mfcc[i])
            ]

        features = [
                       f0_mean, f0_std, f0_max, f0_min,
                       jitter, ppq5, shimmer, apq5,
                       hnr, hnr_std, hnr_max, hnr_min
                   ] + mfcc_stats + delta_stats + delta2_stats

        results.append(features)


    return np.array(results)