# Main source: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2022.1072801/full
import numpy as np
import scipy.signal
import scipy.stats
from feature_extraction.frequency_domain_extraction import BrainRythm
import pywt

SAMPLE_RATE = 250


def time_frequency_domain_feature_extraction(eeg_data):
    features = {}
    continous_wavelet_transform(eeg_data, features)
    return features


# MARK: Main computations


def discrete_wavelet_transform(data, dictionary, wavelet="db4", level=4, top_k=10):
    dictionary["discrete_wavelet"] = {}

    for i, channel in enumerate(data, 1):
        ch_key = f"EEG_Ch{i}"
        f = extract_dwt_features(channel, wavelet=wavelet, level=level)
        # REMOVE prints
        print(ch_key)
        print(f)


def continous_wavelet_transform(data, dictionary, epoch_duration=5):
    dictionary["continous_wavelet"] = {}
    samples_per_epoch = SAMPLE_RATE * epoch_duration
    for i, channel in enumerate(data, 1):
        ch_key = f"EEG_Ch{i}"
        dictionary["continous_wavelet"][ch_key] = {}
        # Step 1: Partition eeg signal into non-overlapping epochs of 5s duration
        num_epochs = len(channel) // samples_per_epoch
        epochs = np.array_split(channel[: num_epochs * samples_per_epoch], num_epochs)
        # Step 2: Compute CWT of each epoch, projecting them into the time frequency domain
        #         Use the Mexican hat mother wavelet function
        all_cwt_epochs = []
        for epoch in epochs:
            freqs = np.linspace(1, 30, 100)
            widths = hz_to_scale(freqs)
            cwt_matrix = scipy.signal.cwt(epoch, scipy.signal.morlet2, widths, w=5)
            all_cwt_epochs.append(np.abs(cwt_matrix))
        # Step 3: Average epochs from each channel to form signal averagte epoch
        avg_cwt = np.mean(np.array(all_cwt_epochs), axis=0)
        # Step 4: partition each epochs in five EEG rythms
        for rythm in BrainRythm:
            (low_f, high_f) = rythm.value
            freq_idxs = np.where((freqs >= low_f) & (freqs <= high_f))[0]
            if len(freq_idxs) == 0:
                continue
            band_cwt = avg_cwt[freq_idxs, :]
        # Step 5: From each EEG sub band, extract mean, std, skewness, kurtosis and entropy
        values = band_cwt.flatten()

        dictionary["continous_wavelet"][ch_key][rythm.name]["mean"] = np.mean(values)
        dictionary["continous_wavelet"][ch_key][rythm.name]["std"] = np.std(values)
        dictionary["continous_wavelet"][ch_key][rythm.name]["skewness"] = (
            scipy.stats.skew(values)
        )
        dictionary["continous_wavelet"][ch_key][rythm.name]["kurtosis"] = (
            scipy.stats.kurtosis(values)
        )

        power = values**2
        power /= np.sum(power) + 1e-8

        dictionary["continous_wavelet"][ch_key][rythm.name]["entropy"] = (
            scipy.stats.entropy(power)
        )


# MARK: Helpers


def hz_to_scale(freq):
    return SAMPLE_RATE / freq


def extract_dwt_features(channel_data, dictionary, wavelet="db4", level=4):
    coeffs = pywt.wavedec(channel_data, wavelet, level=level)
    features = []
    for c in coeffs:
        features.append(np.mean(c))
        features.append(np.std(c))
        features.append(scipy.stats.skew(c))
        features.append(scipy.stats.kurtosis(c))

    return np.array(features)


def fisher_score(features, labels):
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("Fisher score is only implemented for binary classification")

    c0_idx = labels == classes[0]
    c1_idx = labels == classes[1]

    mean0 = np.mean(features[c0_idx], axis=0)
    mean1 = np.mean(features[c1_idx], axis=0)

    var0 = np.var(features[c0_idx], axis=0)
    var1 = np.var(features[c1_idx], axis=0)

    # Between-class and within-class variance
    numerator = (mean0 - mean1) ** 2
    denominator = var0 + var1 + 1e-8  # avoid division by zero

    return numerator / denominator  # Fisher Score


def select_top_features(features, fisher_scores, top_k=10):
    sorted_indices = np.argsort(fisher_scores)[::-1]
    return features[:, sorted_indices[:top_k]]
