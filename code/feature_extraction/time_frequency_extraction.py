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
    discrete_wavelet_transform(eeg_data, features)
    return features


# MARK: Main computations


def discrete_wavelet_transform(data, dictionary, wavelet="db4", level=4, top_k=10):
    dictionary["discrete_wavelet"] = {}

    for i, channel in enumerate(data, 1):
        ch_key = f"EEG_Ch{i}"
        feature_dict = extract_dwt_features(channel, wavelet=wavelet, level=level)
        dictionary["discrete_wavelet"][ch_key] = feature_dict


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
    (cA, cD) = pywt.wavedec(channel_data, wavelet, level=level)
    ret = {}
    ret["approx"]["mean"] = np.mean(cA)
    ret["approx"]["std"] = np.std(cA)
    ret["approx"]["skew"] = scipy.stats.skew(cA)
    ret["approx"]["kurtosis"] = scipy.stats.kurtosis(cA)
    ret["approx"]["entropy"] = scipy.stats.entropy(cA)

    ret["detailed"]["mean"] = np.mean(cD)
    ret["detailed"]["std"] = np.std(cD)
    ret["detailed"]["skew"] = scipy.stats.skew(cD)
    ret["detailed"]["kurtosis"] = scipy.stats.kurtosis(cD)
    ret["detailed"]["kurtosis"] = scipy.stats.entropy(cD)

    return ret
