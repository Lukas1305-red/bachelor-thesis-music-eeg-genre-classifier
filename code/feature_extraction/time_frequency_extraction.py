# Main source: https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2022.1072801/full
import numpy as np
import scipy.stats
import pywt

from code.feature_extraction.frequency_domain_extraction import BrainRythm
from code.feature_extraction.feature_extraction_utils import flatten_dict


SAMPLE_RATE = 250


def time_frequency_domain_feature_extraction(eeg_data):
    features = {}
    continous_wavelet_transform(eeg_data, features)
    discrete_wavelet_transform(eeg_data, features)
    return flatten_dict(features)


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
    wavelet = "morl"

    for i, channel in enumerate(data, 1):
        ch_key = f"EEG_Ch{i}"
        dictionary["continous_wavelet"][ch_key] = {}

        num_epochs = len(channel) // samples_per_epoch
        epochs = np.array_split(channel[: num_epochs * samples_per_epoch], num_epochs)

        all_cwt_epochs = []

        min_scale = 1
        max_scale = SAMPLE_RATE // 2
        scales = np.logspace(np.log10(min_scale), np.log10(max_scale), 100)

        for epoch in epochs:
            coef, freqs = pywt.cwt(
                epoch, scales, wavelet, sampling_period=1 / SAMPLE_RATE
            )
            all_cwt_epochs.append(np.abs(coef))

        avg_cwt = np.mean(np.array(all_cwt_epochs), axis=0)

        # Extract features per brain rhythm
        for rythm in BrainRythm:
            (low_f, high_f) = rythm.value
            freq_idxs = np.where((freqs >= low_f) & (freqs <= high_f))[0]
            if len(freq_idxs) == 0:
                continue

            band_cwt = avg_cwt[freq_idxs, :]
            values = band_cwt.flatten()

            dictionary["continous_wavelet"][ch_key][rythm.name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "skewness": scipy.stats.skew(values),
                "kurtosis": scipy.stats.kurtosis(values),
            }

            # Proper entropy calculation
            power = values**2
            power_norm = power / (np.sum(power) + 1e-8)
            dictionary["continous_wavelet"][ch_key][rythm.name]["entropy"] = (
                scipy.stats.entropy(power_norm)
            )


# MARK: Helpers


def hz_to_scale(freq):
    return SAMPLE_RATE / freq


def extract_dwt_features(channel_data, wavelet="db4", level=4):
    coeffs = pywt.wavedec(channel_data, wavelet, level=level)
    cA = coeffs[0]
    cDs = coeffs[1:]

    ret = {
        "approx": {},
        "detailed": {},
    }

    ret["approx"]["mean"] = np.mean(cA)
    ret["approx"]["std"] = np.std(cA)
    ret["approx"]["skew"] = scipy.stats.skew(cA)
    ret["approx"]["kurtosis"] = scipy.stats.kurtosis(cA)
    power = cA**2
    power_norm = power / (np.sum(power) + 1e-8)
    ret["approx"]["entropy"] = scipy.stats.entropy(power_norm)

    cD_all = np.concatenate(cDs)
    ret["detailed"]["mean"] = np.mean(cD_all)
    ret["detailed"]["std"] = np.std(cD_all)
    ret["detailed"]["skew"] = scipy.stats.skew(cD_all)
    ret["detailed"]["kurtosis"] = scipy.stats.kurtosis(cD_all)
    power = cD_all**2
    power_norm = power / (np.sum(power) + 1e-8)
    ret["detailed"]["entropy"] = scipy.stats.entropy(power_norm)

    return ret
