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
    wavelet = "morl"  # Morlet wavelet

    for i, channel in enumerate(data, 1):
        ch_key = f"EEG_Ch{i}"
        dictionary["continous_wavelet"][ch_key] = {}

        # Step 1: Partition signal into non-overlapping epochs
        num_epochs = len(channel) // samples_per_epoch
        epochs = np.array_split(channel[: num_epochs * samples_per_epoch], num_epochs)

        # Step 2: Compute CWT
        all_cwt_epochs = []
        for epoch in epochs:
            freqs = np.linspace(1, 30, 100)
            scales = pywt.scale2frequency(wavelet, 1) * SAMPLE_RATE / freqs
            coef, _ = pywt.cwt(epoch, scales, wavelet, sampling_period=1 / SAMPLE_RATE)
            all_cwt_epochs.append(np.abs(coef))

        # Step 3: Average across epochs
        avg_cwt = np.mean(np.array(all_cwt_epochs), axis=0)

        # Step 4: Partition into brain rhythm bands
        for rythm in BrainRythm:
            (low_f, high_f) = rythm.value
            freq_idxs = np.where((freqs >= low_f) & (freqs <= high_f))[0]
            if len(freq_idxs) == 0:
                continue
            band_cwt = avg_cwt[freq_idxs, :]

            # Step 5: Extract statistical features
            values = band_cwt.flatten()
            dictionary["continous_wavelet"][ch_key][rythm.name] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "skewness": scipy.stats.skew(values),
                "kurtosis": scipy.stats.kurtosis(values),
            }

            power = values**2
            power /= np.sum(power) + 1e-8
            dictionary["continous_wavelet"][ch_key][rythm.name]["entropy"] = (
                scipy.stats.entropy(power)
            )

    return dictionary


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
    ret["approx"]["entropy"] = scipy.stats.entropy(cA)

    cD_all = np.concatenate(cDs)
    ret["detailed"]["mean"] = np.mean(cD_all)
    ret["detailed"]["std"] = np.std(cD_all)
    ret["detailed"]["skew"] = scipy.stats.skew(cD_all)
    ret["detailed"]["kurtosis"] = scipy.stats.kurtosis(cD_all)
    ret["detailed"]["kurtosis"] = scipy.stats.entropy(cD_all)

    return ret
