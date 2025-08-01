import scipy.signal
from enum import Enum
import numpy as np

from code.feature_extraction.feature_extraction_utils import flatten_dict

SAMPLING_RATE = 250


class BrainRythm(Enum):
    ALPHA = (8, 13)
    BETA = (13, 30)
    THETA = (4, 8)
    DELTA = (0.5, 4)


def frequency_domain_feature_extraction(eeg_data):
    frequency_features = {}
    power_spectral_density_welch(eeg_data, 2, frequency_features)
    # power_spectral_density_fft(eeg_data, 2, frequency_features)
    # periodogram(eeg_data, frequency_features)
    return flatten_dict(frequency_features)


def extract_brain_rythm(f, psd, rythm: BrainRythm):
    borders = rythm.value
    idx = np.logical_and(f >= borders[0], f <= borders[1])
    power = np.trapezoid(psd[idx], f[idx])
    return power


def get_peak_frequency(f, psd, rythm: BrainRythm):
    borders = rythm.value
    idx = np.logical_and(f >= borders[0], f <= borders[1])
    return f[idx][np.argmax(psd[idx])]


def extract_frequency_features(f, psd, key, channel_idx, dictionary):
    channel_key = f"EEG_Ch{channel_idx}"
    if key not in dictionary:
        dictionary[key] = {}

    total_power = np.trapezoid(psd, f)
    psd_norm = psd / total_power
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    mean_freq = np.trapz(f * psd, f) / total_power
    # rms_bandwith = np.sqrt(np.trapezoid(((f - mean_freq) ** 2) * psd, f) / total_power)

    dictionary[key][channel_key] = {}
    # dictionary[key][channel_key]["f_bins"] = f
    # dictionary[key][channel_key]["power"] = psd
    dictionary[key][channel_key]["spectral_entropy"] = spectral_entropy
    dictionary[key][channel_key]["mean_freq"] = mean_freq
    # dictionary[key][channel_key]["rms_bandwidth"] = rms_bandwith

    rythm_power = {}

    for rythm in BrainRythm:
        power = extract_brain_rythm(f, psd, rythm=rythm)
        peak = get_peak_frequency(f, psd, rythm=rythm)
        # rel_power = power / total_power
        dictionary[key][channel_key][f"{rythm.name}"] = power
        # dictionary[key][channel_key][f"{rythm.name}_relative"] = rel_power
        dictionary[key][channel_key][f"{rythm.name}_peak"] = peak
        rythm_power[f"{rythm.name}"] = power

    # rythm ratios
    dictionary[key][channel_key]["alpha_beta_ratio"] = rythm_power[
        f"{BrainRythm.ALPHA.name}"
    ] / (rythm_power[f"{BrainRythm.BETA.name}"] + 1e-10)
    dictionary[key][channel_key]["theta_alpha_ratio"] = (
        rythm_power[f"{BrainRythm.THETA.name}"]
        / rythm_power[f"{BrainRythm.ALPHA.name}"]
    )
    dictionary[key][channel_key]["theta_beta_ratio"] = (
        rythm_power[f"{BrainRythm.THETA.name}"] / rythm_power[f"{BrainRythm.BETA.name}"]
    )


def power_spectral_density_welch(data, window_size, dictionary):
    for i, channel in enumerate(data, 1):
        f, psd_welch = scipy.signal.welch(
            channel, fs=SAMPLING_RATE, nperseg=SAMPLING_RATE * window_size
        )
        extract_frequency_features(f, psd_welch, "psd_welch", i, dictionary)


def power_spectral_density_fft(data, window_size, dictionary):
    for i, channel in enumerate(data, 1):
        n_samples = SAMPLING_RATE * window_size
        segment = channel[:n_samples]
        fft_vals = np.fft.fft(segment)
        fft_freqs = np.fft.fftfreq(n_samples, d=1 / SAMPLING_RATE)

        pos_mask = fft_freqs >= 0
        freqs = fft_freqs[pos_mask]
        psd = np.abs(fft_vals[pos_mask]) ** 2 / (n_samples * SAMPLING_RATE)

        extract_frequency_features(freqs, psd, "psd_fft", i, dictionary)


def periodogram(eeg_data, dictionary):
    dictionary["periodogram"] = {}
    for i, channel in enumerate(eeg_data, 1):
        freq, psd = scipy.signal.periodogram(
            channel, SAMPLING_RATE, return_onesided=True
        )
        extract_frequency_features(freq, psd, "periodogram", i, dictionary)
