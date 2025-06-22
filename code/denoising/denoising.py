import scipy.signal
import scipy.stats
import numpy as np
from sklearn.decomposition import FastICA

# *****THRESHOLD CONSTANTS*****

PTP_MAX = 100
KURTOSIS_MAX = 5
DELTA_MAX = 0.5
GAMMA_MAX = 0.2


# Source: https://swharden.com/blog/2020-09-23-signal-filtering-in-python/
def bandpass(data, edges, sample_rate, poles=5):
    sos = scipy.signal.butter(poles, edges, "bandpass", fs=sample_rate, output="sos")
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


def notch_filter(data, f0, sample_rate, Q=30):
    b, a = scipy.signal.iirnotch(f0, Q, fs=sample_rate)
    return scipy.signal.filtfilt(b, a, data)


def apply_z_normalization(song_data):
    mean = np.mean(song_data, axis=1, keepdims=True)
    std = np.std(song_data, axis=1, keepdims=True)
    return (song_data - mean) / std


def bandpower(frequencies, psd, band):
    band_freqs = np.logical_and(frequencies >= band[0], frequencies <= band[1])
    return np.trapezoid(psd[band_freqs], frequencies[band_freqs])


def component_rejected(component):
    # Peak-to-peak amplitude
    ptp = np.ptp(component)
    # Kurtosis
    kurt = scipy.stats.kurtosis(component)
    # Power in specific bands via PSD
    sample_rate = 250
    f, psd = scipy.signal.welch(component, fs=sample_rate, nperseg=512)
    total_power = np.trapezoid(f, psd)
    power_delta = bandpower(f, psd, (0.5, 4))
    power_gamma = bandpower(f, psd, (30, 45))

    if ptp > PTP_MAX:
        return True
    if kurt > KURTOSIS_MAX:
        return True
    if (power_delta / total_power) > DELTA_MAX:
        return True
    if (power_gamma / total_power) > GAMMA_MAX:
        return True
    return False


def artifact_removal_ica(data):
    data = data - np.mean(data, axis=1, keepdims=True)
    song_data_transposed = np.transpose(data)

    ica = FastICA(n_components=8, random_state=2025, max_iter=1000)
    sources = ica.fit_transform(song_data_transposed)
    mixing_matrix = ica.mixing_

    rejected_components = []

    # reject components when thresholds are met
    for i in range(sources.shape[1]):
        component = sources[:, i]
        if component_rejected(component):
            rejected_components.append(i)

    cleaned_sources = sources.copy()
    for i in rejected_components:
        cleaned_sources[:, i] = 0

    cleaned_data = np.transpose(np.dot(cleaned_sources, np.transpose(mixing_matrix)))
    return cleaned_data
