import scipy.signal
import scipy.stats
import numpy as np

# Source: https://swharden.com/blog/2020-09-23-signal-filtering-in-python/


def bandpass(data, edges, sample_rate, poles=5):
    sos = scipy.signal.butter(poles, edges, "bandpass", fs=sample_rate, output="sos")
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


def notch_filter(data, f0, sample_rate, Q=30):
    b, a = scipy.signal.iirnotch(f0, Q, fs=sample_rate)
    return scipy.signal.filtfilt(b, a, data)


def artifact_removal_ica(song_data):
    # song_data has the shape (8. 7000ish)
    # center data
    song_data = song_data - np.mean(song_data, axis=1, keepdims=True)
    # song_data_transposed = np.transpose(song_data)

    # ica = FastICA(n_components=8, random_state=2025, max_iter=1000)
    # sources = ica.fit_transform(song_data_transposed)
    # mixing_matrix = ica.mixing_

    # reject components when thresholds are met

    # skew, kurtosis, amplitudes, etc.
