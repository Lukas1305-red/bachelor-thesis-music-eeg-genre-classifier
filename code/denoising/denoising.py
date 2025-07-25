import scipy.signal
import scipy.stats
import numpy as np
from sklearn.decomposition import FastICA

# *****THRESHOLD CONSTANTS*****

DELTA_MAX = 0.5
SLOW_MAX = 0.65
BETA_MAX = 0.35
SPARSITY_MIN = 3


# Source: https://swharden.com/blog/2020-09-23-signal-filtering-in-python/
def bandpass(data, edges, sample_rate, poles=5):
    sos = scipy.signal.butter(poles, edges, "bandpass", fs=sample_rate, output="sos")
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


def notch_filter(data, f0, sample_rate, Q=30):
    b, a = scipy.signal.iirnotch(f0, Q, fs=sample_rate)
    return scipy.signal.filtfilt(b, a, data)


def apply_z_normalization(song_data, epsilon=1e-8):
    mean = np.mean(song_data, axis=1, keepdims=True)
    std = np.std(song_data, axis=1, keepdims=True)
    return (song_data - mean) / (std + epsilon)


def bandpower(frequencies, psd, band):
    band_freqs = np.logical_and(frequencies >= band[0], frequencies <= band[1])
    return np.trapezoid(psd[band_freqs], frequencies[band_freqs])


def component_rejected(
    component, ptp_threshold, kurt_threshold, skew_threshold, autorcorr_threshold
):
    # Peak-to-peak amplitude
    ptp = np.ptp(component)
    # Kurtosis
    kurt = scipy.stats.kurtosis(component)
    # Skewness
    skew = scipy.stats.skew(component)
    # autocorr
    autocorr = autocorr_lag(component)

    # Power in specific bands via PSD
    sample_rate = 250
    f, psd = scipy.signal.welch(component, fs=sample_rate, nperseg=512)
    total_power = np.trapezoid(f, psd)
    power_delta = bandpower(f, psd, (0.5, 4))
    power_beta = bandpower(f, psd, (12, 30))
    score = 0
    if ptp > ptp_threshold:
        score += 1
    if kurt > kurt_threshold:
        score += 1
    if skew > skew_threshold:
        score += 1
    if autocorr > autorcorr_threshold:
        score += 1

    if (power_delta / total_power) > DELTA_MAX:
        score += 1
    if (power_beta / total_power) > BETA_MAX:
        score += 1
    return score >= 2


def autocorr_lag(x):
    x = x - np.mean(x)
    return np.corrcoef(x[:-1], x[1:])[0, 1]


def get_thresholds(sources, percentile):
    ptp_values = [np.ptp(sources[:, i]) for i in range(sources.shape[1])]
    kurt_values = [scipy.stats.kurtosis(sources[:, i]) for i in range(sources.shape[1])]
    skew_values = [scipy.stats.skew(sources[:, i]) for i in range(sources.shape[1])]
    autocorr_values = [autocorr_lag(sources[:, i]) for i in range(sources.shape[1])]

    # Calculate thresholds
    ptp_threshold = np.percentile(ptp_values, percentile)
    kurt_threshold = np.percentile(kurt_values, percentile)
    skew_threshold = np.percentile(skew_values, percentile)
    autocorr_threshold = np.percentile(autocorr_values, percentile)

    return ptp_threshold, kurt_threshold, skew_threshold, autocorr_threshold


def calculate_sparsity(mixing_matrix, relative_threshold=0.2):
    max_abs = np.max(np.abs(mixing_matrix), axis=0)
    mask = np.abs(mixing_matrix) > (relative_threshold * max_abs)
    sparsity = np.sum(mask, axis=0)
    return sparsity


def artifact_removal_ica(data):
    data = data - np.mean(data, axis=1, keepdims=True)
    print(f"The shape of the data inside the ICA: {np.shape(data)}")
    song_data_transposed = np.transpose(data)

    ica = FastICA(n_components=6, random_state=2025, max_iter=1000)
    sources = ica.fit_transform(song_data_transposed)
    mixing_matrix = ica.mixing_

    # sparsity = calculate_sparsity(mixing_matrix=mixing_matrix)

    # TODO: Change to 90 or even 95 for less strict component removal
    ptp_threshold, kurt_threshold, skew_threshold, autocorr_threshold = get_thresholds(
        sources, 90
    )

    rejected_components = []

    # reject components when thresholds are met
    for i in range(sources.shape[1]):
        component = sources[:, i]
        if component_rejected(
            component, ptp_threshold, kurt_threshold, skew_threshold, autocorr_threshold
        ):
            rejected_components.append(i)
        # elif sparsity[i] < SPARSITY_MIN:
        #     rejected_components.append(i)

    cleaned_sources = sources.copy()
    for i in rejected_components:
        cleaned_sources[:, i] = 0

    print(f"Rejected components: {len(rejected_components)}")
    cleaned_data = np.transpose(np.dot(cleaned_sources, np.transpose(mixing_matrix)))
    return cleaned_data
