import scipy.signal
import scipy.stats
import numpy as np
from sklearn.decomposition import FastICA

# *****THRESHOLD CONSTANTS*****

KURTOSIS_PERCENTILE = 95
VAR_PERCENTILE = 95
SKEW_PERCENTILE = 95


# Source: https://swharden.com/blog/2020-09-23-signal-filtering-in-python/
def bandpass(data, edges, sample_rate, poles=5):
    sos = scipy.signal.butter(poles, edges, "bandpass", fs=sample_rate, output="sos")
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data


def notch_filter(data, f0, sample_rate, Q=30):
    b, a = scipy.signal.iirnotch(f0, Q, fs=sample_rate)
    return scipy.signal.filtfilt(b, a, data)


def check_rejection(
    kurt,
    skew_val,
    var_percentage,
    focal_concentration,
    group_A_percentage,
    group_B_percentage,
    component_variances,
    sources,
):
    # calculate thresholds
    kurtosis_vals = scipy.stats.kurtosis(sources, axis=0, fisher=True)
    skew_vals = np.abs(scipy.stats.skew(sources, axis=0))
    var_percentages = component_variances / np.sum(component_variances)

    kurtosis_threshold = np.percentile(kurtosis_vals, KURTOSIS_PERCENTILE)
    skew_threshold = np.percentile(skew_vals, SKEW_PERCENTILE)
    var_threshold = np.percentile(var_percentages, VAR_PERCENTILE)

    if kurt > kurtosis_threshold:
        return True
    if skew_val > skew_threshold:
        return True
    if var_percentage > var_threshold:
        return True
    # if focal_concentration > FOCAL_MAX:
    #     return True
    # if group_A_percentage > GROUP_MAX or group_B_percentage > GROUP_MAX:
    #     return True
    return False


def artifact_removal_ica(song_data):
    # song_data has the shape (8. 7000ish)
    # center data
    song_data = song_data - np.mean(song_data, axis=1, keepdims=True)
    song_data_transposed = np.transpose(song_data)

    ica = FastICA(n_components=8, random_state=2025, max_iter=1000)
    sources = ica.fit_transform(song_data_transposed)
    mixing_matrix = ica.mixing_

    rejected_components = []

    component_variances = np.var(sources, axis=0)
    total_variance = np.sum(component_variances)
    # reject components when thresholds are met
    for i in range(sources.shape[1]):
        # calculate important values
        kurt = scipy.stats.kurtosis(sources[:, i], fisher=True)
        skew_val = np.abs(scipy.stats.skew(sources[:, i]))
        var_percentage = component_variances[i] / total_variance

        component_weights = np.abs(mixing_matrix[:, i])
        max_weight = np.max(component_weights)
        total_weight = np.sum(component_weights)
        focal_concentration = max_weight / total_weight

        group_A = [0, 1, 3]
        group_B = [4, 5, 6, 7]
        weight_A = np.sum(component_weights[group_A])
        weight_B = np.sum(component_weights[group_B])

        group_A_percentage = weight_A / total_weight
        group_B_percentage = weight_B / total_weight

        # decide whether to reject component
        rejection_condition = check_rejection(
            kurt,
            skew_val,
            var_percentage,
            focal_concentration,
            group_A_percentage,
            group_B_percentage,
        )

        if rejection_condition:
            rejected_components.append(i)

    sources_cleaned = sources.copy()
    sources_cleaned[:, rejected_components] = 0

    reconstructed_data = ica.inverse_transform(sources_cleaned)
    cleaned_song_data = reconstructed_data.T

    return cleaned_song_data
