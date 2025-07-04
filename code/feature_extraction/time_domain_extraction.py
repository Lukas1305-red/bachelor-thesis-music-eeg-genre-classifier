import numpy as np
import scipy.stats
from statsmodels.tsa.ar_model import AutoReg
import antropy as ant


def time_domain_feature_extraction(song_data):
    # song_data in shape (8, 7000ish)
    time_domain_features = {}

    time_domain_features["max"] = np.max(song_data, axis=1)
    time_domain_features["min"] = np.min(song_data, axis=1)
    time_domain_features["std"] = np.std(song_data, axis=1)
    time_domain_features["mean"] = np.mean(song_data, axis=1)
    time_domain_features["median"] = np.median(song_data, axis=1)
    time_domain_features["var"] = np.var(song_data, axis=1)
    time_domain_features["range"] = np.ptp(song_data, axis=1)

    time_domain_features["skew"] = scipy.stats.skew(song_data, axis=1)
    time_domain_features["kurtosis"] = scipy.stats.kurtosis(song_data, axis=1)

    stacked = np.stack(song_data, axis=0)  # shape: (channels, samples)
    time_domain_features["rms"] = np.sqrt(np.mean(stacked**2, axis=1))
    time_domain_features["zcr"] = np.array([zcr(signal) for signal in song_data])

    time_domain_features["h_mobility"] = np.array(
        [ant.hjorth_params(signal)[0] for signal in song_data]
    )
    time_domain_features["h_complexity"] = np.array(
        [ant.hjorth_params(signal)[1] for signal in song_data]
    )

    time_domain_features["ar_model"] = [
        autoregressive_modeling(signal) for signal in song_data
    ]

    time_domain_features["higuchi"] = np.array(
        [ant.higuchi_fd(signal) for signal in song_data]
    )

    time_domain_features["mad"] = np.array([mad(signal) for signal in song_data])

    time_domain_features["moment3"] = scipy.stats.moment(song_data, order=3, axis=1)
    time_domain_features["moment4"] = scipy.stats.moment(song_data, order=4, axis=1)

    return time_domain_features


def zcr(signal):
    return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)


def mad(signal):
    return np.mean(np.abs(signal - np.mean(signal)))


def autoregressive_modeling(signal):
    model = AutoReg(signal, lags=10)
    fitted_model = model.fit()

    return {
        "coefficients": fitted_model.params[1:],
        "intercept": fitted_model.params[0],
        "aic": fitted_model.aic,
        "bic": fitted_model.bic,
        "residuals": fitted_model.resid,
    }
