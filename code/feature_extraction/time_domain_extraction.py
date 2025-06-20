import numpy as np
import scipy.stats
from statsmodels.tsa.ar_model import AutoReg
import antropy as ant


def statistical_values(song_data):
    # song_data in shape (8, 7000ish)
    print("TODO")
    # return this as dict
    statistical_features = {}

    # max, min, mean, std, range, etc.
    statistical_features["max"] = np.max(song_data, axis=1)
    statistical_features["min"] = np.min(song_data, axis=1)
    statistical_features["std"] = np.std(song_data, axis=1)
    statistical_features["mean"] = np.mean(song_data, axis=1)
    statistical_features["median"] = np.median(song_data, axis=1)
    statistical_features["var"] = np.var(song_data, axis=1)
    statistical_features["range"] = np.ptp(song_data, axis=1)

    statistical_features["skew"] = scipy.stats.skew(song_data, axis=1)
    statistical_features["kurtosis"] = scipy.stats.kurtosis(song_data, axis=1)

    statistical_features["rms"] = np.sqrt(np.mean(song_data**2, axis=1))
    statistical_features["zcr"] = np.array([zcr(signal) for signal in song_data])

    statistical_features["h_mobility"] = np.array(
        [ant.hjorth_params(signal)[0] for signal in song_data]
    )
    statistical_features["h_complexity"] = np.array(
        [ant.hjorth_params(signal)[1] for signal in song_data]
    )

    statistical_features["ar_model"] = [
        autoregressive_modeling(signal) for signal in song_data
    ]

    statistical_features["haiguchi"] = np.array(
        [ant.higuchi_fd(signal) for signal in song_data]
    )

    return statistical_features


def zcr(signal):
    return np.sum(np.diff(np.sign(signal)) != 0) / len(signal)


def autoregressive_modeling(signal):
    model = AutoReg(signal, lags=10)
    fitted_model = model.fit()

    return {
        "coefficients": fitted_model.params[1:],
        "intercept": fitted_model.intercept[0],
        "aic": fitted_model.aic,
        "bic": fitted_model.bic,
        "residuals": fitted_model.resid,
    }
