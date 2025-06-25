import scipy.signal
from enum import Enum
import numpy as np

SAMPLING_RATE = 250

class BrainRythm(Enum):
    ALPHA = (8, 13)
    BETA = (13, 30)
    THETA = (4, 8)
    DELTA = (0.5, 4)

def frequency_domain_feature_extraction(eeg_data):
    frequency_features = {}
    power_spectral_density_welch(eeg_data, 2, frequency_features)
    power_spectral_density_fft(eeg_data, 2, frequency_features)



def extract_brain_rythm(f, psd, rythm: BrainRythm):
    borders = rythm.value
    idx = np.logical_and(f >= borders[0], f <=borders[1])
    power = np.trapezoid(psd[idx], f[idx])
    return power


def power_spectral_density_welch(data, window_size, dictionary):
    dictionary["psd_welch"] = {}
    sample_rate = 250
    i = 0
    for channel in data:
        i += 1 # increase before all operation since we don't want to start at 0
        f, psd_welch = scipy.signal.welch(channel, fs=sample_rate, nperseg=sample_rate * window_size)
        dictionary["psd_welch"][f"EEG_Ch{i}"] = {}
        dictionary["psd_welch"][f"EEG_Ch{i}"]["f_bins"] = f
        dictionary["psd_welch"][f"EEG_Ch{i}"]["power"] = psd_welch

        for rythm in BrainRythm:
            power = extract_brain_rythm(f, psd_welch, rythm=rythm)
            dictionary["psd_welch"][f"EEG_Ch{i}"][f"{rythm.name}"] = power

def power_spectral_density_fft(data, window_size, dictionary):
    dictionary["psd_fft"] = {}
    sample_rate = 250
    n_samples = sample_rate * window_size
    i = 0
    for i, channel in enumerate(data, 1):
        if len(channel) >= n_samples:
            segment = channel[:n_samples]

            fft_vals = np.fft.fft(segment)
            fft_freqs = np.fft.fftfreq(n_samples, d=1/sample_rate)

            pos_mask = fft_freqs >= 0
            freqs = fft_freqs[pos_mask]
            psd = np.abs(fft_vals[pos_mask]) ** 2 / n_samples

            dictionary["psd_fft"][f"EEG_Ch{i}"] = {}
            dictionary["psd_fft"][f"EEG_Ch{i}"]["f_bins"] = freqs
            dictionary["psd_fft"][f"EEG_Ch{i}"]["power"] = psd

            for rythm in BrainRythm:
                power = extract_brain_rythm(freqs, psd, rythm=rythm)
                dictionary["psd_fft"][f"EEG_Ch{i}"][f"{rythm.name}"] = power

def periodogram(eeg_data, dictionary):
    dictionary["periodogram"] = {}
    sample_rate = 250
    for i, channel in enumerate(eeg_data, 1):
        freq, psd = scipy.signal.periodogram(channel, sample_rate, return_onesided=True)
        dictionary["periodogram"][f"EEG_Ch{i}"] = {}
        dictionary["periodogram"][f"EEG_Ch{i}"]["f_bins"] = freq
        dictionary["periodogram"][f"EEG_Ch{i}"]["power"] = psd

        for rythm in BrainRythm:
                power = extract_brain_rythm(freq, psd, rythm=rythm)
                dictionary["periodogram"][f"EEG_Ch{i}"][f"{rythm.name}"] = power




    
