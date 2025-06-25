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
    power_spectral_density(eeg_data, 2, frequency_features)



def extract_brain_rythm(f, psd, rythm: BrainRythm):
    borders = rythm.value
    idx = np.logical_and(f >= borders[0], f <=borders[1])
    power = np.trapezoid(psd[idx], f[idx])
    return power


def power_spectral_density(data, window_size, dictionary):
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
        
    
