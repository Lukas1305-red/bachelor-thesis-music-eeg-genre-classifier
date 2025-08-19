from code.feature_extraction.time_frequency_extraction import (
    time_frequency_domain_feature_extraction,
)

from code.feature_extraction.frequency_domain_extraction import (
    frequency_domain_feature_extraction,
)

from code.feature_extraction.time_domain_extraction import (
    time_domain_feature_extraction,
)

import numpy as np

SAMPLE_RATE = 250


def feature_extraction(song_data):
    """
    song_data: song_data should have the shape (n_channels, n_samples)
    """
    ret = {}
    print(np.shape(song_data))
    song_data = np.array(song_data)
    song_data_first_4s = song_data[:, : SAMPLE_RATE * 4]
    assert np.shape(song_data_first_4s) == (7, SAMPLE_RATE * 4)
    ret["time_domain"] = time_domain_feature_extraction(song_data_first_4s)
    ret["frequency_domain"] = frequency_domain_feature_extraction(song_data_first_4s)
    ret["time_frequency_domain"] = time_frequency_domain_feature_extraction(
        song_data_first_4s
    )

    features = {}
    keys = ["time_domain", "frequency_domain", "time_frequency_domain"]

    for key in keys:
        if key != "time_domain":
            current = ret[key]
            for feature_key, value in current.items():
                features[feature_key] = value
    return features
