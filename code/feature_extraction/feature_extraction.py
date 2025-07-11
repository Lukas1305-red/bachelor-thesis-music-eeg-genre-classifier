from code.feature_extraction.frequency_domain_extraction import (
    frequency_domain_feature_extraction,
)
from code.feature_extraction.time_domain_extraction import (
    time_domain_feature_extraction,
)
from code.feature_extraction.time_frequency_extraction import (
    time_frequency_domain_feature_extraction,
)


def feature_extraction(song_data):
    """
    song_data: song_data should have the shape (n_channels, n_samples)
    """
    ret = {}
    ret["time_domain"] = time_domain_feature_extraction(song_data)
    ret["frequency_domain"] = frequency_domain_feature_extraction(song_data)
    ret["time_frequency_domain"] = time_frequency_domain_feature_extraction(song_data)

    features = {}
    keys = ["time_domain", "frequency_domain", "time_frequency_domain"]

    for key in keys:
        current = ret[key]
        for feature_key, value in current.items():
            features[feature_key] = value
    return features
