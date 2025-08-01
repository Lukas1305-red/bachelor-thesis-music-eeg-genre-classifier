import pickle
import numpy as np
from code.feature_extraction.feature_extraction import feature_extraction


def load_deap_dataset():
    eeg_data = load_deap_eeg()
    assert np.shape(eeg_data) == (1280, 32, 8064)

    feature_array = []
    for song in eeg_data:
        song_features = feature_extraction(song_data=song)
        feature_values = list(song_features.values())
        feature_array.append(feature_values)
    feature_array_np = np.array(feature_array)

    labels = load_deap_labels()

    return feature_array_np, labels


def load_deap_eeg():
    eeg_data = load_all_subjects()
    eeg_data_np = np.array(eeg_data, axis=0)
    eeg_data_reshaped = eeg_data_np.reshape(1280, 40, 8064)  # no assertion necessary
    eeg_data_reshaped_eeg = eeg_data_reshaped[:, 32, :]  # only keep EEG channels
    return eeg_data_reshaped_eeg


# TODO: Write in .csv or pickle
def load_deap_labels():
    deap_labels = [
        "Folk",
        "Rock",
        "Pop",
        "Rock",
        "Rock",
        "Rock",
        "Electronic",
        "Pop",
        "Rock",
        "Metal",
        "Pop",
        "Folk",
        "Folk",
        "Folk",
        "Folk",
        "Folk",
        "Electronic",
        "Folk",
        "Folk",
        "Pop",
        "Folk",
        "Rock",
        "Folk",
        "Pop",
        "Folk",
        "Folk",
        "Pop",
        "Pop",
        "Pop",
        "Folk",
        "Metal",
        "Metal",
        "Metal",
        "Electronic",
        "Metal",
        "Metal",
        "Metal",
        "Metal",
        "Metal",
        "Metal",
    ]
    labels = []
    # iterate over all subjects
    for i in range(32):
        labels.extend(deap_labels)

    assert np.shape(labels) == (1280)
    return labels


def load_all_subjects():
    all_subjects = []
    for i in range(32):
        subject = f"s{str(i + 1).zfill(2)}"
        subject_data = load_subject(subject)
        all_subjects.append(subject_data)
    assert np.shape(all_subjects) == (32, 40, 40, 8064)
    return all_subjects


def load_subject(subject: str):
    file_path = f"data/deap/deap/subjects/{subject}.dat"
    subject_data = pickle.load(open(file_path, "rb"), encoding="latin1")
    subject_signal_data = np.array(subject_data["deap"])
    assert np.shape(subject_signal_data) == (40, 40, 8064)
    return subject_signal_data
