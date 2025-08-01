import pickle
import numpy as np


def load_deap_dataset():
    eeg_data = load_deap_eeg()
    labels = load_deap_labels()
    return eeg_data, labels


def load_deap_eeg():
    eeg_data = load_all_subjects()
    eeg_data_np = np.array(eeg_data, axis=0)
    eeg_data_np = eeg_data_np[:, 32, :]  # only keep EEG channels
    assert np.shape(eeg_data_np) == (40, 32, 8064)
    return eeg_data_np


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
    return all_subjects


def load_subject(subject: str):
    file_path = f"data/deap/deap/subjects({subject}.dat)"
    subject_data = pickle.load(open(file_path, "rb"), encoding="latin1")
    return np.array(subject_data["deap"])
