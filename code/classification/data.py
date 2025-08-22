from enum import Enum
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from collections import Counter


DATA_PATH = "data/eeg_data_first_4s_unnormalized_v3/features"
LABEL_PATH = "data/eeg_data_v7_denoised/pickle"


class FreqDomain(Enum):
    welch = "psd_welch"
    periodogram = "periodogram"
    fft = "psd_fft"


def load_data_session(session_number: int, process=False, ret_label_encoder=False):
    str_session = str(session_number).zfill(3)
    print(f"Load Session {str_session}")
    filename = f"{str_session}.pkl"
    df_data: pd.DataFrame = pd.read_pickle(f"{DATA_PATH}/{filename}")
    df_label: pd.DataFrame = pd.read_pickle(f"{LABEL_PATH}/{filename}")
    genres = extract_genres(df_label)
    features = extract_features(df_data)

    print(f"Finished loading session {str_session}!")
    if process:
        le = LabelEncoder()
        y_numerical = le.fit_transform(genres)
        counter = Counter(y_numerical)

        for label, count in sorted(counter.items()):
            print(f"Class {label}: {count} samples")

        X_combined = df_data.to_numpy()
        print("\n=== Feature Info ===")
        print(f"X shape: {X_combined.shape}")
        print(f"y shape: {y_numerical.shape}")
        if ret_label_encoder:
            return X_combined, y_numerical, le
        else:
            return X_combined, y_numerical

    return features, genres


def extract_features(df_data):
    X_df_buf = df_data.drop(get_problematic_column_names(df_data), axis=1)
    X_df = filter_out_freq_domains(X_df_buf, domain_to_keep=FreqDomain.welch)
    return X_df


def extract_genres(df_labels):
    genres = df_labels["genre"]
    return genres.to_list()


def load_data(psd_to_keep: FreqDomain, ret_label_encoder=False):
    X = []
    y = []
    print("Start loading data...")
    for i in range(20):
        X_sess, y_sess = load_data_session(session_number=i)
        y.extend(y_sess)
        X.append(X_sess.values)

    le = LabelEncoder()
    y_numerical = le.fit_transform(y)
    counter = Counter(y_numerical)
    for label, count in sorted(counter.items()):
        print(f"Class {label}: {count} samples")
    X_combined = np.vstack(X)

    print("Finished loading data!")
    if ret_label_encoder:
        return X_combined, y_numerical, le
    else:
        return X_combined, y_numerical


def remove_constant_features(X_original, feature_names):
    print("Remove almost constant features...")
    selector = VarianceThreshold()
    X_filtered = selector.fit_transform(X_original)

    # Get indices of features that were kept
    kept_features_mask = selector.get_support()
    removed_features_mask = ~kept_features_mask

    # Log which features have been removed
    n_removed = X_original.shape[1] - X_filtered.shape[1]
    if n_removed > 0:
        removed_feature_names = [
            feature_names[i]
            for i in range(len(feature_names))
            if removed_features_mask[i]
        ]
        print(f"Removed {n_removed} constant features:")
        for i, feature_name in enumerate(removed_feature_names):
            print(f"  - {feature_name}")
    else:
        print("No constant features found to remove.")
    print(f"Final shape: {X_original.shape} -> {X_filtered.shape}")
    return X_filtered


def get_problematic_column_names(df: pd.DataFrame):
    nan_columns = df.columns[df.isnull().any()].tolist()
    inf_columns = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
    obj_columns = df.columns[df.dtypes == "object"].tolist()
    ret = list(set(nan_columns + inf_columns + obj_columns))
    print(f"These are the columns that are problematic: {ret}")
    return ret


def filter_out_freq_domains(df: pd.DataFrame, domain_to_keep: FreqDomain):
    ret = df.copy()
    for domain in FreqDomain:
        if domain != domain_to_keep:
            # filter out all occurences that have the prefix
            filtered_df = ret.filter(regex=f"^(?!{domain.name}_)")
            ret = filtered_df
    return ret


def remove_reference_channel(df: pd.DataFrame):
    # The channel 3 is the reference channel
    # Eliminate all columns that have Ch3 in them
    ch3_columns = []
    for col in df.columns:
        if "ch3" in col.lower():
            ch3_columns.append(col)
    df_cleaned = df.drop(columns=ch3_columns)
    print(f"Removed {len(ch3_columns)} features that came from Channel 3.")
    return df_cleaned
