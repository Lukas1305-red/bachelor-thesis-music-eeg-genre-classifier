import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "../../data/eeg_data_v3_features/features"
LABEL_PATH = "../../data/eeg_data_v7_denoised/pickle"


def load_data():
    X = []
    y = []
    for i in range(20):
        session_number = str(i).zfill(3)
        filename = f"{session_number}.pkl"
        df_data: pd.DataFrame = pd.read_pickle(f"{DATA_PATH}/{filename}")
        df_label: pd.DataFrame = pd.read_pickle(f"{LABEL_PATH}/{filename}")

        # Extract label from df_label
        genres = df_label["genre"]
        y.extend(genres.toList())
        # drop data columns
        X_df = df_data.drop(get_problematic_column_names(df_data), axis=1)
        # store features
        X.append(X_df.values)

    le = LabelEncoder()
    y_numerical = le.fit_transform(y)
    X_combined = np.vstack(X)
    return X_combined, y_numerical


def get_problematic_column_names(df: pd.DataFrame):
    nan_columns = df.columns[df.isnull().any()].tolist()
    inf_columns = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
    obj_columns = df.columns[df.dtypes == "object"].tolist()

    return list(set(nan_columns + inf_columns + obj_columns))


def create_grid_search(estimator, param_grid):
    print("TODO")
