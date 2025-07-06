import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "../../data/eeg_data_v3_features/features"
LABEL_PATH = "../../data/eeg_data_v7_denoised/pickle"


def load_data():
    X = []
    y = []
    for i in range(20):
        session_number = str(i).zfill(3)
        filename = f"{session_number}.pkl"
        # df_data = pd.read_pickle(f"{DATA_PATH}/{filename}")
        df_label = pd.read_pickle(f"{LABEL_PATH}/{filename}")

        # Extract label from df_label
        genres = df_label["genre"]
        y.append(genres)

        # Store the feature vectors

    le = LabelEncoder()
    y_numerical = le.fit_transform(y)
    return X, y_numerical


def create_grid_search(estimator, param_grid):
    print("TODO")
