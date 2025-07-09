import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.metrics import (
    make_scorer,
    f1_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.base import clone

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
    # define cross validation
    cv = KFold(n_splits=20, shuffle=False, random_state=1997)

    # define scoring
    scoring = {
        "f1_macro": make_scorer(f1_score, average="macro"),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "recall_macro": make_scorer(recall_score, average="macro"),
    }

    return GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        refit="balanced_accuracy",
        cv=cv,
    )


def perform_grid_search(estimator, param_grid, estimator_name):
    # Experiment tracking
    mlflow.sklearn.autolog(max_tuning_runs=10)

    X, y = load_data()

    with mlflow.start_run(run_name=f"{estimator_name} Hyperparameter Tuning"):
        grid_search = create_grid_search(estimator=estimator, param_grid=param_grid)

        grid_search.fit(X, y)

        # get the best model
        best_estimator = grid_search.best_estimator_

        cm = draw_confusion_matrix(best_estimator)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")

        mlflow.log_figure(plt.gcf(), "confusion_matrix.png")


def draw_confusion_matrix(estimator, X, y):
    estimator_clone = clone(estimator)
    y_pred = cross_val_predict(
        estimator_clone, X, y, cv=KFold(n_splits=20, shuffle=False, random_state=1997)
    )
    cm = confusion_matrix(y, y_pred)
    return cm
