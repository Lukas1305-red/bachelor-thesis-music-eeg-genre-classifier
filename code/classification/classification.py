import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from enum import Enum
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    make_scorer,
    f1_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from sklearn.base import clone


class FreqDomain(Enum):
    welch = "psd_welch"
    periodogram = "periodogram"
    fft = "psd_fft"


DATA_PATH = "data/eeg_data_v4_features/features"
LABEL_PATH = "data/eeg_data_v7_denoised/pickle"


def load_data(psd_to_keep: FreqDomain):
    X = []
    y = []
    print("Start loading data...")
    for i in range(20):
        session_number = str(i).zfill(3)
        print(f"Load Session {session_number}...")
        filename = f"{session_number}.pkl"
        df_data: pd.DataFrame = pd.read_pickle(f"{DATA_PATH}/{filename}")
        df_label: pd.DataFrame = pd.read_pickle(f"{LABEL_PATH}/{filename}")

        # Extract label from df_label
        genres = df_label["genre"]
        y.extend(genres.to_list())
        # drop data columns
        X_df_buf = df_data.drop(get_problematic_column_names(df_data), axis=1)
        print(f"Filter frequencies only for {psd_to_keep.name}...")
        X_df = filter_out_freq_domains(X_df_buf, domain_to_keep=psd_to_keep)
        # store features
        X.append(X_df.values)
        print(f"Finished loading session {session_number}!")

    le = LabelEncoder()
    y_numerical = le.fit_transform(y)
    X_combined = np.vstack(X)
    print("Finished loading data!")
    return X_combined, y_numerical


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
            filtered_df = df.filter(regex=f"^(?!{domain.name}_)")
            ret = filtered_df
    return ret


def create_search(estimator, params, session_validation, random_search: bool):
    # define cross validation
    if session_validation:
        cv = KFold(n_splits=20, shuffle=False)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=1997)

    # define scoring
    scoring = {
        "f1_macro": make_scorer(f1_score, average="macro"),
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "recall_macro": make_scorer(recall_score, average="macro"),
    }

    if random_search:
        return RandomizedSearchCV(
            estimator=estimator,
            param_distributions=params,
            n_iter=100,
            cv=cv,
            scoring=scoring,
            refit="balanced_accuracy",
            n_jobs=-1,
            random_state=1997,
            verbose=1,
        )
    else:
        return GridSearchCV(
            estimator=estimator,
            param_grid=params,
            scoring=scoring,
            refit="balanced_accuracy",
            cv=cv,
            n_jobs=-1,
            verbose=2,
        )


def perform_hyperparam_tuning(
    estimator,
    param_grid,
    estimator_name,
    session_validation,
    psd: FreqDomain,
    random_search: bool,
):
    # Experiment tracking
    mlflow.sklearn.autolog(max_tuning_runs=10)
    print(mlflow.get_registry_uri())

    X, y = load_data(psd_to_keep=psd)
    with mlflow.start_run(run_name=f"{estimator_name} Hyperparameter Tuning"):
        grid_search = create_search(
            estimator=estimator,
            params=param_grid,
            session_validation=session_validation,
            random_search=random_search,
        )

        grid_search.fit(X, y)

        # Log best parameters and score
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        # Log all CV results
        results_df = pd.DataFrame(grid_search.cv_results_)
        mlflow.log_table(results_df, "grid_search_results.json")

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        print(grid_search.best_estimator_)


def draw_confusion_matrix(estimator, X, y):
    estimator_clone = clone(estimator)
    y_pred = cross_val_predict(
        estimator_clone, X, y, cv=KFold(n_splits=20, shuffle=False, random_state=1997)
    )
    cm = confusion_matrix(y, y_pred)
    return cm
