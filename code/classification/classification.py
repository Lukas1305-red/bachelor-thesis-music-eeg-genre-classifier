import mlflow.sklearn
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from enum import Enum
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import make_scorer, f1_score, recall_score, balanced_accuracy_score
from sklearn.feature_selection import VarianceThreshold
from datetime import datetime


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
        # X_df = filter_out_freq_domains(X_df_buf, FreqDomain.welch)
        feature_names = X_df_buf.columns.tolist()

        # store features
        X.append(X_df_buf.values)
        print(f"Finished loading session {session_number}!")

    le = LabelEncoder()
    y_numerical = le.fit_transform(y)
    X_combined = np.vstack(X)

    X_filtered = remove_constant_features(
        X_original=X_combined, feature_names=feature_names
    )

    print("Finished loading data!")
    return X_filtered, y_numerical


def recursive_feature_selection(X):
    print("Start Feature Selectoin")


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


def create_search(estimator, params, session_validation, random_search: bool):
    # define cross validation
    if session_validation:
        cv = KFold(n_splits=20, shuffle=False)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1997)

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
            n_iter=300,
            cv=cv,
            scoring=scoring,
            refit="balanced_accuracy",
            n_jobs=-1,
            random_state=1997,
            verbose=1,
            error_score="raise",
            return_train_score=True,
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


def setup_mlflow_experiment(
    estimator_name: str, psd: FreqDomain, session_validation: bool
):
    """Set up MLflow experiment with descriptive names and tags"""

    # Create descriptive experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    slurm_id = os.environ.get("SLURM_JOB_ID", "local")
    cv_type = "SessionCV" if session_validation else "StratifiedCV"

    # Create descriptive folder name
    experiment_name = f"EEG_{estimator_name}_{psd.name}_{cv_type}_{timestamp}"

    # Set custom tracking URI to get descriptive folder names
    tracking_uri = f"file:./mlruns_descriptive/{experiment_name}"
    mlflow.set_tracking_uri(tracking_uri)

    # Set or create experiment (will be stored in the descriptive folder)
    try:
        experiment = mlflow.get_experiment_by_name("default")  # Use default experiment
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                name="default",
                tags={
                    "project": "EEG-Music-Genre-Classification",
                    "dataset_version": "v4_features",
                    "estimator": estimator_name,
                    "psd_method": psd.name,
                    "cv_type": cv_type,
                    "slurm_job_id": str(slurm_id),
                    "created_date": datetime.now().isoformat(),
                },
            )
        else:
            experiment_id = experiment.experiment_id
    except Exception:
        # Fallback to setting experiment
        mlflow.set_experiment("default")
        experiment_id = mlflow.get_experiment_by_name("default").experiment_id

    return experiment_name, experiment_id


def perform_hyperparam_tuning(
    estimator,
    param_grid,
    estimator_name,
    session_validation,
    psd: FreqDomain,
    random_search: bool,
):
    # Experiment tracking
    experiment_name, experiment_id = setup_mlflow_experiment(
        estimator_name, psd, session_validation
    )

    mlflow.sklearn.autolog(max_tuning_runs=1)

    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {experiment_name}")

    # Create descriptive run name
    cv_type = "Session" if session_validation else "Stratified"
    search_type = "Random" if random_search else "Grid"
    timestamp = datetime.now().strftime("%H%M%S")

    run_name = (
        f"{estimator_name}_{psd.name}_{cv_type}CV_{search_type}Search_{timestamp}"
    )

    X, y = load_data(psd_to_keep=psd)
    with mlflow.start_run(run_name=f"{run_name}"):
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
        mlflow.sklearn.log_model(
            sk_model=grid_search.best_estimator_, artifact_path="best_model"
        )

        # Log all CV results
        results_df = pd.DataFrame(grid_search.cv_results_)
        mlflow.log_table(results_df, "grid_search_results.json")

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        print(grid_search.best_estimator_)
