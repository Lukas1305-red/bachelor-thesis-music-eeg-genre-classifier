import mlflow
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import make_scorer, f1_score, recall_score, accuracy_score
from enum import Enum
from datetime import datetime
import os


class FreqDomain(Enum):
    welch = "psd_welch"
    periodogram = "periodogram"
    fft = "psd_fft"


def create_search(estimator, params, session_validation, random_search: bool):
    # define cross validation
    if session_validation:
        cv = KFold(n_splits=20, shuffle=False)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1997)

    # define scoring
    scoring = {
        "f1_macro": make_scorer(f1_score, average="macro"),
        "accuracy": make_scorer(accuracy_score),
        "recall_macro": make_scorer(recall_score, average="macro"),
    }

    if random_search:
        return RandomizedSearchCV(
            estimator=estimator,
            param_distributions=params,
            n_iter=50,
            cv=cv,
            scoring=scoring,
            refit="f1_macro",
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
            verbose=1,
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
    experiment_name = f"{estimator_name}_{timestamp}_{slurm_id}"

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
