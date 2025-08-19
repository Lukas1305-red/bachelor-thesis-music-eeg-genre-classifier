import mlflow
import mlflow.sklearn
from enum import Enum
from datetime import datetime
import numpy as np

from code.classification.data import load_data, load_data_session
from code.classification.deap_data import load_deap_dataset
from code.classification.learning import setup_mlflow_experiment, create_search
from sklearn.model_selection import StratifiedKFold, ParameterSampler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from code.classification.feature_selector import MRMRSelector


class FreqDomain(Enum):
    welch = "psd_welch"
    periodogram = "periodogram"
    fft = "psd_fft"


def perform_hyperparam_tuning(
    estimator,
    param_grid,
    estimator_name,
    session_validation,
    psd: FreqDomain,
    random_search: bool,
    deap: bool,
    subj: int = None,
):
    # Experiment tracking
    experiment_name, experiment_id = setup_mlflow_experiment(
        estimator_name, psd, session_validation
    )
    if deap:
        if subj is None:
            X, y = load_deap_dataset()
        else:
            X, y = load_deap_dataset(subject=subj)
    else:
        if subj is None:
            X, y = load_data(psd_to_keep=psd)
        else:
            X, y = load_data_session(session_number=subj, process=True)

    mlflow.sklearn.autolog(max_tuning_runs=1)

    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {experiment_name}")

    # Create descriptive run name
    cv_type = "Session" if session_validation else "Stratified"
    search_type = "Random" if random_search else "Grid"
    timestamp = datetime.now().strftime("%H%M%S")
    dataset = "DEAP" if deap else "Headset"

    run_name = f"{estimator_name}_{psd.name}_{cv_type}CV_{search_type}Search_{timestamp}_{dataset}_Dataset_Only_Freq_Domain"

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

        # Get the index of the best estimator
        best_index = grid_search.best_index_

        # Get the mean test accuracy of the best estimator
        best_accuracy = grid_search.cv_results_["mean_test_accuracy"][best_index]
        best_f1 = grid_search.cv_results_["mean_test_f1_macro"][best_index]

        # Print it
        print(f"Best estimator params: {grid_search.best_params_}")
        print(f"Best estimator accuracy: {best_accuracy:.4f}")
        print(f"Best estimator balanced accuracy: {grid_search.best_score_}")

        print(f"Best estimator f-score: {best_f1:.4f}")
        print(f"Best estimator: {grid_search.best_estimator_}")


def perform_subject_based_param_tuning(base_pipeline, param_grid, num_subjects):
    """
    Only works with SVM for now
    """
    RANDOM_STATE = 1997
    CV_FOLD = 4
    N_ITER = 50
    # storage
    X, y = load_deap_dataset()
    param_sets = ParameterSampler(param_grid, N_ITER, random_state=RANDOM_STATE)
    results = []
    run_counter = 0

    for params in param_sets:
        print(f"===========PARAMETER SET {run_counter}==========")
        run_counter += 1
        subject_acc_scores = []
        subject_f1_scores = []
        for i in range(num_subjects):
            first_song_idx = 40 * i
            last_song_idx = 40 * (i + 1)
            X_subj = X[first_song_idx:last_song_idx, :]
            y_subj = y[first_song_idx:last_song_idx]

            cv = StratifiedKFold(
                n_splits=CV_FOLD, shuffle=True, random_state=RANDOM_STATE
            )
            fold_acc_scores = []
            fold_f1_scores = []
            for train_idx, test_idx in cv.split(X_subj, y_subj):
                clf = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("selector", MRMRSelector(k=params["selector__k"])),
                        (
                            "svm",
                            SVC(
                                C=params["svm__C"],
                                kernel=params["svm__kernel"],
                                gamma=params["svm__gamma"],
                                random_state=1997,
                                class_weight="balanced",
                            ),
                        ),
                    ]
                )
                clf.fit(X_subj[train_idx], y_subj[train_idx])
                y_pred = clf.predict(X_subj[test_idx])
                fold_acc_scores.append(accuracy_score(y_subj[test_idx], y_pred))
                fold_f1_scores.append(f1_score(y_subj[test_idx], y_pred))
            subject_acc_scores.append(np.mean(fold_acc_scores))
            subject_f1_scores.append(np.mean(fold_f1_scores))

        mean_acc = np.mean(subject_acc_scores)
        mean_f1 = np.mean(subject_f1_scores)
        combined_score = 0.5 * mean_acc + 0.5 * mean_f1
        if mean_acc > 0.6:
            print(f"Mean accuracy: {mean_acc}")
            print(f"Mean f1-score: {mean_f1}")
            print(f"Combined score: {combined_score}")
            print(f"Parameters: {params}")
        else:
            print(f"Mean accuracy: {mean_acc}, lower than 0.63 :(")
        print(f"Subject accuracies: {subject_acc_scores}")
        print("=====================")

        # Save results
        results.append(
            {
                "params": params,
                "mean_accuracy": mean_acc,
                "mean_f1": mean_f1,
                "combined_score": combined_score,
                "subject_accuracies": subject_acc_scores,
            }
        )

    best_result = max(results, key=lambda x: x["combined_score"])
    best_acc = max(results, key=lambda x: x["mean_accuracy"])

    print("Best Combined Score:", best_result["combined_score"])
    print("Mean Accuracy:", best_result["mean_accuracy"])
    print("Mean F1:", best_result["mean_f1"])
    print("Subject accuracies:", best_result["subject_accuracies"])
    print("Best Parameters:", best_result["params"])
    print("================================")
    print("Best Mean Accuracy:", best_acc["mean_accuracy"])
    print("Combined Score:", best_acc["combined_score"])
    print("Mean F1:", best_acc["mean_f1"])
    print("Subject accuracies:", best_acc["subject_accuracies"])
    print("Best Parameters:", best_acc["params"])
