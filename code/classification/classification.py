import mlflow
import mlflow.sklearn
from enum import Enum
from datetime import datetime
import numpy as np

from code.classification.data import load_data, load_data_session
from code.classification.deap_data import load_deap_dataset
from code.classification.learning import setup_mlflow_experiment, create_search
from sklearn.model_selection import StratifiedKFold, ParameterGrid, ShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from code.classification.feature_selector import MRMRSelector


class FreqDomain(Enum):
    welch = "psd_welch"
    periodogram = "periodogram"
    fft = "psd_fft"


# ---------------------------------------------------------------------------
# Original function (unchanged) — kept for reference / non-subject-based runs
# ---------------------------------------------------------------------------

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

    cv_type = "Session" if session_validation else "Stratified"
    search_type = "Random" if random_search else "Grid"
    timestamp = datetime.now().strftime("%H%M%S")
    dataset = "DEAP" if deap else "Headset"

    run_name = (
        f"{estimator_name}_{psd.name}_{cv_type}CV_{search_type}Search_"
        f"{timestamp}_{dataset}_Dataset_Only_Freq_Domain"
    )

    with mlflow.start_run(run_name=run_name):
        grid_search = create_search(
            estimator=estimator,
            params=param_grid,
            session_validation=session_validation,
            random_search=random_search,
        )
        grid_search.fit(X, y)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)
        mlflow.sklearn.log_model(
            sk_model=grid_search.best_estimator_, artifact_path="best_model"
        )

        best_index = grid_search.best_index_
        best_accuracy = grid_search.cv_results_["mean_test_accuracy"][best_index]
        best_f1 = grid_search.cv_results_["mean_test_f1_macro"][best_index]

        print(f"Best estimator params: {grid_search.best_params_}")
        print(f"Best estimator accuracy: {best_accuracy:.4f}")
        print(f"Best estimator balanced accuracy: {grid_search.best_score_}")
        print(f"Best estimator f-score: {best_f1:.4f}")
        print(f"Best estimator: {grid_search.best_estimator_}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_subject_data(
    subject_idx: int,
    deap: bool,
    psd: FreqDomain,
) -> tuple[np.ndarray, np.ndarray]:
    """Load data for a single subject from the appropriate dataset."""
    if deap:
        return load_deap_dataset(subject=subject_idx)
    else:
        return load_data_session(session_number=subject_idx, process=True)


def _build_pipeline(params: dict, random_state: int = 1997) -> Pipeline:
    """Construct a fresh Pipeline from a parameter dict."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("selector", MRMRSelector(k=params["selector__k"])),
        ("svm", SVC(
            C=params["svm__C"],
            kernel=params["svm__kernel"],
            gamma=params.get("svm__gamma", "scale"),
            random_state=random_state,
            class_weight="balanced",
        )),
    ])


def _inner_best_params(
    X_tv: np.ndarray,
    y_tv: np.ndarray,
    param_list: list,
    inner_cv,
    random_state: int = 1997,
) -> dict:
    """
    Run inner CV over param_list on the outer train+val split (X_tv / y_tv).
    Returns the param dict that achieves the highest mean macro-F1 on the
    inner validation folds.
    """
    best_score = -1.0
    best_params = param_list[0]

    for params in param_list:
        inner_f1_scores = []
        for train_idx, val_idx in inner_cv.split(X_tv, y_tv):
            clf = _build_pipeline(params, random_state)
            clf.fit(X_tv[train_idx], y_tv[train_idx])
            y_val_pred = clf.predict(X_tv[val_idx])
            inner_f1_scores.append(
                f1_score(y_tv[val_idx], y_val_pred, average="macro", zero_division=0)
            )
        mean_f1 = float(np.mean(inner_f1_scores))
        if mean_f1 > best_score:
            best_score = mean_f1
            best_params = params

    return best_params


# ---------------------------------------------------------------------------
# Option A — Full-fold nested CV  (supervisor's main suggestion)
#
# Structure per subject:
#   Outer loop : StratifiedKFold(n_outer)   →  determines TEST fold
#   Inner loop : StratifiedKFold(n_inner)   →  determines VAL fold for
#                                               hyperparameter selection
#   Final step : retrain on outer train+val with best params, predict on TEST
#
# With n_outer=5, n_inner=4:
#   each outer fold uses n_inner inner folds to select params,
#   then tests on the held-out 20% — no overlap between selection and test.
# ---------------------------------------------------------------------------

def perform_subject_based_nested_cv(
    param_grid: dict,
    num_subjects: int,
    deap: bool,
    psd: FreqDomain = FreqDomain.welch,
    n_outer: int = 5,
    n_inner: int = 4,
    n_iter: int = 50,
    random_state: int = 1997,
) -> list[dict]:
    """
    Nested cross-validation for subject-based hyperparameter evaluation.

    For every subject the function:
      1. Loads subject data from DEAP or the headset dataset (controlled by
         ``deap`` and ``psd``).
      2. Splits the trials into n_outer folds (outer loop).
      3. For each outer fold, uses the remaining data as a pool from which
         n_inner inner folds pick the best hyperparameters.
      4. Retrains on the full outer pool with the selected hyperparameters.
      5. Evaluates on the outer TEST fold (never seen during steps 3-4).

    Parameters
    ----------
    param_grid    : hyperparameter search space
    num_subjects  : number of subjects to iterate over
    deap          : if True, load from DEAP; if False, load from headset data
    psd           : frequency-domain method (ignored when deap=True)
    n_outer       : number of outer folds
    n_inner       : number of inner folds for hyperparameter selection
    n_iter        : number of random parameter combinations to sample
    random_state  : reproducibility seed

    Returns a list of per-subject result dicts.
    """
    param_list = list(ParameterGrid(param_grid))
    dataset_label = "DEAP" if deap else f"Headset_{psd.name}"
    all_results = []

    for i in range(num_subjects):
        print(f"========= SUBJECT {i:02d} START ({dataset_label}) =========")
        X_subj, y_subj = _load_subject_data(i, deap, psd)

        outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)
        inner_cv = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=random_state)

        outer_acc, outer_f1, best_params_per_fold = [], [], []

        for outer_fold, (outer_tv_idx, test_idx) in enumerate(
            outer_cv.split(X_subj, y_subj)
        ):
            X_tv, y_tv = X_subj[outer_tv_idx], y_subj[outer_tv_idx]
            X_test, y_test = X_subj[test_idx], y_subj[test_idx]

            # --- inner loop: pick best hyperparameters on train+val pool ---
            best_params = _inner_best_params(
                X_tv, y_tv, param_list, inner_cv, random_state
            )
            best_params_per_fold.append(best_params)

            # --- retrain on full outer train+val, evaluate on test ----------
            final_clf = _build_pipeline(best_params, random_state)
            final_clf.fit(X_tv, y_tv)
            y_pred = final_clf.predict(X_test)

            fold_acc = accuracy_score(y_test, y_pred)
            fold_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            outer_acc.append(fold_acc)
            outer_f1.append(fold_f1)

            print(
                f"  outer fold {outer_fold + 1}/{n_outer} "
                f"| acc={fold_acc:.4f}  f1={fold_f1:.4f} "
                f"| params={best_params}"
            )

        mean_acc = float(np.mean(outer_acc))
        mean_f1 = float(np.mean(outer_f1))
        print(
            f"Subject {i:02d} summary → "
            f"mean acc={mean_acc:.4f}  mean f1={mean_f1:.4f}  "
            f"fold accs={[round(a, 4) for a in outer_acc]}\n"
        )

        all_results.append({
            "subject": i,
            "mean_accuracy": mean_acc,
            "mean_f1": mean_f1,
            "fold_accuracies": outer_acc,
            "fold_f1_scores": outer_f1,
            "best_params_per_fold": best_params_per_fold,
        })

    overall_acc = float(np.mean([r["mean_accuracy"] for r in all_results]))
    overall_f1 = float(np.mean([r["mean_f1"] for r in all_results]))
    print("=" * 60)
    print(f"Overall mean accuracy : {overall_acc:.4f}")
    print(f"Overall mean F1       : {overall_f1:.4f}")
    print("=" * 60)

    return all_results


# ---------------------------------------------------------------------------
# Option B — Small-holdout nested CV  (supervisor's alternative suggestion)
#
# Instead of holding back a full inner fold for validation, only 1 or 3
# trials are withheld. This keeps more data in training while still
# providing a separate validation signal for hyperparameter selection.
#
# Uses ShuffleSplit for the inner validation so the holdout size is exact
# regardless of class balance.
# ---------------------------------------------------------------------------

def perform_subject_based_nested_cv_small_val(
    param_grid: dict | list,
    num_subjects: int,
    deap: bool,
    psd: FreqDomain = FreqDomain.welch,
    n_outer: int = 5,
    n_inner_splits: int = 8,
    n_val_trials: int = 3,
    random_state: int = 1997,
) -> list[dict]:
    
    # Use ParameterGrid for exact grid search, matching our Option A setup
    param_list = list(ParameterGrid(param_grid))
    dataset_label = "DEAP" if deap else f"Headset_{psd.name}"
    all_results = []

    for i in range(num_subjects):
        print(f"========= SUBJECT {i:02d} START ({dataset_label}) =========")
        X_subj, y_subj = _load_subject_data(i, deap, psd)

        # Outer loop still uses Stratified CV (16 train, 4 test)
        outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=random_state)

        outer_acc, outer_f1, best_params_per_fold = [], [], []

        for outer_fold, (outer_tv_idx, test_idx) in enumerate(
            outer_cv.split(X_subj, y_subj)
        ):
            X_tv, y_tv = X_subj[outer_tv_idx], y_subj[outer_tv_idx]
            X_test, y_test = X_subj[test_idx], y_subj[test_idx]

            # Inner validation: ShuffleSplit holds out exactly n_val_trials each time.
            # This prevents StratifiedKFold from crashing on tiny datasets!
            inner_cv = ShuffleSplit(
                n_splits=n_inner_splits,
                test_size=n_val_trials,
                random_state=random_state,
            )

            best_params = _inner_best_params(
                X_tv, y_tv, param_list, inner_cv, random_state
            )
            best_params_per_fold.append(best_params)

            final_clf = _build_pipeline(best_params, random_state)
            final_clf.fit(X_tv, y_tv)
            y_pred = final_clf.predict(X_test)

            fold_acc = accuracy_score(y_test, y_pred)
            fold_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            outer_acc.append(fold_acc)
            outer_f1.append(fold_f1)

            print(
                f"  outer fold {outer_fold + 1}/{n_outer} "
                f"| acc={fold_acc:.4f}  f1={fold_f1:.4f} "
                f"| params={best_params}"
            )

        mean_acc = float(np.mean(outer_acc))
        mean_f1 = float(np.mean(outer_f1))
        print(
            f"Subject {i:02d} summary → "
            f"mean acc={mean_acc:.4f}  mean f1={mean_f1:.4f}\n"
        )

        all_results.append({
            "subject": i,
            "mean_accuracy": mean_acc,
            "mean_f1": mean_f1,
            "fold_accuracies": outer_acc,
            "fold_f1_scores": outer_f1,
            "best_params_per_fold": best_params_per_fold,
        })

    overall_acc = float(np.mean([r["mean_accuracy"] for r in all_results]))
    overall_f1 = float(np.mean([r["mean_f1"] for r in all_results]))
    print("=" * 60)
    print(f"Overall mean accuracy : {overall_acc:.4f}")
    print(f"Overall mean F1       : {overall_f1:.4f}")
    print("=" * 60)

    return all_results