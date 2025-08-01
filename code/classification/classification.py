import pandas as pd
import mlflow
import mlflow.sklearn
from enum import Enum
from datetime import datetime

from code.classification.data import load_data
from code.classification.deap_data import load_deap_dataset
from code.classification.learning import setup_mlflow_experiment, create_search


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
    dataset = "DEAP" if deap else "Headset"

    run_name = f"{estimator_name}_{psd.name}_{cv_type}CV_{search_type}Search_{timestamp}_{dataset}_Dataset"
    if deap:
        X, y = load_deap_dataset()
    else:
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
        results_df = pd.DataFrame(grid_search.cv_results_)

        # Print all rows, showing relevant columns
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(
                results_df[
                    [
                        "params",
                        "mean_test_f1_macro",
                        "mean_test_accuracy",
                        "mean_test_recall_macro",
                        "rank_test_f1_macro",
                    ]
                ].sort_values(by="rank_test_f1_macro")
            )
        print(grid_search.best_estimator_)
