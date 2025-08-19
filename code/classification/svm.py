from code.classification.classification import perform_hyperparam_tuning, FreqDomain
from code.classification.feature_selector import MRMRSelector
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

NUM_SUBJECTS = 32


def train(scaling=True):
    print("Initialize")
    if scaling:
        svm = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("selector", MRMRSelector()),
                ("svm", SVC(random_state=1997, class_weight="balanced")),
            ]
        )
        params = {
            "selector__k": [2, 5, 8, 10, 15, 20],
            "svm__C": [0.001, 0.1, 1, 10, 100],
            "svm__kernel": ["linear", "rbf"],
            "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
        }
    else:
        svm = SVC(random_state=1997)
        params = {
            "kernel": ["rbf", "linear"],
            "C": [0.001, 0.1, 1, 10, 100],
            "gamma": ["scale", 0.1, 0.01, 0.001],
            "decision_function_shape": ["ovr"],
        }

    print("Training")
    # perform_subject_based_param_tuning(
    #    base_pipeline=svm,
    #    param_grid=params,
    #    num_subjects=NUM_SUBJECTS
    # )
    perform_hyperparam_tuning(
        svm,
        params,
        "DEAP_SUBJ_00",
        session_validation=False,
        psd=FreqDomain.welch,
        random_search=False,
        deap=True,
    )
    print("Finish")


if __name__ == "__main__":
    train(scaling=True)
