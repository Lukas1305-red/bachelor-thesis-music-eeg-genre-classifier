from code.classification.classification import perform_hyperparam_tuning, FreqDomain
from code.classification.feature_selector import MRMRSelector
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


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
            "selector__k": [16, 32, 64, 96, 128, 256],
            "svm__C": [0.1, 1, 10],
            "svm__kernel": ["linear", "rbf"],
        }
    else:
        svm = SVC(random_state=1997)
        params = {
            "kernel": ["rbf", "linear"],
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", 0.1, 0.01, 0.001],
            "decision_function_shape": ["ovr"],
        }

    print("Training")
    perform_hyperparam_tuning(
        svm,
        params,
        "SVM_DEAP_BINARY_2S_MRMR",
        session_validation=False,
        psd=FreqDomain.welch,
        random_search=True,
        deap=True,
    )
    print("Finish")


if __name__ == "__main__":
    train(scaling=True)
