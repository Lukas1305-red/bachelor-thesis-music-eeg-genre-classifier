from code.classification.classification import perform_hyperparam_tuning, FreqDomain
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train(scaling=True):
    print("Initialize")
    if scaling:
        svm = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("svm", SVC(random_state=1997)),
            ]
        )

        params = {
            "svm__kernel": ["linear"],
            "svm__C": [0.05, 0.1, 1, 10, 100],
            "svm__gamma": ["scale", 0.1, 0.01, 0.001],
            "svm__decision_function_shape": ["ovr"],
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
        "SVM",
        session_validation=False,
        psd=FreqDomain.welch,
        random_search=True,
        deap=True,
    )
    print("Finish")


if __name__ == "__main__":
    train(scaling=True)
