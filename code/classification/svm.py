from code.classification.classification import perform_hyperparam_tuning, FreqDomain
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def train(scaling=True):
    print("Initialize")
    if scaling:
        svm = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lda",
                    LinearDiscriminantAnalysis(n_components=1),
                ),  # . n_components = classes - 1
                ("svm", SVC(random_state=1997, class_weight="balanced")),
            ]
        )

        params = {
            "svm__kernel": ["rbf", "linear"],
            "svm__C": [0.1, 1, 10],
            "svm__gamma": ["scale", 0.01, 0.1],
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
        "SVM_DEAP_BINARY_TIME_FREQ_",
        session_validation=False,
        psd=FreqDomain.welch,
        random_search=True,
        deap=True,
    )
    print("Finish")


if __name__ == "__main__":
    train(scaling=True)
