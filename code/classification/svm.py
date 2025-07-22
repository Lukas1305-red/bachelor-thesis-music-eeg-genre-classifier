from code.classification.classification import perform_hyperparam_tuning, FreqDomain
from sklearn.svm import SVC


def train():
    svm = SVC(random_state=1997)

    params2 = {
        "kernel": ["rbf", "poly", "linear"],
        "C": [0.01, 0.1, 0.5, 1, 10, 100],
        "gamma": ["auto", "scale", 1, 0.75, 0.5, 0.25, 0.1, 0.01, 0.001],
        "decision_function_shape": ["ovo", "ovr"],
    }

    perform_hyperparam_tuning(
        svm,
        params2,
        "SVM",
        session_validation=False,
        psd=FreqDomain.welch,
        random_search=True,
    )


if __name__ == "__main__":
    train()
