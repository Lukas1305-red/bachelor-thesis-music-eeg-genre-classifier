from code.classification.classification import perform_hyperparam_tuning, FreqDomain
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


def train():
    svm_pipeline = Pipeline(
        [("scaler", StandardScaler()), ("svm", SVC(random_state=1997))]
    )

    params = {
        "svm__kernel": ["rbf", "linear"],  # Include linear kernel for comparison
        "svm__C": np.logspace(-3, 2, 20),  # Logarithmic scale: 0.001 to 100
        "svm__gamma": ["scale", "auto"]
        + list(np.logspace(-4, 0, 15)),  # Include 'scale'/'auto' + log scale
        "svm__class_weight": ["balanced", None],  # Test both balanced and None
    }

    perform_hyperparam_tuning(
        svm_pipeline,
        params,
        "SVM",
        session_validation=False,
        psd=FreqDomain.welch,
        random_search=True,
    )


if __name__ == "__main__":
    train()
