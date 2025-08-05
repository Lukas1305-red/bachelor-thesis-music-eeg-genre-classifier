from code.classification.classification import perform_hyperparam_tuning, FreqDomain
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def train(scaling=True):
    print("Initialize")
    if scaling:
        svm = Pipeline(
            [
                ("scaler", StandardScaler()),
                # (
                #     "lda",
                #     LinearDiscriminantAnalysis(n_components=1),
                # ),
                ("pca", PCA(random_state=1997)),
                ("svm", SVC(random_state=1997)),
            ]
        )
        params = {
            # 'lda__solver': ['lsqr', 'eigen'],
            # 'lda__shrinkage': ['auto', 0.7, 0.8, 0.9],
            "pca__n_components": [40, 50, 60, 70],
            "svm__kernel": ["rbf", "linear"],
            "svm__C": [0.1, 1, 10, 100],
            "svm__gamma": ["scale", "auto", 0.1, 0.01, 0.001],
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
        random_search=False,
        deap=True,
    )
    print("Finish")


if __name__ == "__main__":
    train(scaling=True)
