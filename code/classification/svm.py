from code.classification.classification import perform_hyperparam_tuning, FreqDomain
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier


class FeatureSelectorFromModel(BaseEstimator, TransformerMixin):
    def __init__(self, model=None, top_n=10):
        self.model = model if model is not None else RandomForestClassifier()
        self.top_n = top_n

    def fit(self, X, y=None):
        self.model.fit(X, y)
        self.importances_ = self.model.feature_importances_

        # Get indices of top_n features
        self.top_indices_ = np.argsort(self.importances_)[::-1][: self.top_n]
        return self

    def transform(self, X):
        return X[:, self.top_indices_]


class MRMRSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=10, n_bins=10):
        self.k = k
        self.n_bins = n_bins
        self.selected_indices_ = []

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        n_features = X.shape[1]

        # Step 1: Discretize X for mutual info estimation
        est = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy="uniform")
        X_disc = est.fit_transform(X)

        # Step 2: Compute relevance
        relevance = mutual_info_classif(X, y, discrete_features="auto")

        # Step 3: Compute redundancy matrix
        redundancy = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i + 1, n_features):
                mi = mutual_info_score(X_disc[:, i], X_disc[:, j])
                redundancy[i, j] = redundancy[j, i] = mi

        # Step 4: MRMR selection
        selected = []
        remaining = list(range(n_features))

        for _ in range(min(self.k, n_features)):
            mrmr_scores = []
            for feature in remaining:
                rel = relevance[feature]
                red = (
                    np.mean([redundancy[feature, s] for s in selected])
                    if selected
                    else 0
                )
                score = rel - red
                mrmr_scores.append(score)

            best_idx = np.argmax(mrmr_scores)
            selected_feature = remaining[best_idx]
            selected.append(selected_feature)
            remaining.remove(selected_feature)

        self.selected_indices_ = selected
        return self

    def transform(self, X):
        return np.array(X)[:, self.selected_indices_]

    def get_support(self):
        """Returns mask of selected features for compatibility with other selectors."""
        support = np.zeros((len(self.selected_indices_),), dtype=bool)
        support[self.selected_indices_] = True
        return support


def train(scaling=True):
    print("Initialize")
    if scaling:
        svm = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("selector", MRMRSelector()),
                ("svm", SVC(random_state=1997)),
            ]
        )
        params = {
            "selector__k": [5, 10, 20],
            "svm__C": [0.1, 1, 10, 100],
            "svm__kernel": ["rbf", "linear"],
            "svm__gamma": ["scale", "auto", 0.01, 0.001],
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
