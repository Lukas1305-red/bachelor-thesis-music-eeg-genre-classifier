from code.classification.classification import perform_hyperparam_tuning, FreqDomain
from sklearn.ensemble import RandomForestClassifier


def train():
    params = {
        "n_estimators": range(10, 101),
        "criterion": ["gini", "entropy"],
        "max_depth": range(2, 51),
        "min_samples_split": range(2, 11),
        "min_samples_leaf": range(1, 11),
        "max_features": ["sqrt", "log2", None],
        "bootstrap": [True, False],
    }

    rf = RandomForestClassifier(random_state=1997)

    perform_hyperparam_tuning(
        rf,
        params,
        "Random Forest",
        session_validation=True,
        psd=FreqDomain.welch,
        random_search=True,
    )


if __name__ == "__main__":
    train()
