from code.classification.classification import perform_hyperparam_tuning, FreqDomain
from sklearn.ensemble import RandomForestClassifier


def train():
    rf = RandomForestClassifier(random_state=1997)

    # Define steps in ranges, e.g. n_estimators 50
    params = {
        "n_estimators": list(range(50, 200, 25)),  # Reduced range
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4, 5],  # Much shallower trees
        "min_samples_split": list(range(20, 100, 10)),  # Higher minimum splits
        "min_samples_leaf": list(range(15, 50, 5)),  # Higher minimum leaves
        "max_features": ["sqrt", "log2", 0.1, 0.2],  # Added fraction options
        "bootstrap": [True],
    }

    perform_hyperparam_tuning(
        rf,
        params,
        "Random Forest",
        session_validation=False,
        psd=FreqDomain.welch,
        random_search=True,
    )


if __name__ == "__main__":
    train()
