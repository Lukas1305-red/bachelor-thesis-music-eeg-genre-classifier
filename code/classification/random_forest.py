from code.classification.classification import perform_grid_search
from sklearn.ensemble import RandomForestClassifier


def train():
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, None],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"],
    }

    rf = RandomForestClassifier(random_state=1997)

    perform_grid_search(rf, param_grid, "Random Forest", session_validation=False)


if __name__ == "__main__":
    train()
