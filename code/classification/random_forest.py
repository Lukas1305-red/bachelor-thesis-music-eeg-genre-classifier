from code.classification.classification import perform_hyperparam_tuning, FreqDomain
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline


def train():
    rf = RandomForestClassifier(random_state=1997)

    pipeline = Pipeline(
        [("feature_selection", SelectKBest(score_func=f_classif)), ("classifier", rf)]
    )
    params = {
        "feature_selection__k": [50, 100, 200, "all"],
        "n_estimators": randint(200, 800),
        "criterion": ["gini", "entropy"],
        "max_depth": [None, 20, 30, 50],
        "min_samples_split": randint(2, 15),
        "min_samples_leaf": randint(1, 8),
        "max_features": ["sqrt", "log2"],
        "bootstrap": [True],
    }

    perform_hyperparam_tuning(
        pipeline,
        params,
        "Random Forest",
        session_validation=False,
        psd=FreqDomain.welch,
        random_search=True,
    )


if __name__ == "__main__":
    train()
