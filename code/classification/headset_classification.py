from code.classification.classification import perform_hyperparam_tuning, FreqDomain
from code.classification.feature_selector import MRMRSelector
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline


def train(scaling=True):
    print("Initialize")
    if scaling:
        svm = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("selector", MRMRSelector()),
                ("svm", SVC(random_state=1997, class_weight="balanced")),
            ]
        )
        params = {
            "selector__k": [
                2,
                5,
                8,
                10,
                15,
                20,
                30,
                50,
            ],  # add larger values for richer features
            "svm__C": [
                0.001,
                0.01,
                0.1,
                1,
                10,
                100,
                1000,
            ],  # add finer granularity + higher regularization
            "svm__kernel": [
                "linear",
                "rbf",
                "sigmoid",
            ],  # add poly & sigmoid to catch nonlinearities
            "svm__gamma": [
                "scale",
                "auto",
                0.0001,
                0.001,
                0.01,
                0.1,
                1,
                10,
            ],  # add wider gamma search
        }
    else:
        svm = SVC(random_state=1997)
        params = {
            "kernel": ["rbf", "linear"],
            "C": [0.001, 0.1, 1, 10, 100],
            "gamma": ["scale", 0.1, 0.01, 0.001],
            "decision_function_shape": ["ovr"],
        }

    print("Training")
    for i in range(20):
        print(f"========= TUNING SESSION {i} START ============")
        perform_hyperparam_tuning(
            svm,
            params,
            f"HEADSET_SESS_{i}V2",
            session_validation=False,
            psd=FreqDomain.welch,
            random_search=False,
            deap=False,
            subj=i,
        )
        print(f"========= TUNING SESSION {i} END ============")
    print("Finish")


if __name__ == "__main__":
    train(scaling=True)
