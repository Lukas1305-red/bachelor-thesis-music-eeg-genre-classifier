from code.classification.classification import perform_hyperparam_tuning, FreqDomain
from code.classification.feature_selector import MRMRSelector
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier


def train(scaling=True):
    clf = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        alpha=0.01,
        learning_rate_init=0.001,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=1997,
        solver="adam",
    )

    mlp = Pipeline(
        [("scaler", StandardScaler()), ("selector", MRMRSelector(k=50)), ("mlp", clf)]
    )

    params = {
        "selector__k": [150, 200, 300, 500],  # Focus around the best value
        "mlp__hidden_layer_sizes": [
            (32,),
            (64,),
            (128,),  # Try larger single layers
            (64, 32),
            (128, 64),  # Larger two-layer networks
            (64, 32, 16),  # Three layers with tapering
            (128, 64, 32),
        ],
        "mlp__alpha": [0.001, 0.01, 0.05, 0.1],  # Fine-tune regularization
        "mlp__learning_rate_init": [0.0005, 0.001, 0.005, 0.01],
        "mlp__solver": ["adam", "lbfgs"],  # Try both optimizers
    }

    print("Training")
    perform_hyperparam_tuning(
        mlp,
        params,
        "MLP_DEAP_BINARY_2S_MRMR",
        session_validation=False,
        psd=FreqDomain.welch,
        random_search=False,
        deap=True,
    )
    print("Finish")


if __name__ == "__main__":
    train(scaling=True)
