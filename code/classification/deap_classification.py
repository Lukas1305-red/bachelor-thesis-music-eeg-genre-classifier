from code.classification.classification import perform_subject_based_nested_cv
from code.classification.feature_selector import MRMRSelector

def train():
    print("Initialize")

    params = {
        "selector__k": [2, 5, 8, 10, 15, 20, 30, 50],
        "svm__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        "svm__kernel": ["linear", "rbf", "sigmoid"],
        "svm__gamma": ["scale", "auto", 0.0001, 0.001, 0.01, 0.1, 1, 10],
    }

    print("Training")
    results = perform_subject_based_nested_cv(
        param_grid=params,
        num_subjects=32,
        deap=True,
        n_outer=5,
        n_inner=4,
        n_iter=50,
    )
    print("Finish")

if __name__ == "__main__":
    train()