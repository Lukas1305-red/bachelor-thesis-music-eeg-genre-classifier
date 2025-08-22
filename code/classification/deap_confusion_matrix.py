import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from code.classification.feature_selector import MRMRSelector
from code.classification.deap_data import load_deap_dataset

DATA_PATH = "data"
MATRIX_PATH = f"{DATA_PATH}/results/deap/confusion_matrix"
FILE_PATH = f"{DATA_PATH}/svm_gridsearch_results.csv"


def create_matrices():
    df = pd.read_csv(FILE_PATH)
    global_labels = []
    global_predictions = []
    for i, row in df.iterrows():
        print(f"=======Subject {i}=========")
        # Get the hyperparams from the dataframe
        k_features = row["k_features"]
        C_param = float(row["C"])

        gamma_value = row["gamma"]
        try:
            gamma_param = float(gamma_value)
        except ValueError:
            gamma_param = gamma_value  # "scale" or "auto"
        kernel_param = row["kernel"]

        # Define ML pipeline
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("selector", MRMRSelector(k=k_features)),
                (
                    "svm",
                    SVC(
                        random_state=1997,
                        class_weight="balanced",
                        C=C_param,
                        kernel=kernel_param,
                        gamma=gamma_param,
                    ),
                ),
            ]
        )

        # Initialize CV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1997)

        # Load subject
        X, y, le = load_deap_dataset(subject=i, ret_label_encoder=True)
        all_labels = []
        all_predictions = []
        for j, (train_index, test_index) in enumerate(cv.split(X, y)):
            print(f"======Fold{j}=========")
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            all_labels.extend(y_test)
            all_predictions.extend(y_pred)

        all_labels_words = le.inverse_transform(all_labels)
        all_predictions_words = le.inverse_transform(all_predictions)
        global_labels.extend(all_labels_words)
        global_predictions.extend(all_predictions_words)
        classes = le.classes_
        matrix = confusion_matrix(
            all_labels_words, all_predictions_words, labels=classes, normalize="true"
        )
        display = ConfusionMatrixDisplay(matrix, display_labels=classes)
        display.plot(values_format=".1f", im_kw={"vmin": 0, "vmax": 1})
        plt.savefig(f"{MATRIX_PATH}/confusion_matrix_subj_{i}")
        plt.close()

    classes = le.classes_
    matrix = confusion_matrix(
        global_labels, global_predictions, labels=classes, normalize="true"
    )
    display.plot(values_format=".2f", im_kw={"vmin": 0, "vmax": 1})
    plt.savefig(f"{MATRIX_PATH}/00_global_confusion_matrix")
    plt.close()


if __name__ == "__main__":
    create_matrices()
