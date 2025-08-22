import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from code.classification.feature_selector import MRMRSelector
from code.classification.data import load_data, FreqDomain

DATA_PATH = "data"
MATRIX_PATH = f"{DATA_PATH}/results/headset/confusion_matrix"
FILE_PATH = f"{DATA_PATH}/svm_grid_summary_headset.csv"


def create_matrices():
    df = pd.read_csv(FILE_PATH)
    global_predictions = []
    global_labels = []
    X, y, le = load_data(psd_to_keep=FreqDomain.welch, ret_label_encoder=True)
    offset = 0
    for i, row in df.iterrows():
        print(f"=======Session {i}=========")
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
        all_labels = []
        all_predictions = []

        first_idx = i * 50 + offset
        if i == 10:
            last_idx = first_idx + 49
            offset -= 1
        else:
            last_idx = first_idx + 50

        X_sess = X[first_idx:last_idx, :]
        y_sess = y[first_idx:last_idx]
        for j, (train_index, test_index) in enumerate(cv.split(X_sess, y_sess)):
            print(f"======Fold{j}=========")
            X_train, X_test = X_sess[train_index], X_sess[test_index]
            y_train, y_test = y_sess[train_index], y_sess[test_index]

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
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [3, 1]}
        )

        # Plot confusion matrix
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix, display_labels=classes
        )
        display.plot(values_format=".1f", ax=ax1, im_kw={"vmin": 0, "vmax": 1})

        # Create legend in second subplot
        ax2.axis("off")  # Hide axes
        legend_text = "\n".join(
            [f"{i}: {class_name}" for i, class_name in enumerate(classes)]
        )
        ax2.text(
            0.1,
            0.9,
            "Class Labels:\n\n" + legend_text,
            transform=ax2.transAxes,
            fontsize=12,
            verticalalignment="top",
        )

        plt.tight_layout()
        plt.savefig(
            f"{MATRIX_PATH}/confusion_matrix_sess_{i}", dpi=300, bbox_inches="tight"
        )
        plt.close()

    classes = le.classes_
    matrix = confusion_matrix(
        global_labels, global_predictions, labels=classes, normalize="true"
    )
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [3, 1]}
    )

    # Plot confusion matrix
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classes)
    display.plot(values_format=".2f", ax=ax1, im_kw={"vmin": 0, "vmax": 1})

    # Create legend in second subplot
    ax2.axis("off")  # Hide axes
    legend_text = "\n".join(
        [f"{i}: {class_name}" for i, class_name in enumerate(classes)]
    )
    ax2.text(
        0.1,
        0.9,
        "Class Labels:\n\n" + legend_text,
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig(
        f"{MATRIX_PATH}/00_global_confusion_matrix_headset",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    create_matrices()
