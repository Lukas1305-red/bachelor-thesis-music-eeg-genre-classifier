import pickle
import numpy as np
from code.feature_extraction.feature_extraction import feature_extraction
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def load_deap_dataset():
    labels = load_deap_labels()
    le = LabelEncoder()
    labels_numeric = le.fit_transform(labels)
    print("Loading DEAP EEG signals...")
    eeg_data = load_deap_eeg()
    print(np.shape(eeg_data))
    assert np.shape(eeg_data) == (1280, 32, 8064)

    print("Extract Features from DEAP Dataset...")
    feature_array = []
    keys = None
    for song in eeg_data:
        song_features = feature_extraction(song_data=song)
        if keys is None:
            keys = song_features.keys()
        feature_values = []
        for k in keys:
            feature_values.append(song_features[k])
        feature_array.append(feature_values)
    feature_array_np = np.array(feature_array)
    feature_array_np = feature_array_np[
        :, np.all(np.isfinite(feature_array_np), axis=0)
    ]
    print(np.shape(feature_array_np))

    # Perform comprehensive feature analysis
    print("\n" + "=" * 50)
    print("STARTING COMPREHENSIVE FEATURE ANALYSIS")
    print("=" * 50)

    perform_feature_analysis(feature_array_np, labels_numeric, labels, list(keys))

    return feature_array_np, labels_numeric


def load_deap_eeg():
    eeg_data = load_all_subjects()
    eeg_data_np = np.array(eeg_data)
    eeg_data_reshaped = eeg_data_np.reshape(1280, 40, 8064)  # no assertion necessary
    eeg_data_reshaped_eeg = eeg_data_reshaped[:, :32, :]  # only keep EEG channels
    return eeg_data_reshaped_eeg


# TODO: Write in .csv or pickle
def load_deap_labels():
    deap_labels = [
        "POP",
        "ROCK",
        "POP",
        "POP",
        "ROCK",
        "ROCK",
        "POP",
        "POP",
        "POP",
        "ROCK",
        "POP",
        "POP",
        "POP",
        "POP",
        "POP",
        "POP",
        "POP",
        "POP",
        "POP",
        "POP",
        "ROCK",
        "ROCK",
        "POP",
        "POP",
        "POP",
        "POP",
        "POP",
        "POP",
        "POP",
        "POP",
        "ROCK",
        "ROCK",
        "ROCK",
        "ROCK",
        "ROCK",
        "ROCK",
        "ROCK",
        "ROCK",
        "ROCK",
        "ROCK",
    ]
    print(len(deap_labels))
    labels = []
    # iterate over all subjects
    for i in range(32):
        labels.extend(deap_labels)

    print(np.shape(labels))
    assert np.shape(labels) == (1280,)
    return labels


def load_all_subjects():
    all_subjects = []
    for i in range(32):
        subject = f"s{str(i + 1).zfill(2)}"
        subject_data = load_subject(subject)
        all_subjects.append(subject_data)
    assert np.shape(all_subjects) == (32, 40, 40, 8064)
    return all_subjects


def load_subject(subject: str):
    file_path = f"data/deap/deap/subjects/{subject}.dat"
    subject_data = pickle.load(open(file_path, "rb"), encoding="latin1")
    subject_signal_data = np.array(subject_data["data"])
    assert np.shape(subject_signal_data) == (40, 40, 8064)
    return subject_signal_data


###########################


##########################################


def perform_feature_analysis(features, labels_numeric, labels_original, feature_names):
    """
    Comprehensive feature analysis function that creates all relevant visualizations and analysis.

    Parameters:
    - features: numpy array of extracted features (n_samples, n_features)
    - labels_numeric: encoded labels (0, 1, ...)
    - labels_original: original string labels for interpretation
    - feature_names: list of feature names
    """

    # Create output directory for plots
    import os

    os.makedirs("feature_analysis_plots", exist_ok=True)

    print(f"Analyzing {features.shape[0]} samples with {features.shape[1]} features")
    print(f"Classes: {np.unique(labels_original)}")

    # 1. Basic statistical analysis
    print("\n1. Basic Statistical Analysis...")
    basic_stats_analysis(features, labels_numeric, feature_names)

    # 2. Feature distribution analysis
    print("\n2. Feature Distribution Analysis...")
    feature_distribution_analysis(
        features, labels_numeric, labels_original, feature_names
    )

    # 3. Correlation analysis
    print("\n3. Correlation Analysis...")
    correlation_analysis(features, feature_names)

    # 4. Dimensionality reduction visualizations
    print("\n4. Dimensionality Reduction Analysis...")
    dimensionality_reduction_analysis(features, labels_numeric, labels_original)

    # 5. Feature importance analysis
    print("\n5. Feature Importance Analysis...")
    feature_importance_analysis(features, labels_numeric, feature_names)

    # 6. Class separability analysis
    print("\n6. Class Separability Analysis...")
    class_separability_analysis(features, labels_numeric, labels_original)

    # 7. Outlier detection
    print("\n7. Outlier Detection Analysis...")
    outlier_detection_analysis(features, labels_numeric, labels_original)

    print("\n" + "=" * 50)
    print("FEATURE ANALYSIS COMPLETED")
    print("All plots saved in 'feature_analysis_plots/' directory")
    print("=" * 50)


def basic_stats_analysis(features, labels_numeric, feature_names):
    """Basic statistical analysis of features."""

    # Overall statistics
    df_stats = pd.DataFrame(
        {
            "Feature": feature_names[
                : features.shape[1]
            ],  # Handle case where we have fewer features than names
            "Mean": np.mean(features, axis=0),
            "Std": np.std(features, axis=0),
            "Min": np.min(features, axis=0),
            "Max": np.max(features, axis=0),
            "Skewness": stats.skew(features, axis=0),
            "Kurtosis": stats.kurtosis(features, axis=0),
        }
    )

    print("Feature Statistics Summary:")
    print(df_stats.describe())

    # Save statistics
    df_stats.to_csv("feature_analysis_plots/feature_statistics.csv", index=False)

    # Plot feature statistics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    axes[0, 0].hist(np.mean(features, axis=0), bins=20, alpha=0.7)
    axes[0, 0].set_title("Distribution of Feature Means")
    axes[0, 0].set_xlabel("Mean Value")

    axes[0, 1].hist(np.std(features, axis=0), bins=20, alpha=0.7)
    axes[0, 1].set_title("Distribution of Feature Standard Deviations")
    axes[0, 1].set_xlabel("Std Value")

    axes[1, 0].hist(stats.skew(features, axis=0), bins=20, alpha=0.7)
    axes[1, 0].set_title("Distribution of Feature Skewness")
    axes[1, 0].set_xlabel("Skewness")

    axes[1, 1].hist(stats.kurtosis(features, axis=0), bins=20, alpha=0.7)
    axes[1, 1].set_title("Distribution of Feature Kurtosis")
    axes[1, 1].set_xlabel("Kurtosis")

    plt.tight_layout()
    plt.savefig(
        "feature_analysis_plots/basic_statistics.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def feature_distribution_analysis(
    features, labels_numeric, labels_original, feature_names
):
    """Analyze feature distributions across classes."""

    # Select top 12 most variable features for visualization
    feature_vars = np.var(features, axis=0)
    top_features_idx = np.argsort(feature_vars)[-12:]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.ravel()

    unique_labels = np.unique(labels_original)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))

    for i, feature_idx in enumerate(top_features_idx):
        for j, label in enumerate(unique_labels):
            mask = labels_numeric == j
            axes[i].hist(
                features[mask, feature_idx],
                alpha=0.6,
                label=f"{label}",
                color=colors[j],
                bins=20,
            )

        feature_name = (
            feature_names[feature_idx]
            if feature_idx < len(feature_names)
            else f"Feature_{feature_idx}"
        )
        axes[i].set_title(f"{feature_name}")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(
        "feature_analysis_plots/feature_distributions.png", dpi=300, bbox_inches="tight"
    )
    plt.close()


def correlation_analysis(features, feature_names):
    """Analyze correlations between features."""

    # Compute correlation matrix
    corr_matrix = np.corrcoef(features.T)

    # Plot correlation matrix
    plt.figure(figsize=(12, 10))

    # Use a subset of features if too many
    if features.shape[1] > 50:
        # Select every nth feature to make visualization readable
        step = features.shape[1] // 50
        selected_indices = range(0, features.shape[1], step)
        corr_subset = corr_matrix[np.ix_(selected_indices, selected_indices)]
        selected_names = [
            feature_names[i] if i < len(feature_names) else f"F{i}"
            for i in selected_indices
        ]
    else:
        corr_subset = corr_matrix
        selected_names = feature_names[: features.shape[1]]

    sns.heatmap(
        corr_subset,
        annot=False,
        cmap="coolwarm",
        center=0,
        xticklabels=selected_names,
        yticklabels=selected_names,
    )
    plt.title("Feature Correlation Matrix")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        "feature_analysis_plots/correlation_matrix.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Find highly correlated feature pairs
    high_corr_pairs = []
    threshold = 0.8
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            if abs(corr_matrix[i, j]) > threshold:
                high_corr_pairs.append((i, j, corr_matrix[i, j]))

    print(
        f"Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > {threshold})"
    )

    if high_corr_pairs:
        df_corr = pd.DataFrame(
            high_corr_pairs, columns=["Feature_1_Idx", "Feature_2_Idx", "Correlation"]
        )
        df_corr.to_csv("feature_analysis_plots/high_correlations.csv", index=False)


def dimensionality_reduction_analysis(features, labels_numeric, labels_original):
    """Perform dimensionality reduction and visualization."""

    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    unique_labels = np.unique(labels_original)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)

    for i, label in enumerate(unique_labels):
        mask = labels_numeric == i
        axes[0, 0].scatter(
            features_pca[mask, 0],
            features_pca[mask, 1],
            c=[colors[i]],
            label=label,
            alpha=0.6,
        )
    axes[0, 0].set_title(
        f"PCA (Explained Variance: {pca.explained_variance_ratio_.sum():.3f})"
    )
    axes[0, 0].legend()
    axes[0, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.3f})")
    axes[0, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.3f})")

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_tsne = tsne.fit_transform(features_scaled)

    for i, label in enumerate(unique_labels):
        mask = labels_numeric == i
        axes[0, 1].scatter(
            features_tsne[mask, 0],
            features_tsne[mask, 1],
            c=[colors[i]],
            label=label,
            alpha=0.6,
        )
    axes[0, 1].set_title("t-SNE")
    axes[0, 1].legend()

    # LDA
    if len(unique_labels) > 1:
        lda = LinearDiscriminantAnalysis(n_components=min(2, len(unique_labels) - 1))
        features_lda = lda.fit_transform(features_scaled, labels_numeric)

        if features_lda.shape[1] == 2:
            for i, label in enumerate(unique_labels):
                mask = labels_numeric == i
                axes[1, 0].scatter(
                    features_lda[mask, 0],
                    features_lda[mask, 1],
                    c=[colors[i]],
                    label=label,
                    alpha=0.6,
                )
            axes[1, 0].set_xlabel("LD1")
            axes[1, 0].set_ylabel("LD2")
        else:
            # 1D LDA
            for i, label in enumerate(unique_labels):
                mask = labels_numeric == i
                axes[1, 0].hist(
                    features_lda[mask, 0],
                    alpha=0.6,
                    label=label,
                    color=colors[i],
                    bins=20,
                )
            axes[1, 0].set_xlabel("LD1")

        axes[1, 0].set_title("Linear Discriminant Analysis")
        axes[1, 0].legend()

    # PCA explained variance plot
    pca_full = PCA()
    pca_full.fit(features_scaled)
    cumsum_var = np.cumsum(pca_full.explained_variance_ratio_)

    axes[1, 1].plot(
        range(1, min(51, len(cumsum_var) + 1)), cumsum_var[:50], "bo-", markersize=4
    )
    axes[1, 1].axhline(y=0.95, color="r", linestyle="--", label="95% Variance")
    axes[1, 1].set_xlabel("Number of Components")
    axes[1, 1].set_ylabel("Cumulative Explained Variance")
    axes[1, 1].set_title("PCA Explained Variance")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig(
        "feature_analysis_plots/dimensionality_reduction.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def feature_importance_analysis(features, labels_numeric, feature_names):
    """Analyze feature importance using various methods."""

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import f_classif, mutual_info_classif

    # Random Forest feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features, labels_numeric)
    rf_importance = rf.feature_importances_

    # ANOVA F-test
    f_scores, p_values = f_classif(features, labels_numeric)

    # Mutual Information
    mi_scores = mutual_info_classif(features, labels_numeric, random_state=42)

    # Create comparison plot
    n_top = min(20, features.shape[1])  # Show top 20 features

    # Get top features for each method
    top_rf_idx = np.argsort(rf_importance)[-n_top:]
    top_f_idx = np.argsort(f_scores)[-n_top:]
    top_mi_idx = np.argsort(mi_scores)[-n_top:]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Random Forest
    axes[0].barh(range(n_top), rf_importance[top_rf_idx])
    axes[0].set_yticks(range(n_top))
    axes[0].set_yticklabels(
        [feature_names[i] if i < len(feature_names) else f"F{i}" for i in top_rf_idx]
    )
    axes[0].set_title("Random Forest Feature Importance")
    axes[0].set_xlabel("Importance")

    # ANOVA F-test
    axes[1].barh(range(n_top), f_scores[top_f_idx])
    axes[1].set_yticks(range(n_top))
    axes[1].set_yticklabels(
        [feature_names[i] if i < len(feature_names) else f"F{i}" for i in top_f_idx]
    )
    axes[1].set_title("ANOVA F-Score")
    axes[1].set_xlabel("F-Score")

    # Mutual Information
    axes[2].barh(range(n_top), mi_scores[top_mi_idx])
    axes[2].set_yticks(range(n_top))
    axes[2].set_yticklabels(
        [feature_names[i] if i < len(feature_names) else f"F{i}" for i in top_mi_idx]
    )
    axes[2].set_title("Mutual Information")
    axes[2].set_xlabel("MI Score")

    plt.tight_layout()
    plt.savefig(
        "feature_analysis_plots/feature_importance.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Save importance scores
    importance_df = pd.DataFrame(
        {
            "Feature": [
                feature_names[i] if i < len(feature_names) else f"Feature_{i}"
                for i in range(features.shape[1])
            ],
            "RF_Importance": rf_importance,
            "F_Score": f_scores,
            "MI_Score": mi_scores,
            "P_Value": p_values,
        }
    )
    importance_df.to_csv(
        "feature_analysis_plots/feature_importance_scores.csv", index=False
    )


def class_separability_analysis(features, labels_numeric, labels_original):
    """Analyze class separability."""

    unique_labels = np.unique(labels_original)
    n_classes = len(unique_labels)

    # Calculate class centroids
    centroids = []
    for i in range(n_classes):
        mask = labels_numeric == i
        centroid = np.mean(features[mask], axis=0)
        centroids.append(centroid)
    centroids = np.array(centroids)

    # Calculate pairwise distances between centroids
    from scipy.spatial.distance import pdist, squareform

    centroid_distances = squareform(pdist(centroids))

    # Plot class centroid distances
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        centroid_distances,
        annot=True,
        cmap="viridis",
        xticklabels=unique_labels,
        yticklabels=unique_labels,
    )
    plt.title("Pairwise Distances Between Class Centroids")
    plt.tight_layout()
    plt.savefig(
        "feature_analysis_plots/class_separability.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Calculate within-class and between-class scatter
    overall_mean = np.mean(features, axis=0)

    # Within-class scatter
    within_class_scatter = 0
    for i in range(n_classes):
        mask = labels_numeric == i
        class_data = features[mask]
        class_mean = centroids[i]
        within_class_scatter += np.sum((class_data - class_mean) ** 2)

    # Between-class scatter
    between_class_scatter = 0
    for i in range(n_classes):
        mask = labels_numeric == i
        n_samples = np.sum(mask)
        class_mean = centroids[i]
        between_class_scatter += n_samples * np.sum((class_mean - overall_mean) ** 2)

    # Fisher's ratio (between-class / within-class)
    fisher_ratio = between_class_scatter / within_class_scatter

    print("Class Separability Analysis:")
    print(f"  Within-class scatter: {within_class_scatter:.4f}")
    print(f"  Between-class scatter: {between_class_scatter:.4f}")
    print(f"  Fisher's ratio: {fisher_ratio:.4f}")


def outlier_detection_analysis(features, labels_numeric, labels_original):
    """Detect and analyze outliers."""

    from sklearn.ensemble import IsolationForest
    from sklearn.covariance import EllipticEnvelope

    # Standardize features for outlier detection
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Isolation Forest
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers_iso = iso_forest.fit_predict(features_scaled)

    # Elliptic Envelope
    elliptic = EllipticEnvelope(contamination=0.1, random_state=42)
    outliers_elliptic = elliptic.fit_predict(features_scaled)

    # Statistical outliers (Z-score > 3)
    z_scores = np.abs(stats.zscore(features_scaled, axis=0))
    outliers_zscore = np.any(z_scores > 3, axis=1)

    print("Outlier Detection Results:")
    print(f"  Isolation Forest: {np.sum(outliers_iso == -1)} outliers")
    print(f"  Elliptic Envelope: {np.sum(outliers_elliptic == -1)} outliers")
    print(f"  Z-score (>3): {np.sum(outliers_zscore)} outliers")

    # Visualize outliers using PCA
    pca = PCA(n_components=2)
    features_pca = pca.fit_transform(features_scaled)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    unique_labels = np.unique(labels_original)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    # Plot normal vs outliers for each method
    methods = [
        ("Isolation Forest", outliers_iso == -1),
        ("Elliptic Envelope", outliers_elliptic == -1),
        ("Z-score", outliers_zscore),
    ]

    for idx, (method_name, outlier_mask) in enumerate(methods):
        # Plot normal points
        normal_mask = ~outlier_mask
        for i, label in enumerate(unique_labels):
            class_normal = normal_mask & (labels_numeric == i)
            axes[idx].scatter(
                features_pca[class_normal, 0],
                features_pca[class_normal, 1],
                c=[colors[i]],
                alpha=0.6,
                s=20,
                label=f"{label} (normal)",
            )

        # Plot outliers
        axes[idx].scatter(
            features_pca[outlier_mask, 0],
            features_pca[outlier_mask, 1],
            c="red",
            marker="x",
            s=50,
            label="Outliers",
        )

        axes[idx].set_title(f"{method_name}\n({np.sum(outlier_mask)} outliers)")
        axes[idx].legend()
        axes[idx].set_xlabel("PC1")
        axes[idx].set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(
        "feature_analysis_plots/outlier_detection.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
