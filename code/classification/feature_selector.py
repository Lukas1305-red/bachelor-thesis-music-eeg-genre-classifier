from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from mrmr import mrmr_classif


class FeatureSelectorFromModel(BaseEstimator, TransformerMixin):
    def __init__(self, model=None, top_n=10):
        self.model = model if model is not None else RandomForestClassifier()
        self.top_n = top_n

    def fit(self, X, y=None):
        self.model.fit(X, y)
        self.importances_ = self.model.feature_importances_

        # Get indices of top_n features
        self.top_indices_ = np.argsort(self.importances_)[::-1][: self.top_n]
        return self

    def transform(self, X):
        return X[:, self.top_indices_]

    def get_support(self):
        """Returns mask of selected features for compatibility with other selectors."""
        n_features = len(self.importances_)
        support = np.zeros(n_features, dtype=bool)
        support[self.top_indices_] = True
        return support


class MRMRSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=64, relevance="f", redundancy="c", denominator="mean"):
        """
        MRMR feature selector using the mrmr package.

        Parameters:
        -----------
        k : int
            Number of features to select
        relevance : str, default 'f'
            Method for relevance calculation ('f' for F-statistic, 'mic' for MIC)
        redundancy : str, default 'c'
            Method for redundancy calculation ('c' for correlation, 'mic' for MIC)
        denominator : str, default 'mean'
            How to aggregate redundancy ('mean' or 'max')
        """
        self.k = k
        self.relevance = relevance
        self.redundancy = redundancy
        self.denominator = denominator
        self.selected_features_ = []
        self.selected_indices_ = []

    def fit(self, X, y):
        # print(f"Starting MRMR feature selection for {self.k} features...")

        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            # Create column names if not provided
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_df = pd.DataFrame(X, columns=feature_names)
        else:
            X_df = X.copy()

        # Convert y to pandas Series if needed
        if isinstance(y, np.ndarray):
            y_series = pd.Series(y)
        else:
            y_series = y.copy()

        try:
            # Run MRMR selection
            self.selected_features_ = mrmr_classif(
                X=X_df,
                y=y_series,
                K=min(self.k, X_df.shape[1]),  # Don't select more than available
                relevance=self.relevance,
                redundancy=self.redundancy,
                denominator=self.denominator,
                show_progress=False,
            )

            # Get indices for selected features
            if isinstance(X, np.ndarray):
                # Map feature names back to indices
                self.selected_indices_ = [
                    int(name.split("_")[1]) for name in self.selected_features_
                ]
            else:
                # Get indices from DataFrame columns
                self.selected_indices_ = [
                    X_df.columns.get_loc(feat) for feat in self.selected_features_
                ]

        except Exception as e:
            print(f"MRMR selection failed: {e}")
            print("Falling back to top-k features by mutual information...")
            # Fallback to simple mutual info selection
            from sklearn.feature_selection import mutual_info_classif

            mi_scores = mutual_info_classif(X_df, y_series)
            self.selected_indices_ = np.argsort(mi_scores)[::-1][: self.k]
            self.selected_features_ = [X_df.columns[i] for i in self.selected_indices_]

        # print(f"Selected {len(self.selected_features_)} features")
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            return X[:, self.selected_indices_]
        else:
            return X.iloc[:, self.selected_indices_].values

    def get_support(self, indices=False):
        """Returns mask or indices of selected features for compatibility."""
        if hasattr(self, "n_features_in_"):
            n_features = self.n_features_in_
        else:
            # Estimate from selected indices
            n_features = (
                max(self.selected_indices_) + 1 if self.selected_indices_ else 0
            )

        if indices:
            return np.array(self.selected_indices_)
        else:
            support = np.zeros(n_features, dtype=bool)
            support[self.selected_indices_] = True
            return support

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for display."""
        if input_features is None:
            return [f"feature_{i}" for i in self.selected_indices_]
        else:
            return [input_features[i] for i in self.selected_indices_]
