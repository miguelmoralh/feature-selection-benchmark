from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.model_selection import BaseCrossValidator
from sklearn.base import clone
import shap
import gc
import warnings
import traceback
import time
from multiprocessing import Process, Queue
from typing import Optional

class ShapFeatureImportanceSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector that uses SHAP (SHapley Additive exPlanations) values for feature selection.
    
    This class implements feature selection using SHAP values calculated through cross-validation.
    It supports both regression and classification tasks and works with any tree-based model
    that is compatible with the SHAP library. The selector calculates feature importance scores
    using SHAP values and selects features based on a threshold.

    Parameters
    ----------
    model : BaseEstimator
        A scikit-learn compatible model (preferably tree-based) that will be used to calculate SHAP values.
    cv : BaseCrossValidator, optional
        Cross-validation splitter. If None, a default splitter will be used.
    threshold : float, default=0.01
        The threshold value for feature selection. Features with normalized SHAP values
        below this threshold will be excluded.
    task : str, default='regression'
        Type of machine learning task. Either 'regression' or 'classification'.
    operation_timeout : int, default=300
        Maximum time in seconds allowed for processing each fold.

    Attributes
    ----------
    selected_features_ : list
        List of selected feature names after fitting.
    feature_importances_ : dict
        Dictionary containing normalized importance scores for each feature.
    chunk_size_ : int
        Calculated optimal chunk size for SHAP value computation.
    max_samples_ : int
        Calculated optimal number of samples to use for SHAP calculation.

    Examples
    --------
    >>> from sklearn.model_selection import KFold
    >>> from catboost import CatBoostRegressor
    >>> selector = ShapFeatureImportanceSelector(
    ...     model=CatBoostRegressor(),
    ...     cv=KFold(n_splits=3),
    ...     threshold=0.01,
    ...     task='regression'
    ... )
    >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        model: BaseEstimator,
        cv: Optional[BaseCrossValidator] = None,
        threshold: float = 0.01,
        task: str = 'regression',
        operation_timeout: int = 300
    ):
        self.model = model
        self.cv = cv
        self.threshold = threshold
        self.task = task
        self.operation_timeout = operation_timeout
        self.selected_features_ = None
        self.chunk_size_ = None
        self.max_samples_ = None
        
    def _calculate_optimal_parameters(self, X):
        """
        Calculate optimal chunk size and maximum samples based on dataset characteristics.

        This method determines the optimal parameters for SHAP value calculation based on
        the size of the input dataset. It uses different strategies for small, medium,
        and large datasets to balance computation time and memory usage.

        Parameters
        ----------
        X : pd.DataFrame
            The input feature matrix.

        Returns
        -------
        tuple
            A tuple containing (chunk_size, max_samples) where:
            - chunk_size: Optimal size for processing data in chunks
            - max_samples: Maximum number of samples to use for SHAP calculation
        """
        n_samples, n_features = X.shape
        
        # Calculate optimal chunk size based on dataset size
        if n_samples < 1000:
            chunk_size = min(20, max(5, n_samples // 10))
        else:
            chunk_size = min(100, max(20, n_samples // 50))
            
        # Calculate optimal sample size based on dataset size
        if n_samples < 200:
            max_samples = n_samples
        elif n_samples < 1000:
            max_samples = min(500, max(100, n_samples // 2))
        else:
            max_samples = 1000
            
        return chunk_size, max_samples

    def _sample_data(self, X, y=None, max_samples=None):
        """
        Sample data adaptively based on dataset size and task type.

        This method implements an adaptive sampling strategy that takes into account
        the type of task (classification/regression) and the dataset size. For
        classification tasks, it attempts to maintain class distribution through
        stratified sampling when possible.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to sample from.
        y : array-like, optional
            Target vector for stratified sampling in classification tasks.
        max_samples : int, optional
            Maximum number of samples to return.

        Returns
        -------
        tuple
            (sampled_X, sampled_y) containing the sampled data.
        """
        if max_samples and len(X) > max_samples:
            if self.task == 'classification' and y is not None:
                unique_classes = np.unique(y)
                if len(unique_classes) <= 10:  # Only stratify for reasonable number of classes
                    from sklearn.model_selection import StratifiedShuffleSplit
                    sss = StratifiedShuffleSplit(n_splits=1, train_size=max_samples)
                    indices = next(sss.split(X, y))[0]
                else:
                    indices = np.random.choice(len(X), max_samples, replace=False)
            else:
                indices = np.random.choice(len(X), max_samples, replace=False)
            return X.iloc[indices], y[indices] if y is not None else None
        return X, y

    def process_fold(self, fold_idx, train_idx, val_idx, X, y, chunk_size, max_samples, queue):
        """
        Process a single cross-validation fold to calculate SHAP values.

        This method handles the computation of SHAP values for a single fold in the
        cross-validation process. It fits the model on training data, creates a SHAP
        explainer, and calculates SHAP values in chunks to manage memory usage.

        Parameters
        ----------
        fold_idx : int
            Index of the current fold.
        train_idx : array-like
            Indices for training data.
        val_idx : array-like
            Indices for validation data.
        X : pd.DataFrame
            Complete feature matrix.
        y : array-like
            Complete target vector.
        chunk_size : int
            Size of chunks for processing.
        max_samples : int
            Maximum number of samples to use.
        queue : multiprocessing.Queue
            Queue for returning results to the main process.
        """
        try:
            print(f"\nProcessing fold {fold_idx}")
            
            # Prepare data
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
            X_val = X.iloc[val_idx]
            
            # Sample validation data using adaptive parameters
            X_val, _ = self._sample_data(X_val, max_samples=max_samples)
            
            gc.collect()
            
            print(f"Fitting model on {len(X_train)} samples...")
            model = clone(self.model)
            model.fit(X_train, y_train)
            
            print("Creating explainer...")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                explainer = shap.TreeExplainer(
                    model,
                    feature_perturbation="interventional"
                )
            
            print(f"Processing {len(X_val)} samples in chunks of {chunk_size}...")
            shap_values = []
            
            # Process data in chunks to manage memory
            for i in range(0, len(X_val), chunk_size):
                chunk = X_val.iloc[i:i + chunk_size]
                chunk_shap = explainer(chunk)
                
                if isinstance(chunk_shap, shap.Explanation):
                    values = np.abs(chunk_shap.values)
                    if len(values.shape) == 3:
                        values = np.mean(values, axis=2)
                else:
                    values = np.abs(chunk_shap)
                
                shap_values.append(values)
                del chunk_shap, values
                gc.collect()
            
            feature_importances = np.mean(np.vstack(shap_values), axis=0)
            importance_dict = dict(zip(X.columns, feature_importances))
            
            print(f"Fold {fold_idx} completed successfully")
            queue.put(('success', importance_dict))
            
        except Exception as e:
            print(f"Error in fold {fold_idx}:")
            print(traceback.format_exc())
            queue.put(('error', str(e)))
        finally:
            gc.collect()

    def fit(self, X, y):
        """
        Fit the feature selector to the data.

        This method implements the main feature selection process:
        1. Calculates optimal parameters for the dataset
        2. Runs cross-validation to compute SHAP values
        3. Aggregates results across folds
        4. Selects features based on importance threshold

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to fit the selector on.
        y : array-like
            Target vector.

        Returns
        -------
        self
            The fitted selector instance.
        """
        try:
            # Calculate optimal parameters
            self.chunk_size_, self.max_samples_ = self._calculate_optimal_parameters(X)
            print(f"\nDataset shape: {X.shape}")
            print(f"Using chunk_size={self.chunk_size_}, max_samples={self.max_samples_}")
            
            feature_importances = {feature: [] for feature in X.columns}
            successful_folds = 0
            total_folds = self.cv.get_n_splits(X, y)
            
            print(f"\nStarting cross-validation with {total_folds} folds...")
            
            # Process each fold
            for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X, y), 1):
                queue = Queue()
                process = Process(
                    target=self.process_fold,
                    args=(fold_idx, train_idx, val_idx, X, y, self.chunk_size_, 
                         self.max_samples_, queue)
                )
                
                process.start()
                
                # Monitor process completion
                completed = False
                start_time = time.time()
                
                while time.time() - start_time < self.operation_timeout:
                    process.join(0.1)
                    
                    if not queue.empty():
                        status, result = queue.get()
                        if status == 'success':
                            successful_folds += 1
                            for feature, importance in result.items():
                                feature_importances[feature].append(importance)
                            completed = True
                            break
                        elif status == 'error':
                            print(f"Fold {fold_idx} failed: {result}")
                            break
                    
                    if not process.is_alive():
                        break
                
                # Handle process cleanup
                if not completed and process.is_alive():
                    print(f"Fold {fold_idx} timed out after {self.operation_timeout} seconds")
                    process.terminate()
                    process.join(1)
                    if process.is_alive():
                        process.kill()
                
                # Clear queue
                while not queue.empty():
                    _ = queue.get()
                
                gc.collect()
            
            if successful_folds == 0:
                warnings.warn("No successful folds. Using all features.")
                self.selected_features_ = list(X.columns)
                return self

            # Calculate and normalize feature importances
            mean_importances = {
                feature: np.mean(scores) if scores else 0
                for feature, scores in feature_importances.items()
            }
            
            total_importance = sum(mean_importances.values())
            if total_importance > 0:
                self.feature_importances_ = {
                    feature: importance / total_importance
                    for feature, importance in mean_importances.items()
                }
            else:
                self.feature_importances_ = {
                    feature: 1.0 / len(mean_importances)
                    for feature in mean_importances.keys()
                }
            
            # Select features based on threshold
            self.selected_features_ = [
                feature for feature, importance in self.feature_importances_.items()
                if importance >= self.threshold
            ]
            
            if not self.selected_features_:
                warnings.warn("No features selected. Using all features.")
                self.selected_features_ = list(X.columns)
            
            print(f"\nFeature selection completed:")
            print(f"- Started with {len(X.columns)} features")
            print(f"- Removed {len(X.columns) - len(self.selected_features_)} features")
            print(f"- Retained {len(self.selected_features_)} features")
            
            return self
            
        except Exception as e:
            print("Error in fit method:")
            print(traceback.format_exc())
            self.selected_features_ = list(X.columns)
            return self
        finally:
            gc.collect()
    
    def transform(self, X):
        """
        Transform the data by selecting only the important features.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to transform.

        Returns
        -------
        pd.DataFrame
            Transformed feature matrix containing only the selected features.
        """
        if self.selected_features_ is None:
            return X
        return X[self.selected_features_]

    def fit_transform(self, X, y):
        """
        Fit the selector to the data and transform it in one step.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to fit and transform.
        y : array-like
            Target vector.

        Returns
        -------
        pd.DataFrame
            Transformed feature matrix containing only the selected features.
        """
        return self.fit(X, y).transform(X)