import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from typing import List, Dict, Union, Optional, Tuple
from sklearn.model_selection import BaseCrossValidator
from utils.utils_methods import get_predictions, check_classification_targets
from sklearn.preprocessing import label_binarize


class PermutationImportanceSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector using Permutation Feature Importance with Cross-Validation.
    
    This class implements a feature selection algorithm that calculates feature importance 
    by measuring the decrease in model performance when features are randomly permuted.
    The importance scores are calculated during cross-validation to ensure efficiency
    and robust selection.
    
    The algorithm:
    1. For each cross-validation fold:
        a. Fit the model and get baseline score
        b. For each feature:
            - Randomly permute its values n_repeats times
            - Calculate performance drop using the fitted model
            - Store importance for this fold
    2. Average importance scores across folds
    3. Normalize final importance scores to sum to 1
    4. Select features above the threshold
    
    Attributes:
        feature_importances_: Dict[str, float]
            Mapping of features to their normalized importance scores
        selected_features_: List[str]
            Features that remained after selection
        removed_features_: List[str]
            Features that were removed due to low importance
    """
    
    def __init__(self, 
                 model: BaseEstimator,
                 scorer: callable,
                 metric_direction: str = 'maximize',
                 n_repeats: int = 5,
                 cv: Optional[BaseCrossValidator] = None,
                 task: str = 'regression',
                 threshold: float = 0.01) -> None:
        """
        Initialize the Permutation Importance Selector.
        
        Parameters:
            model: Any sklearn-compatible estimator
                The machine learning model to use for importance calculation
            scorer: sklearn.metrics._scorer._PredictScorer
                Scoring function (e.g., make_scorer(roc_auc_score))
            metric_direction: str, default='maximize'
                Direction of the optimization ('maximize' or 'minimize')
            n_repeats: int, default=5
                Number of times to repeat permutation for each feature
            cv: sklearn cross-validation splitter
                Cross-validation strategy
            task: str, default='regression'
                Type of task: 'regression', 'binary_classification', 
                or 'multiclass_classification'
            threshold
                Threshold to select features
        """
        self.model = model
        self.scorer = scorer
        self.metric_direction = metric_direction
        self.n_repeats = n_repeats
        self.cv = cv
        self.task = task
        self.threshold = threshold
        
        
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'PermutationImportanceSelector':
        """
        Fit the Permutation Importance Selector to the data.
        
        This method:
        1. Calculates feature importances using cross-validation
        2. Normalizes importance scores
        3. Selects features based on normalized importance threshold
        
        Parameters:
            X: Feature matrix
            y: Target variable
            
        Returns:
            self: The fitted selector
        """
        # Calculate importance scores with cross-validation
        feature_importances = self._calculate_cv_importances(X, y)
        
        # Normalize importance scores
        total_importance = sum(abs(score) for score in feature_importances.values())
        self.feature_importances_ = {
            feature: abs(score)/total_importance 
            for feature, score in feature_importances.items()
        }
        
        # Calculate adaptive threshold and select features
        self.selected_features_ = [
            feature for feature, importance in self.feature_importances_.items()
            if importance >= self.threshold
        ]
        
        self.removed_features_ = [
            feature for feature in X.columns 
            if feature not in self.selected_features_
        ]
        
        self._print_results(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by removing low importance features.
        
        Parameters:
            X: Feature matrix to transform
            
        Returns:
            Transformed feature matrix with selected features only
        """
        return X[self.selected_features_]
    
    def _calculate_cv_importances(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Calculate feature importances using cross-validation.
        
        For each fold:
        1. Fit model and get baseline score
        2. For each feature:
            - Permute values multiple times
            - Calculate performance drop
            - Store importance scores
        3. Average importances across folds
        
        Parameters:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Dict mapping features to their averaged importance scores
        """
        print(f"\nPermutation feature importance started!")
        
        # Initialize importance scores dictionary for each fold
        fold_importances = {feature: [] for feature in X.columns}
        
        # Convert y to numpy array for consistent indexing
        y_array = y.values.ravel() if isinstance(y, (pd.Series, pd.DataFrame)) else np.array(y).ravel()
        
        # Handle classification specific checks
        if self.task in ['binary_classification', 'multiclass_classification']:
            check_classification_targets(y_array)
            classes = np.unique(y_array)
        else:
            classes = None
        
        # Iterate through CV folds
        fold_count = 1
        for train_idx, val_idx in self.cv.split(X, y_array):
            
            # Split data
            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]
            y_train = y_array[train_idx] 
            y_val = y_array[val_idx] 
            
            # Skip fold if not all classes present in classification
            if self.task in ['binary_classification', 'multiclass_classification']:
                if len(np.unique(y_train)) < len(classes) or len(np.unique(y_val)) < len(classes):
                    continue
            
            # Fit model and get baseline score for this fold
            model = clone(self.model)
            model.fit(X_train, y_train)
            y_pred = get_predictions(model, X_val, self.task, classes)
            
            # Calculate baseline score based on task type
            if self.task == 'multiclass_classification':
                baseline_score = self._calculate_multiclass_score(y_val, y_pred, self.scorer)
            else:
                baseline_score = self.scorer._score_func(y_val, y_pred, **self.scorer._kwargs)
                            
            print(f"Processing fold {fold_count} with baseline score of {baseline_score}")
            # Calculate importance for each feature in this fold
            for feature in X.columns:
                feature_scores = []
                
                # Perform multiple permutations
                for _ in range(self.n_repeats):
                    # Create permuted validation set
                    X_val_permuted = X_val.copy()
                    X_val_permuted[feature] = np.random.permutation(X_val_permuted[feature])
                    
                    # Get predictions
                    y_pred_permuted = get_predictions(model, X_val_permuted, self.task)
                    
                    # Calculate permuted score based on task type
                    if self.task == 'multiclass_classification':
                        permuted_score = self._calculate_multiclass_score(y_val, y_pred_permuted, self.scorer)
                    else:
                        permuted_score = self.scorer._score_func(y_val, y_pred_permuted, **self.scorer._kwargs)
                    
                    # Calculate importance based on metric direction
                    if self.metric_direction == 'maximize':
                        if permuted_score > baseline_score: # In case the score is increases when a feature is permuted
                            importance = 0.0
                        else:
                            importance = baseline_score - permuted_score
                    else:
                        if baseline_score > permuted_score: # In case the score is decreases when a feature is permuted
                            importance = 0.0
                        else:
                            importance = permuted_score - baseline_score
                    
                    feature_scores.append(importance)
                
                # Average importance for this feature in this fold
                fold_importance = np.mean(feature_scores)
                fold_importances[feature].append(fold_importance)
                
            fold_count += 1
            
        if not any(fold_importances.values()):
            raise ValueError("No valid folds found for scoring.")
            
        # Average importance scores across folds
        final_importances = {
            feature: np.mean(scores) for feature, scores in fold_importances.items()
            if scores  # Only average if we have scores for this feature
        }
                
        return final_importances
    
    def _calculate_multiclass_score(self, y_true: np.ndarray, y_pred: np.ndarray, scorer: callable) -> float:
        """
        Calculate score for multiclass classification with special handling for average precision score.
        
        Parameters:
            y_true: True labels
            y_pred: Predicted probabilities
            scorer: Scoring function
            
        Returns:
            float: Calculated score
        """
        # Check if using average precision score
        if 'average_precision_score' in str(scorer._score_func):
            classes = np.unique(y_true)
            # Binarize true labels
            y_true_bin = label_binarize(y_true, classes=classes)
            # Calculate average precision score directly
            score = scorer._score_func(y_true_bin, y_pred, average='weighted')
        else:
            # For other metrics (like ROC AUC), use scorer normally
            score = scorer._score_func(y_true, y_pred, **scorer._kwargs)
        
        return score
        
    def _print_results(self, X: pd.DataFrame) -> None:
        """Print the feature selection results with normalized importance scores."""
        print(f"\nPermutation Importance Feature selection completed:")
        print(f"- Started with {len(X.columns)} features")
        print(f"- Removed {len(self.removed_features_)} features")
        print(f"- Retained {len(self.selected_features_)} features")
