import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import List, Dict, Union, Optional
from sklearn.model_selection import BaseCrossValidator
from utils.utils_methods import compute_cv_score

class SeqBackSelectorCV(BaseEstimator, TransformerMixin):
    """
    Universal Sequential Backward Selection with Cross-Validation (SBSCV).
    
    This class implements a feature selection algorithm that works with any type of supervised 
    learning task (binary classification, multiclass classification, or regression). It 
    iteratively removes features based on their impact on model performance, continuing
    until no more features can be removed without significant performance degradation.
    
    The algorithm automatically detects the type of prediction needed based on the scorer
    and model capabilities, making it versatile across different types of problems.
    
    THIS ALGORITHM DO NOT APPLY FOR HIGH DIM DATA BECAUSE ITS UNFEASIBLE COMPUTATIONAL COST. 
    A LOSS THRESHOLD IS DEFINED AS STOPPING CRITERIA TO AVOID RUNNING ALL THE ALGORITHM AND REDUCE
    COMPUTATIONAL COST WHILE LOOSING A VERY SMALL RATE OF PERFORMANCE. BY THIS WAY WE MAY FIND A LOCAL
    OPTIMA INSTEAD OF THE GLOBAL BUT THE RESULTS STILL BEING IMPRESSIVE.
    
    Attributes:
        features_to_remove_: List[str]
            Features identified for removal during fitting
        feature_scores_: Dict[str, float]
            Mapping of removed features to their removal iteration scores
        scores_history_: List[float]
            Scores achieved at each iteration
        selected_features_: List[str]
            Features that remained after selection
    """
    
    def __init__(self, 
                 model: BaseEstimator,
                 scorer: callable,
                 metric_direction: str = 'maximize',
                 loss_threshold: float = 0.01,
                 min_features: int = 1,
                 cv: Optional[BaseCrossValidator] = None, 
                 categorical_features: Optional[List[str]] = None,
                 task: str = 'regression') -> None:
        """
        Initialize the Sequential Backward Selector.
        
        Parameters:
            model: Any sklearn-compatible estimator
                The machine learning model to use for feature evaluation.
                Can be classifier (binary/multiclass) or regressor.
            scorer: sklearn.metrics._scorer._PredictScorer
                Scoring function (e.g., make_scorer(roc_auc_score))
                Must be compatible with the model's predictions and task type:
                - For binary classification: can use predict_proba (ROC-AUC or average precision score)
                - For multiclass: can use predict or predict_proba (ROC-AUC ovo or average precision score)
                - For regression: uses predict (MAPE)
            metric_direction: str, default='maximize'
                Direction of the optimization ('maximize' or 'minimize')
                - 'maximize' for metrics like accuracy, ROC-AUC
                - 'minimize' for metrics like MSE, MAE
            loss_threshold: float, default=0.01 (1%)
                Maximum allowed performance drop when removing a feature
                It works for all metric scales and it is reresented in percentage.
                - 1 represents 100%
                - 0.1 represents 10%  
                - 0.01 represents 1%
                - 0.001 represents 0.1%
                - 0.0001 represents 0.01%
            min_features: int, default=1
                Minimum number of features to retain
            cv: sklearn cross-validation splitter
                Cross-validation strategy:
                - StratifiedKFold recommended for classification
                - KFold recommended for regression
            categorical_features: List[str], optional
                List of categorical feature names to track during selection
            task : str, default='regression'
                Defines the task of the dataset we are using.
                Possible values:
                - regression
                - binary_classification
                - multiclass_classification
            
        Raises:
            ValueError: If min_features is less than 1
        """
        if min_features < 1:
            raise ValueError("min_features must be at least 1")
        
        self.model = model
        self.scorer = scorer
        self.metric_direction = metric_direction
        self.loss_threshold = loss_threshold
        self.min_features = min_features
        self.cv = cv
        self.categorical_features = categorical_features or []
        self.task = task
        self.supports_categorical = self._check_categorical_support()

        
    
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'SeqBackSelectorCV':
        """
        Fit the Sequential Backward Selector to the data.
        
        This method iteratively removes features while monitoring model performance,
        continuing until no feature can be removed without significant performance drop.
        
        Parameters:
            X: Feature matrix
            y: Target variable (binary, multiclass, or continuous)
        
        Returns:
            self: The fitted selector
            
        Raises:
            ValueError: If X has fewer features than min_features
        """        
        if len(X.columns) < self.min_features:
            raise ValueError(f"Input data has {len(X.columns)} features, "
                           f"which is less than min_features={self.min_features}")
            
        # Initialize tracking variables and get initial score
        _, features = self._initialize_tracking(X, y)
        
        iteration = 0
        while features and len(features) > self.min_features:
            # Get current performance score
            base_score = compute_cv_score(X.drop(columns=self.features_to_remove_), y, self.model, self.cv, self.scorer, self.task)
            
            # Find worst performing feature
            worst_feature, best_score = self._evaluate_feature_removal(X, y, features, base_score)
            
            # Stop if no feature can be removed
            if worst_feature is None:
                print("\nStopping: No features can be removed without exceeding performance threshold")
                break
            
            # Calculate actual performance change
            performance_change = (best_score - base_score if self.metric_direction == 'minimize' 
                                else base_score - best_score)
            
            # Update tracking variables
            self.features_to_remove_.append(worst_feature)
            features.remove(worst_feature)
            self.feature_scores_[worst_feature] = best_score
            self.scores_history_.append(best_score)
            
            # Update categorical features if the removed feature was categorical
            self._update_categorical_features(worst_feature)
            
            print(f"Iteration {iteration + 1}: Removed '{worst_feature}' "
                f"(Score: {best_score:.4f}, Performance change: {performance_change:.4f})")
            
            iteration += 1
            
            # Check if we've reached minimum features
            if len(features) == self.min_features:
                print(f"\nStopping: Reached minimum number of features ({self.min_features})")
                break
        
        self.selected_features_ = features
        self._print_results(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by removing the selected features.
        
        Parameters:
            X: Feature matrix to transform
                
        Returns:
            Transformed feature matrix with selected features only
        """
        return X.drop(columns=self.features_to_remove_)
    
    def fit_transform(self, 
                     X: pd.DataFrame, 
                     y: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """
        Fit the selector to the data and transform it in one step.
        
        Parameters:
            X: Feature matrix
            y: Target variable (binary, multiclass, or continuous)
                
        Returns:
            Transformed feature matrix with selected features only
        """
        return self.fit(X, y).transform(X)
    
    def _check_categorical_support(self) -> bool:
        """
        Check if the model supports categorical features.
        
        Returns:
            bool: True if model supports categorical features, False otherwise
        """
        # First verify if the model has get_params
        if not hasattr(self.model, 'get_params'):
            return False
            
        # Then verify if cat_features is in the valid parameters 
        valid_params = self.model.get_params().keys()
        return 'cat_features' in valid_params
    
    def _update_categorical_features(self, removed_feature: str) -> None:
        """
        Update the list of categorical features when a feature is removed.
        Only updates if model supports categorical features.
        """
        if not self.supports_categorical:
            return
            
        if removed_feature in self.categorical_features:
            self.categorical_features.remove(removed_feature)
            self.model.set_params(cat_features=self.categorical_features)
    
    def _initialize_tracking(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> tuple:
        """
        Initialize tracking variables and compute initial score.
        
        Parameters:
            X: Feature matrix
            y: Target variable
        
        Returns:
            tuple: (initial_score, list of features)
        """
        self.features_to_remove_: List[str] = []
        self.feature_scores_: Dict[str, float] = {}
        self.scores_history_: List[float] = []
        
        initial_score = compute_cv_score(X, y, self.model, self.cv, self.scorer, self.task)
        self.scores_history_.append(initial_score)
        print(f"\nSBS Initial score with all features: {initial_score:.4f}")
        
        return initial_score, list(X.columns)

    def _evaluate_feature_removal(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], 
                                features: List[str], base_score: float) -> tuple:
        """
        Evaluate the impact of removing each feature and identify the feature whose removal 
        leads to the best performance, then check if it meets the threshold criteria.
        
        Parameters:
            X: Feature matrix
            y: Target variable
            features: List of current features
            base_score: Current base score
        
        Returns:
            tuple: (feature_to_remove, its_score) or (None, None) if no feature meets criteria
        """
        best_score_after_removal = float('-inf') if self.metric_direction == 'maximize' else float('inf')
        feature_to_remove = None
        smallest_performance_drop = float('inf')
        X_current = X.drop(columns=self.features_to_remove_)
        abs_threshold = abs(base_score * self.loss_threshold)
        
        for feature in features:
            if feature in self.features_to_remove_:
                continue
                
            X_temp = X_current.drop(columns=[feature])
            
            original_cat_features = None
            if self.supports_categorical:
                temp_categorical = [f for f in self.categorical_features if f != feature]
                original_cat_features = self.model.get_params().get('cat_features', [])
                self.model.set_params(cat_features=temp_categorical)
            
            try:
                score = compute_cv_score(X_temp, y, self.model, self.cv, self.scorer, self.task)
                
                if self.metric_direction == 'maximize':
                    performance_drop = base_score - score
                    if score > best_score_after_removal:
                        best_score_after_removal = score
                        feature_to_remove = feature
                        smallest_performance_drop = performance_drop
                else:  # minimize
                    performance_drop = score - base_score
                    if score < best_score_after_removal:
                        best_score_after_removal = score
                        feature_to_remove = feature
                        smallest_performance_drop = performance_drop
            
            except Exception as e:
                print(f"Error evaluating feature {feature}: {str(e)}")
                continue
            
            finally:
                if self.supports_categorical and original_cat_features is not None:
                    self.model.set_params(cat_features=original_cat_features)
        
        # After finding the best feature, check if it meets the threshold criteria
        if feature_to_remove is not None and smallest_performance_drop > abs_threshold:
            return None, None  # No feature can be removed within threshold
            
        return feature_to_remove, best_score_after_removal
    
    def _print_results(self, X: pd.DataFrame) -> None:
        """
        Print the final feature selection results.
        
        Parameters:
            X: Original feature matrix for counting features
        """
        print(f"\nSBS Feature selection completed:")
        print(f"- Started with {len(X.columns)} features")
        print(f"- Removed {len(self.features_to_remove_)} features")
        print(f"- Retained {len(self.selected_features_)} features")
        print(f"- Final score: {self.scores_history_[-1]:.4f}")
