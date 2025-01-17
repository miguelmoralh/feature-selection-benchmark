import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Union, Optional, Tuple
from sklearn.model_selection import BaseCrossValidator
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from utils.utils_methods import compute_cv_score

class HybridRFE(BaseEstimator, TransformerMixin):
    """
    Modified Hybrid Recursive Feature Elimination with optimal subset selection.
    
    Implements a feature selection algorithm that combines embedded feature importance
    metrics with cross-validated performance tracking. The algorithm runs through all
    possible feature subsets and selects the smallest subset that maintains performance
    within a threshold of the best observed score.
    
    The selection process works as follows:
    1. Start with all features and calculate initial importance scores
    2. Iteratively:
       - Remove least important feature based on current importance scores
       - Calculate cross-validated performance score for current subset
       - Track features, scores, and importance values
    3. After all iterations:
       - Identify iteration with best performance score
       - Find smallest feature subset with score within threshold of best
    
    Parameters
    ----------
    model : BaseEstimator
        Sklearn-compatible model that either:
        - Is a CatBoost model (supports get_feature_importance)
        - Has feature_importances_ attribute (like RandomForest)
    scorer : callable
        Scoring function matching prediction type:
        - Binary: make_scorer(roc_auc_score)
        - Multiclass: make_scorer(accuracy_score)
        - Regression: make_scorer(r2_score)
    metric_direction : str, default='maximize'
        Whether to maximize (accuracy, ROC-AUC) or minimize (MSE, MAE) the metric
    score_threshold : float, default=0.01 (1%)
        Maximum allowed performance drop from best score when selecting minimal subset
        It works for all metric scales and it is reresented in percentage.
        - 1 represents 100%
        - 0.1 represents 10%  
        - 0.01 represents 1%
        - 0.001 represents 0.1%
        - 0.0001 represents 0.01%
    min_features : int, default=1
        Minimum number of features to retain
    cv : BaseCrossValidator, optional
        Cross-validation splitter (StratifiedKFold for classification, KFold for regression)
    categorical_features : List[str], optional
        List of categorical feature names (only used with CatBoost models)
    task : str, default='regression'
        Defines the task of the dataset we are using.
        Possible values:
        - regression
        - binary_classification
        - multiclass_classification
    
    
    Attributes
    ----------
    iteration_history_ : List[Dict]
        Complete history of each iteration containing:
        - features: List of features in subset
        - score: Performance score achieved
        - importances: Feature importance values
        - categorical_features: Categorical features remaining
    best_iteration_ : int
        Index of iteration that achieved best performance score
    selected_features_ : List[str]
        Features in final selected subset
    supports_categorical : bool
        Whether model supports categorical features (True for CatBoost)
    """
    def __init__(
        self,
        model: BaseEstimator,
        scorer: callable,
        metric_direction: str = 'maximize',
        score_threshold: float = 0.01,
        min_features: int = 1,
        cv: Optional[BaseCrossValidator] = None,
        categorical_features: Optional[List[str]] = None,
        task: str = 'regression'
    ):
        self.model = model
        self.scorer = scorer
        self.metric_direction = metric_direction
        self.score_threshold = score_threshold
        self.min_features = min_features
        self.cv = cv
        self.categorical_features = categorical_features or []
        self.task = task
        
        # Check if model supports categorical features (CatBoost)
        self.supports_categorical = isinstance(
            self.model, (CatBoostClassifier, CatBoostRegressor))
        
        # Initialize tracking attributes
        self.iteration_history_: List[Dict] = []
        self.best_iteration_: Optional[int] = None
        self.selected_features_: List[str] = None
        self.prediction_type_: str = None

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'HybridRFE':
        """
        Fit the feature selector to data and identify optimal feature subset.
        
        Process:
        1. Calculate initial score with all features
        2. Iteratively remove least important features
        3. Select optimal subset based on threshold from best score
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with named columns
        y : Union[pd.Series, np.ndarray]
            Target variable (will be encoded if categorical)
        
        Returns
        -------
        self : HybridRFE
            Fitted selector
            
        Raises
        ------
        ValueError
            If X has fewer features than min_features
        """
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
            
        if len(X.columns) < self.min_features:
            raise ValueError(f"Input has {len(X.columns)} features, "
                           f"less than min_features={self.min_features}")
        
        remaining_features = list(X.columns)
        current_categorical = self.categorical_features.copy()
        
        # Calculate initial score with all features
        current_X = X.copy()
        initial_score = self._compute_score(current_X, y, current_categorical)
        initial_importances = self._get_feature_importances(current_X, y, current_categorical)
        
        # Track initial state
        self.iteration_history_ = [{
            'features': remaining_features.copy(),
            'score': initial_score,
            'importances': initial_importances,
            'categorical_features': current_categorical.copy()
        }]
        
        print(f"\nRFE Initial score with all features: {initial_score:.4f}")
        
        # Run feature elimination iterations
        while len(remaining_features) > self.min_features:
            # Remove least important feature from previous iteration
            least_important = self.iteration_history_[-1]['importances'].idxmin()
            current_X = current_X.drop(columns=[least_important])
            remaining_features.remove(least_important)
            
            if least_important in current_categorical:
                current_categorical.remove(least_important)
            
            # Calculate new scores and importances
            score = self._compute_score(current_X, y, current_categorical)
            importances = self._get_feature_importances(current_X, y, current_categorical)
            
            # Track iteration
            self.iteration_history_.append({
                'features': remaining_features.copy(),
                'score': score,
                'importances': importances,
                'categorical_features': current_categorical.copy()
            })
            
            print(f"Iteration {len(self.iteration_history_)-1}: "
                  f"Removed '{least_important}' (Score: {score:.4f})")
            
        # Select optimal subset of features
        self._select_optimal_subset()
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data by selecting optimal feature subset.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix to transform
        
        Returns
        -------
        pd.DataFrame
            Transformed matrix with selected features only
            
        Raises
        ------
        ValueError
            If selector has not been fitted
        """
        if self.selected_features_ is None:
            raise ValueError("Fit the selector first.")
        return X[self.selected_features_]
    
    def _compute_score(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.Series, np.ndarray],
        categorical_features: List[str]
    ) -> float:
        """
        Compute cross-validated score for current feature subset.
        
        Handles categorical features for CatBoost models by temporarily
        updating model parameters during scoring.
        
        Parameters
        ----------
        X : pd.DataFrame
            Current feature subset
        y : Union[pd.Series, np.ndarray]
            Target variable
        categorical_features : List[str]
            Current categorical features
            
        Returns
        -------
        float
            Cross-validated performance score
        """
        if self.supports_categorical:
            original_cat = self.model.get_params().get('cat_features', [])
            self.model.set_params(cat_features=categorical_features)
        
        try:
            return compute_cv_score(
                X, y, self.model, self.cv, self.scorer, self.task)
        finally:
            if self.supports_categorical:
                self.model.set_params(cat_features=original_cat)
    
    def _get_feature_importances(
        self, 
        X: pd.DataFrame, 
        y: Union[pd.Series, np.ndarray],
        categorical_features: List[str]
    ) -> pd.Series:
        """
        Calculate feature importance scores for current subset.
        
        Handles two cases:
        1. CatBoost models: Uses native get_feature_importance method
        2. Other models: Uses feature_importances_ attribute
        
        Parameters
        ----------
        X : pd.DataFrame
            Current feature subset
        y : Union[pd.Series, np.ndarray]
            Target variable
        categorical_features : List[str]
            Current categorical features
            
        Returns
        -------
        pd.Series
            Feature importance scores indexed by feature names
            
        Raises
        ------
        AttributeError
            If non-CatBoost model lacks feature_importances_ attribute
        """
        if self.supports_categorical:
            cat_indices = [i for i, col in enumerate(X.columns) 
                         if col in categorical_features]
            pool = Pool(data=X, label=y, cat_features=cat_indices)
            
            model_clone = self.model.__class__(**self.model.get_params())
            model_clone.set_params(cat_features=categorical_features)
            model_clone.fit(pool)
            
            importances = model_clone.get_feature_importance(
                type='LossFunctionChange',
                data=pool
            )
        else:
            model_clone = self.model.__class__(**self.model.get_params())
            model_clone.fit(X, y)
            
            if not hasattr(model_clone, 'feature_importances_'):
                raise AttributeError(
                    "Model must have feature_importances_ attribute or "
                    "get_feature_importance method"
                )
            importances = model_clone.feature_importances_
            
        return pd.Series(importances, index=X.columns)
    
    def _select_optimal_subset(self) -> None:
        """
        Select optimal feature subset based on best score and threshold.
        
        Algorithm:
        1. Identify iteration with best overall performance
        2. Calculate threshold score based on best score
        3. Find all iterations within threshold
        4. Select iteration with fewest features among valid ones
        
        This method sets the following attributes:
        - best_iteration_: Index of best performing iteration
        - selected_features_: Features from selected optimal subset
        """
        # Get scores array considering metric direction
        scores = np.array([iter_info['score'] for iter_info in self.iteration_history_])
        if self.metric_direction == 'minimize':
            scores = -scores
        
        # Find best score and valid iterations
        best_score = np.max(scores)
        self.best_iteration_ = np.argmax(scores)
        
        # Calculate absolute threshold based on best score
        abs_threshold = abs(best_score * self.score_threshold)
        
        threshold_score = best_score - abs_threshold
        valid_iterations = np.where(scores >= threshold_score)[0]
        
        # Among valid iterations, select one with fewest features
        n_features = [len(self.iteration_history_[i]['features']) 
                     for i in valid_iterations]
        optimal_idx = valid_iterations[np.argmin(n_features)]
        
        # Set selected features and categorical features
        optimal_info = self.iteration_history_[optimal_idx]
        self.selected_features_ = optimal_info['features']
        
        # Print results
        best_score_actual = self.iteration_history_[self.best_iteration_]['score']
        final_score = optimal_info['score']
        
        print(f"\nRFE Feature selection completed:")
        print(f"Best score {best_score_actual:.4f} at iteration {self.best_iteration_} "
              f"with {len(self.iteration_history_[self.best_iteration_]['features'])} features")
        print(f"Selected iteration {optimal_idx} with {len(self.selected_features_)} features "
              f"(Score: {final_score:.4f})")