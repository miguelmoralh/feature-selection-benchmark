import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import List, Dict, Union, Optional
from sklearn.model_selection import BaseCrossValidator
from utils.utils_methods import compute_cv_score
from feature_selection_methods.wrapper.backward_elimination.sequential_backward_selection import SeqBackSelectorCV

class SeqBackFloatingSelectorCV(BaseEstimator, TransformerMixin):
    """
    Sequential Backward Floating Selection with Cross-Validation (SBFSCV).
    
    Extends Sequential Backward Selection by adding a floating phase that evaluates 
    reintroducing previously removed features after each removal to escape local optima.
    
    Algorithm steps:
    1. Backward phase: Remove the least impactful feature based on performance degradation
    2. Floating phase: Try reintroducing previously removed features
    3. Repeat until minimum features reached or no more features can be removed within threshold
    
    The floating phase is unique to SBFS and helps find better feature subsets by
    allowing backtracking when feature reintroduction improves performance.
    
    Attributes:
        features_to_remove_: List[str]
            Features removed during fitting
        feature_scores_: Dict[str, float]
            Mapping of removed features to their removal iteration scores
        scores_history_: List[float]
            Complete history of scores at each iteration
        selected_features_: List[str]
            Features that remained after selection
        n_features_readded_: int
            Count of features reintroduced during floating phases
    """
    
    def __init__(self, 
                 model: BaseEstimator,
                 scorer: callable,
                 metric_direction: str = 'maximize',
                 loss_threshold: float = 0.001,
                 min_features: int = 1,
                 cv: Optional[BaseCrossValidator] = None,
                 categorical_features: Optional[List[str]] = None,
                 task: str = 'regression') -> None:
        """
        Initialize the Sequential Backward Floating Selector.
        
        Parameters same as SeqBackSelectorCV with identical functionality.
        The floating behavior is handled internally without additional parameters.
        
        Parameters:
            model: Any sklearn-compatible estimator
                The machine learning model to use for feature evaluation.
            scorer: sklearn.metrics._scorer._PredictScorer
                Scoring function (e.g., make_scorer(roc_auc_score))
            metric_direction: str, default='maximize'
                Direction of the optimization ('maximize' or 'minimize')
            loss_threshold: float, default=0.01 (1%)
                Maximum allowed performance drop when removing a feature
            min_features: int, default=1
                Minimum number of features to retain
            cv: sklearn cross-validation splitter
                Cross-validation strategy
            categorical_features: List[str], optional
                List of categorical feature names to track during selection
            task: str, default='regression'
                Type of task ('regression', 'binary_classification', 'multiclass_classification')
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
        self.n_features_readded_ = 0

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'SeqBackFloatingSelectorCV':
        """
        Fit the SBFS algorithm to select optimal features.
        """
        if len(X.columns) < self.min_features:
            raise ValueError(f"Input data has {len(X.columns)} features, "
                        f"which is less than min_features={self.min_features}")
            
        self.n_features_readded_ = 0
        _, features = self._initialize_tracking(X, y)
        
        iteration = 0
        while features and len(features) > self.min_features:
            # Get current performance baseline
            base_score = compute_cv_score(
                X.drop(columns=self.features_to_remove_), y,
                self.model, self.cv, self.scorer, self.task
            )
            
            # Backward phase - find worst feature
            worst_feature, removal_score = self._evaluate_feature_removal(
                X, y, features, base_score
            )
            
            if worst_feature is None:
                print("\nStopping: No features can be removed within threshold")
                break
                
            # Calculate performance change
            performance_change = (removal_score - base_score if self.metric_direction == 'minimize'
                                else base_score - removal_score)
            
            # Update tracking after removal
            if worst_feature in features:
                features.remove(worst_feature)
            if worst_feature not in self.features_to_remove_:
                self.features_to_remove_.append(worst_feature)
                
            self.feature_scores_[worst_feature] = removal_score
            self.scores_history_.append(removal_score)
            
            if self.supports_categorical:
                self._update_categorical_features(worst_feature)
            
            print(f"\nIteration {iteration + 1}: Removed '{worst_feature}' "
                f"(Score: {removal_score:.6f}, Performance change: {performance_change:.6f})")
            
            # Floating phase: reintroduce features if beneficial
            current_score = removal_score
            while True:
                feature_to_readd, new_score, improvement = self._floating_phase(
                    X, y, current_score, worst_feature  # Pass the currently removed feature
                )
                
                if feature_to_readd is None:
                    print("No features can be reintroduced that improve performance")
                    break
                    
                # Update tracking after readdition
                if feature_to_readd in self.features_to_remove_:
                    self.features_to_remove_.remove(feature_to_readd)
                if feature_to_readd not in features:
                    features.append(feature_to_readd)
                    
                if feature_to_readd in self.feature_scores_:
                    del self.feature_scores_[feature_to_readd]
                    
                self.scores_history_.append(new_score)
                current_score = new_score
                self.n_features_readded_ += 1
                
                if self.supports_categorical and feature_to_readd in self.categorical_features:
                    self.model.set_params(cat_features=self.categorical_features)
                
                print(f"Floating phase: Reintroduced '{feature_to_readd}' "
                    f"(New score: {new_score:.6f}, Performance gain: {improvement:.6f})")
                
            iteration += 1
            
            if len(features) == self.min_features:
                print(f"\nStopping: Reached minimum number of features ({self.min_features})")
                break
                
        self.selected_features_ = features
        self._print_results(X)
        return self
    
    def _update_categorical_features(self, removed_feature: str) -> None:
        """
        Update the list of categorical features when a feature is removed.
        Save original categorical status for potential reintroduction.
        """
        if not self.supports_categorical:
            return
                
        # Store original categorical features if not already stored
        if not hasattr(self, 'original_categorical_features_'):
            self.original_categorical_features_ = self.model.get_params().get('cat_features', []).copy()
                
        if removed_feature in self.categorical_features:
            self.categorical_features.remove(removed_feature)
            self.model.set_params(cat_features=self.categorical_features)

    def _floating_phase(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], 
                    current_score: float, current_removed_feature: str) -> tuple:
        """
        Evaluate reintroducing previously removed features for potential improvement.
        
        Parameters:
            X: Feature matrix for evaluation
            y: Target variable
            current_score: Performance score with current feature set
            current_removed_feature: Feature that was just removed in current iteration
            
        Returns:
            tuple: (feature_to_readd, new_score, improvement) or (None, None, None)
        """
        if not self.features_to_remove_:
            return None, None, None
                
        best_score = float('-inf') if self.metric_direction == 'maximize' else float('inf')
        feature_to_readd = None
        best_improvement = float('-inf')
        
        features_to_try = [f for f in self.features_to_remove_ if f != current_removed_feature]
        
        if not features_to_try:
            return None, None, None
        
        # Get original categorical features list
        original_cats = getattr(self, 'original_categorical_features_', [])
        
        for feature in features_to_try:
            temp_removed = [f for f in self.features_to_remove_ if f != feature]
            
            # Prepare categorical features for evaluation
            if self.supports_categorical:
                current_cats = [f for f in self.categorical_features]
                if feature in original_cats:
                    current_cats.append(feature)
                self.model.set_params(cat_features=current_cats)
            
            try:
                score = compute_cv_score(
                    X.drop(columns=temp_removed), y,
                    self.model, self.cv, self.scorer, self.task
                )
                
                if self.metric_direction == 'maximize':
                    improvement = score - current_score
                    if score > best_score:
                        best_score = score
                        feature_to_readd = feature
                        best_improvement = improvement
                else:
                    improvement = current_score - score
                    if score < best_score:
                        best_score = score
                        feature_to_readd = feature
                        best_improvement = improvement
                        
            except Exception as e:
                print(f"Warning: Evaluation failed for feature {feature}: {str(e)}")
                continue
            finally:
                # Restore current categorical features after evaluation
                if self.supports_categorical:
                    self.model.set_params(cat_features=self.categorical_features)
        
        if feature_to_readd is not None:
            abs_threshold = abs(current_score * self.loss_threshold)
            if best_improvement <= abs_threshold:
                return None, None, None
                
            # Update categorical features if readding a categorical feature
            if self.supports_categorical and feature_to_readd in original_cats:
                self.categorical_features.append(feature_to_readd)
                self.model.set_params(cat_features=self.categorical_features)
                    
        return feature_to_readd, best_score, best_improvement if feature_to_readd is not None else None

    def _print_results(self, X: pd.DataFrame) -> None:
        """
        Print final feature selection results including floating statistics.
        
        Parameters:
            X: Original feature matrix for feature counting
        """
        print(f"\nSBFS Feature selection completed:")
        print(f"- Started with {len(X.columns)} features")
        print(f"- Removed {len(self.features_to_remove_)} features")
        print(f"- Reintroduced {self.n_features_readded_} features during floating phases")
        print(f"- Retained {len(self.selected_features_)} features")
        print(f"- Final score: {self.scores_history_[-1]:.4f}")

    # Inherit remaining utility methods
    _check_categorical_support = SeqBackSelectorCV._check_categorical_support
    _initialize_tracking = SeqBackSelectorCV._initialize_tracking
    _evaluate_feature_removal = SeqBackSelectorCV._evaluate_feature_removal
    
    # Inherit transform methods
    transform = SeqBackSelectorCV.transform
    fit_transform = SeqBackSelectorCV.fit_transform