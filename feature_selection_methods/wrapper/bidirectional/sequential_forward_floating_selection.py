import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import List, Dict, Union, Optional
from sklearn.model_selection import BaseCrossValidator
from utils.utils_methods import compute_cv_score
from feature_selection_methods.wrapper.forward_selection.sequential_forward_selection import SeqForwardSelectorCV

class SeqForwardFloatingSelectorCV(BaseEstimator, TransformerMixin):
    """
    Sequential Forward Floating Selection with Cross-Validation (SFFSCV).
    
    Extends Sequential Forward Selection by adding a floating phase that evaluates 
    removing previously selected features after each addition to escape local optima.
    
    Algorithm steps:
    1. Forward phase: Add the best feature based on performance improvement
    2. Floating phase: Try removing previously selected features (except most recent)
    3. Repeat until no more features can be added or removed with sufficient improvement
    
    The floating phase is unique to SFFS and helps find better feature subsets by
    allowing backtracking when feature removal improves performance.
    
    Attributes:
        selected_features_: List[str]
            Features selected during fitting
        feature_scores_: Dict[str, float]
            Mapping of selected features to their scores when selected
        scores_history_: List[float]
            Complete history of scores at each iteration
        available_features_: List[str]
            Features that remained unselected
        n_features_removed_: int
            Count of features removed during floating phases
    """
    
    def __init__(self, 
                 model: BaseEstimator,
                 scorer: callable,
                 metric_direction: str = 'minimize',
                 cv: Optional[BaseCrossValidator] = None,
                 categorical_features: Optional[List[str]] = None,
                 min_improvement: float = 1e-4,
                 task: str = 'regression') -> None:
        """
        Initialize the Sequential Forward Floating Selector.
        
        Parameters same as SeqForwardSelectorCV with identical functionality.
        The floating behavior is handled internally without additional parameters.
        
        Parameters:
            model: Any sklearn-compatible estimator
                The machine learning model to use for feature evaluation.
                Can be classifier (binary/multiclass) or regressor.
            scorer: sklearn.metrics._scorer._PredictScorer
                Scoring function (e.g., make_scorer(roc_auc_score))
                Must be compatible with the model's predictions and task type:
                - For binary classification: can use predict_proba (e.g., ROC-AUC)
                - For multiclass: can use predict or predict_proba
                - For regression: uses predict
            metric_direction: str, default='maximize'
                Direction of the optimization ('maximize' or 'minimize')
                - 'maximize' for metrics like accuracy, ROC-AUC
                - 'minimize' for metrics like MSE, MAE
            cv: sklearn cross-validation splitter
                Cross-validation strategy:
                - StratifiedKFold recommended for classification
                - KFold recommended for regression
            categorical_features: List[str], optional
                List of categorical feature names to track during selection
            min_improvement: float, default=0.0001 (0.01%)
                Minimum improvement required to add a feature
                - For maximize: new_score - base_score must be > min_improvement
                - For minimize: base_score - new_score must be > min_improvement
                It works for all metric scales and it is reresented in percentage.
                - 1 represents 100%
                - 0.1 represents 10%  
                - 0.01 represents 1%
                - 0.001 represents 0.1%
                - 0.0001 represents 0.01%
            task : str, default='regression'
                Defines the task of the dataset we are using.
                Possible values:
                - regression
                - binary_classification
                - multiclass_classification
        """
        self.model = model
        self.scorer = scorer
        self.metric_direction = metric_direction
        self.cv = cv
        self.categorical_features = categorical_features or []
        self.supports_categorical = self._check_categorical_support()
        self.min_improvement = min_improvement
        self.task = task
        self.n_features_removed_ = 0


    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'SeqForwardFloatingSelectorCV':
        """
        Fit the SFFS algorithm to select optimal features.
        
        Combines forward selection with floating removal:
        1. Add best feature based on performance gain
        2. Try removing previously selected features (except the most recently added)
        3. Continue until no more addition or removal o improves score more than the min_improveent
        
        Parameters:
            X: Feature matrix to select from
            y: Target variable (will be encoded if categorical)
            
        Returns:
            self: Fitted selector with selected features
        """
        self.n_features_removed_ = 0  
        _, available_features = self._initialize_tracking(X, y)
        
        iteration = 0
        while available_features:
            # Get current performance baseline
            if self.selected_features_:
                if self.supports_categorical:
                    current_categorical = self._update_categorical_features(self.selected_features_)
                    self.model.set_params(cat_features=current_categorical)
                    
                current_score = compute_cv_score(
                    X[self.selected_features_], y, self.model, 
                    self.cv, self.scorer, self.task
                )
            else:
                current_score = float('-inf') if self.metric_direction == 'maximize' else float('inf')

            # Forward phase
            best_feature, best_score = self._evaluate_feature_addition(X, y, available_features, current_score)
            
            if best_feature is None:
                print("\nStopping: No features can be added that improve performance")
                break
                
            # Calculate actual performance gain
            performance_gain = (current_score - best_score if self.metric_direction == 'minimize' 
                            else best_score - current_score)
                
            # Update tracking after addition
            self.selected_features_.append(best_feature)
            available_features.remove(best_feature)
            self.feature_scores_[best_feature] = best_score
            self.scores_history_.append(best_score)
            
            print(f"\nIteration {iteration + 1}: Added '{best_feature}' "
                f"(Score: {best_score:.6f}, Performance gain: {performance_gain:.6f})")
            
            # Floating phase: remove features if beneficial
            while True:
                feature_to_remove, new_score, improvement = self._floating_phase(X, y, best_score)
                if feature_to_remove is None:
                    print("No features can be removed that improve performance. Continue with adding phase")
                    break
                    
                # Update tracking after removal
                self.selected_features_.remove(feature_to_remove)
                available_features.append(feature_to_remove)
                del self.feature_scores_[feature_to_remove]
                self.scores_history_.append(new_score)
                best_score = new_score
                self.n_features_removed_ += 1
                
                print(f"Floating phase: Removed '{feature_to_remove}' "
                      f"(New score: {new_score:.6f}, Performance gain: {improvement:.6f})")
            
            iteration += 1
        
        self.available_features_ = available_features
        self._print_results(X)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting only the chosen features.
        
        Parameters:
            X: Feature matrix to transform
                
        Returns:
            pd.DataFrame: Transformed data with only selected features
        """
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> pd.DataFrame:
        """
        Fit selector to data and transform it in one step.
        
        Parameters:
            X: Feature matrix to select from and transform
            y: Target variable for fitting
                
        Returns:
            pd.DataFrame: Transformed data with only selected features
        """
        return self.fit(X, y).transform(X)
    
    def _floating_phase(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray], 
                        current_score: float) -> tuple:
        """
        Evaluate removing previously selected features for potential improvement.
        First finds the feature whose removal gives best improvement, then checks
        if that improvement meets the threshold criteria.
        
        Parameters:
            X: Feature matrix for evaluation
            y: Target variable (encoded if necessary)
            current_score: Performance score with current feature set
            
        Returns:
            tuple: (feature_to_remove, new_score, improvement) or (None, None, None) if no improvement
        """
        if len(self.selected_features_) <= 1:
            return None, None, None
            
        best_score = float('-inf') if self.metric_direction == 'maximize' else float('inf')
        feature_to_remove = None
        best_improvement = float('-inf')
        
        # Try removing each feature except the most recent addition
        for feature in self.selected_features_[:-1]:
            temp_features = [f for f in self.selected_features_ if f != feature]
            
            # Handle categorical features if supported
            original_cat_features = None
            if self.supports_categorical:
                temp_categorical = self._update_categorical_features(temp_features)
                original_cat_features = self.model.get_params().get('cat_features', [])
                self.model.set_params(cat_features=temp_categorical)
                
            try:
                score = compute_cv_score(X[temp_features], y, self.model, self.cv, 
                                    self.scorer, self.task)
                                    
                # Calculate improvement
                if self.metric_direction == 'maximize':
                    improvement = score - current_score
                    if score > best_score:
                        best_score = score
                        feature_to_remove = feature
                        best_improvement = improvement
                else:  # minimize
                    improvement = current_score - score
                    if score < best_score:
                        best_score = score
                        feature_to_remove = feature
                        best_improvement = improvement
                        
            except Exception as e:
                print(f"Warning: Evaluation failed for feature {feature}: {str(e)}")
                continue
                
            finally:
                if self.supports_categorical and original_cat_features is not None:
                    self.model.set_params(cat_features=original_cat_features)
        
        # After finding best feature to remove, check if improvement meets threshold
        if feature_to_remove is not None:
            abs_threshold = abs(current_score * self.min_improvement)
            if best_improvement <= abs_threshold:
                return None, None, None
                    
        return feature_to_remove, best_score, best_improvement if feature_to_remove is not None else None
        
    def _print_results(self, X: pd.DataFrame) -> None:
        """
        Print final feature selection results including floating statistics.
        
        Parameters:
            X: Original feature matrix for feature counting
        """
        print(f"\nSFFS Feature selection completed:")
        print(f"- Started with {len(X.columns)} features")
        print(f"- Selected {len(self.selected_features_)} features")
        print(f"- Removed {self.n_features_removed_} features during floating phases")
        print(f"- Left {len(self.available_features_)} features unused")
        print(f"- Final score: {self.scores_history_[-1]:.4f}")
        
    # Inherit remaining utility methods
    _check_categorical_support = SeqForwardSelectorCV._check_categorical_support
    _update_categorical_features = SeqForwardSelectorCV._update_categorical_features
    _initialize_tracking = SeqForwardSelectorCV._initialize_tracking
    _evaluate_feature_addition = SeqForwardSelectorCV._evaluate_feature_addition