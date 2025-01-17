from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import Union, List, Optional
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator

from feature_selection_methods.advanced.shap import ShapFeatureImportanceSelector
from feature_selection_methods.wrapper.forward_selection.sequential_forward_selection import SeqForwardSelectorCV

class HybridShapSfs(BaseEstimator, TransformerMixin):
    """
    Hybrid feature selection combining Shap and Sequential Forward Selection.
    
    Process:
    1. Shap phase: Removes clearly irrelevant features using Shap algorithm 
    The idea is to use a less restrictive threshold than in standard Shap to select more features
    and be sure that relevant features are being selected while most irrelevant ones are being dropped. 
    2. SFFS wrapper phase: Fine-tunes selection using cross-validated model performance
    
    This hybrid approach balances computational efficiency with selection accuracy by:
    - Using Shap with less restrictive threshold to quickly eliminate obviously irrelevant features
    - Applying SFFS on the reduced feature set for precise selection
    
    Parameters:
        model: BaseEstimator
            Sklearn-compatible model for SFFS phase
        scorer: callable
            Scoring function for SFFS phase
        metric_direction: str, default='maximize'
            'maximize' for metrics like accuracy, 'minimize' for errors
        cv: BaseCrossValidator, optional
            Cross-validation strategy
        categorical_features: List[str], optional
            Categorical feature names for model
        min_improvement: float, default=1e-4
            Minimum improvement threshold for SFFS
        task: str, default='regression'
            Type of ML task: 'regression', 'binary_classification', 'multiclass_classification'
        threshold_method : str, default='adaptive'
            Method to determine feature importance threshold ('adaptive' or 'fixed')
        fixed_threshold : float, optional
            Fixed threshold value when threshold_method='fixed'
        shap_leniency: float, default=0.5
            Controls how lenient the SHAP threshold is (0-1).
            Lower values = more lenient (keeps more features)
            - 0.5 means threshold will be 50% of the normal SHAP threshold
            - 0.25 means threshold will be 25% of the normal SHAP threshold
            Higher values = more strict (keeps fewer features)
            - 0.75 means threshold will be 75% of the normal SHAP threshold
            - 1.0 means use the normal SHAP threshold
        n_jobs : int, default=1
            Number of parallel jobs for computation
        chunk_size : int, default=50
            Size of data chunks for SHAP value calculation
        max_samples : int, default=1000
            Maximum number of samples to use for SHAP calculation
        operation_timeout : int, default=300
            Maximum time in seconds for processing each fold
    """
    
    def __init__(self, 
                 model: BaseEstimator,
                 scorer: callable,
                 metric_direction: str = 'minimize',
                 cv: Optional[BaseCrossValidator] = None,
                 categorical_features: Optional[List[str]] = None,
                 min_improvement: float = 1e-4,
                 task: str = 'regression',
                 threshold: float = 0.01,
                 operation_timeout: int = 300) -> None:
        
        self.model = model
        self.scorer = scorer
        self.metric_direction = metric_direction
        self.cv = cv
        self.categorical_features = categorical_features
        self.min_improvement = min_improvement
        self.task = task
        self.operation_timeout = operation_timeout
        self.threshold = threshold
                
        # Initialize individual selectors
        self.shap =  ShapFeatureImportanceSelector(
            model=self.model,
            cv=self.cv,
            threshold=self.threshold,
            task=self.task,
            operation_timeout=self.operation_timeout
            )
        self.sfs = SeqForwardSelectorCV(
            model=self.model,
            scorer=self.scorer,
            metric_direction=self.metric_direction,
            cv=cv,
            categorical_features=self.categorical_features,
            min_improvement=self.min_improvement,
            task=self.task
        )
        
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> 'HybridShapSfs':
        """
        Fit the hybrid selector using two-phase selection.
        
        Parameters:
            X: Feature matrix
            y: Target variable
            
        Returns:
            self: Fitted selector
        """
        
        # Phase 1: SHAP filtering with lenient threshold
        print(f"\nPhase 1: SHAP selection")
        self.shap.fit(X, y)
        selected_columns = self.shap.selected_features_
        
        if len(selected_columns) == 0:
            # If no features are selected by NMI, use all features
            selected_columns = X.columns.tolist()
        
        # Phase 2: Run SFS on original dataset with only SHAP-selected columns
        print(f"\nPhase 2: SFS selection on {len(selected_columns)} SHAP-selected features")
        
        # Update categorical features list for selected features only
        if self.categorical_features:
            self.sfs.categorical_features = [
                f for f in self.categorical_features 
                if f in selected_columns
            ]
        
        X_filtered = X[selected_columns]
        self.sfs.fit(X_filtered, y)
        
        # Store the final selected features
        self.selected_features_ = self.sfs.selected_features_
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using selected features.
        
        Parameters:
            X: Feature matrix
            
        Returns:
            pd.DataFrame: Transformed data with selected features
        """
        return X[self.selected_features_]

    
    def fit_transform(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Fit and transform in one step.
        
        Parameters:
            X: Feature matrix
            y: Target variable
            
        Returns:
            pd.DataFrame: Transformed data with selected features
        """
        return self.fit(X, y).transform(X)
    
    

