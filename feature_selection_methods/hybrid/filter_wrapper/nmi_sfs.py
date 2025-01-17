from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from typing import Union, List, Optional
from sklearn.base import BaseEstimator
from sklearn.model_selection import BaseCrossValidator
from utils.utils_preprocessing import (
    encode_categorical_features, kmeans_discretize, ImputeMissing
)

from feature_selection_methods.filter.bivariate.norm_mutual_info import NormalizedMutualInfoSelector
from feature_selection_methods.wrapper.forward_selection.sequential_forward_selection import SeqForwardSelectorCV

class HybridNmiSfs(BaseEstimator, TransformerMixin):
    """
    Hybrid feature selection combining Normalized mutual information and Sequential Forward Floating Selection.
    
    Process:
    1. NMI filter phase: Removes clearly irrelevant features using normalized mutual information selector. 
    The idea is to use a less restrictive threshold than in standard NormalizedMutualInfoSelector to select more featues.
    2. SFFS wrapper phase: Fine-tunes selection using cross-validated model performance
    
    This hybrid approach balances computational efficiency with selection accuracy by:
    - Using NMI to quickly eliminate obviously irrelevant features
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
        nmi_threshold: float, default=0.01
            NMI threshold for considering a feature relevant to the target.
        num_bins: int, default=5
            Number of bins for FCBF discretization
    """
    
    def __init__(self, 
                 model: BaseEstimator,
                 scorer: callable,
                 metric_direction: str = 'minimize',
                 cv: Optional[BaseCrossValidator] = None,
                 categorical_features: Optional[List[str]] = None,
                 min_improvement: float = 1e-4,
                 task: str = 'regression',
                 threshold: float = 0.05) -> None:
        

        self.model = model
        self.scorer = scorer
        self.metric_direction = metric_direction
        self.cv = cv
        self.categorical_features = categorical_features
        self.min_improvement = min_improvement
        self.task = task
        self.threshold = threshold
        
        # Initialize individual selectors
        self.nmi = NormalizedMutualInfoSelector(threshold=self.threshold)
        self.sfs = SeqForwardSelectorCV(
            model=self.model,
            scorer=self.scorer,
            metric_direction=self.metric_direction,
            cv=self.cv,
            categorical_features=self.categorical_features,
            min_improvement=self.min_improvement,
            task=self.task
        )
        
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]) -> 'HybridNmiSfs':
        """
        Fit the hybrid selector using two-phase selection.
        
        Parameters:
            X: Feature matrix
            y: Target variable
            
        Returns:
            self: Fitted selector
        """                        
        # Phase 1: NMI filtering
        # First fit NMI to get selected features
        
        # Phase 1: NMI filtering using processed data
        X_processed, y_processed = self.process_filter(X, y)
        
        # Fit NMI on processed data
        self.nmi.fit(X_processed, y_processed)
        selected_columns = self.nmi.selected_features_
        
        if len(selected_columns) == 0:
            # If no features are selected by NMI, use all features
            selected_columns = X.columns.tolist()
        
        # Phase 2: Run SFS on original dataset with only NMI-selected columns
        # Use original (unprocessed) data for SFS
        X_filtered = X[selected_columns]  # Using original X, not X_processed
        
        # Update categorical features list for selected features only
        if self.categorical_features:
            self.sfs.categorical_features = [
                f for f in self.categorical_features 
                if f in selected_columns
            ]
        
            
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
    
    def process_filter(self, X: pd.DataFrame, y: Union[pd.Series, pd.DataFrame]):
        """
        Process data for the filter phase.
        """
        X_transformed = X.copy()
        y_transformed = y.copy()
        
        # Handle target variable - ensure it's a Series
        if isinstance(y_transformed, pd.DataFrame):
            y_transformed = y_transformed.iloc[:, 0]  # Take first column if DataFrame
        else:
            y_transformed = pd.Series(y_transformed)
            
        # Impute if there are missing values
        if X_transformed.isna().any().any():
            X_transformed = ImputeMissing().fit_transform(X_transformed)
        
        # Encode categorical variables
        X_transformed = encode_categorical_features(X_transformed)
        
        # Discretize numeric variables
        X_transformed = kmeans_discretize(X_transformed, num_bins=5)
        
        if self.task == 'regression':
            y_transformed = kmeans_discretize(y_transformed, num_bins=5)
            
        return X_transformed, y_transformed