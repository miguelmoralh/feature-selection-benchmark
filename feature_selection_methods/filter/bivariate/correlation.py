from sklearn.base import BaseEstimator, TransformerMixin
from itertools import combinations
import pandas as pd
import numpy as np
from scipy.stats import pointbiserialr
from scipy.stats import spearmanr
from typing import Optional, Set, List, Dict, Tuple


class CorrelationSelector(BaseEstimator, TransformerMixin):
    """
    A feature selector that uses correlation coefficients to select the most relevant
    and non-redundant features from a dataset. The selector follows these procedure:

    1. Feature Selection Process:
       Removes features with low absolute correlation with target:
          - Calculates correlation between each feature and target
          - Removes features below the absolute target threshold
          
    Correlation methods used:
    - Regression: Spearman correlation (range: [-1, 1])
    - Binary classification: Point-biserial correlation (range: [-1, 1])
        
    Parameters:
    -----------
    threshold: float, default=0.1
        Absolute correlation threshold for considering a feature relevant to the target.
        Range [0, 1] representing the minimum absolute correlation.
        
        Recommended threshold ranges based on correlation strength:
        - 0.05-0.1: Permissive selection, includes weak correlations
        - 0.1: Recommended default for diverse datasets
        - 0.2-0.3: More selective, focuses on moderate correlations
        - >0.3: Very selective, might miss important features
        
        For multiple diverse datasets, a threshold around 0.1 is recommended to:
        - Keep features with even weak correlations that might be important in combination
        - Be suitable for both large and small datasets
        - Allow for potential interaction effects
    task : str, default='regression'
        Type of task. Options: 'regression', 'binary_classification'
    remove_constant: bool, default=True
        Whether to automatically remove constant features before correlation calculation
    """
    def __init__(self,  
                 threshold: float = 0.1,
                 task: str = 'regression'):

        if not 0 <= threshold <= 1:
            raise ValueError("threshold_target must be between 0 and 1")
            
        self.threshold = threshold
        self.task = task
        self.selected_features_: Optional[List[str]] = None
        
        if task not in ['regression', 'binary_classification']:
            raise ValueError("Task must be either 'regression' or 'binary_classification'")
        
    def _is_constant(self, x: pd.Series) -> bool:
        """Check if a feature is constant (has zero variance)."""
        return x.nunique(dropna=True) <= 1
    
    def _calculate_correlation(self, x: pd.Series, y: pd.Series) -> float:
        """
        Calculate correlation based on the task type.
        Both methods return values in the range [-1, 1].
        """
        
        # First check if the feature is constant
        if self._is_constant(x):
            return 0.0
        
        try:
            if self.task == 'regression':
                corr = spearmanr(x, y)[0]
            else:  # binary classification
                corr = pointbiserialr(y, x)[0]
            return 0.0 if np.isnan(corr) else corr
        except:
            return 0.0
    
    def calculate_correlation_with_target(self, X: pd.DataFrame, 
                                        y: pd.Series) -> Dict[str, float]:
        """
        Calculate absolute correlation between each feature and the target variable.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable

        Returns:
        --------
        dict
            Dictionary mapping feature names to their absolute correlation scores with the target.
            All values are in the range [0, 1].
        """
        corr_dict = {}
        for column in X.columns:
            corr = self._calculate_correlation(X[column], y)
            corr_dict[column] = abs(corr)  # Convert to absolute value
        return corr_dict
    
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the feature selector to the data.
        
        The fitting process:
        1. Removes features with low absolute correlation with target
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable

        Returns:
        --------
        self
            Fitted transformer
        """
        # Ensure y is a Series
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        
        # Calculate correlation with target and remove low-correlation features
        correlation_target = self.calculate_correlation_with_target(X, y)
        selected_features = {col for col, corr in correlation_target.items() 
                           if corr > self.threshold}
        # Final selected features
        self.selected_features_ = list(selected_features)
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by selecting only the chosen features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features to transform

        Returns:
        --------
        pd.DataFrame
            Transformed DataFrame containing only the selected features

        Raises:
        -------
        ValueError
            If the transformer has not been fitted yet
        """
        if self.selected_features_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        
        return X[self.selected_features_]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit the transformer to the data and return the transformed data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable

        Returns:
        --------
        pd.DataFrame
            Transformed DataFrame containing only the selected features
        """
        return self.fit(X, y).transform(X)