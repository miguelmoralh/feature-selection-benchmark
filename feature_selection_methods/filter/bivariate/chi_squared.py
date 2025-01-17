from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency
import pandas as pd

class ChiSquaredSelector(BaseEstimator, TransformerMixin):
    """
    A feature selector that uses Chi-Squared statistical test to select 
    statistically significant features.
    
    Parameters:
    -----------
    alpha : float, default=0.05
        The significance level for the chi-squared test.
        Features with p-value < alpha are considered significant and selected.
        Common values:
        - 0.05: Standard significance level
        - 0.01: More conservative
        - 0.001: Very conservative, for large datasets
    
    Attributes:
    -----------
    selected_features_ : list
        List of feature names that were selected during fitting.
    """
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.selected_features_ = None
        
    def calculate_chi2_with_target(self, X, y):
        """
        Calculate chi-squared p-values between each feature and the target.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable
            
        Returns:
        --------
        dict
            Dictionary mapping feature names to their chi-squared p-values
        """
        p_values = {}
        for column in X.columns:
            contingency_table = pd.crosstab(X[column], y)
            _, p_val, _, _ = chi2_contingency(contingency_table)
            p_values[column] = p_val
            
        return p_values

    def fit(self, X, y):
        """
        Fit the feature selector to the data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series or pd.DataFrame
            Target variable
            
        Returns:
        --------
        self
            Fitted transformer
        """
        # Ensure y is a Series
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        
        # Calculate p-values and select significant features
        p_values = self.calculate_chi2_with_target(X, y)
        selected_features = {col for col, p_val in p_values.items() 
                           if p_val < self.alpha}
            
        self.selected_features_ = list(selected_features)
        return self
    
    def transform(self, X):
        """
        Transform the data by selecting only the statistically significant features.
        
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
            If the transformer hasn't been fitted yet
        """
        if self.selected_features_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        return X[self.selected_features_]
    
    def fit_transform(self, X, y):
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
    
    def get_feature_names(self):
        """
        Get the names of the selected features.
        
        Returns:
        --------
        list
            Names of the selected features
            
        Raises:
        -------
        ValueError
            If the transformer hasn't been fitted yet
        """
        if self.selected_features_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        return self.selected_features_