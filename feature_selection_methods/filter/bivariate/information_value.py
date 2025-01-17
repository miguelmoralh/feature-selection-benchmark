from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from itertools import combinations


class WOEInformationValueSelector(BaseEstimator, TransformerMixin):
    """
    A feature selector using Weight of Evidence (WOE) and Information Value (IV) 
    for binary classification problems. The selector follows these steps:

    Feature Selection Process:
       Removes features with low predictive power:
          - Calculates Information Value for each feature
          - Removes features below the IV threshold
           
    IV strength guidelines and recommended thresholds:
    - < 0.02: Unpredictive (remove)
    - 0.02 to 0.1: Weak predictive power
    - 0.1 to 0.3: Medium predictive power
    - 0.3 to 0.5: Strong predictive power
    - > 0.5: Suspicious (potential overfitting)
    
    Recommended threshold settings:
    - 0.02: Useless for prediction
    - 0.02 - 0.1: Weak predictor
    - 0.1 - 0.3: Medium predictor
    - 0.3 - 0.5: strong predictor
    - > 0.5: Suspicious (too good predictor)
    
    Parameters:
    -----------
    threshold_iv : float, default=0.02
        Information Value threshold for feature selection.
        Features with IV below this are considered unpredictive.
    task : str, default='regression'
        Defines the task of the dataset we are using.
        Possible values:
        - regression
        - binary_classification
        - multiclass_classification
    """
    
    def __init__(self, threshold_iv=0.02, task='binary_classification'):
        self.threshold_iv = threshold_iv
        self.task = task
        self.selected_features_ = None
        self.woe_encoders_ = {}
        self.iv_values_ = {}

    def fit(self, X, y):
        """
        Fit the feature selector to the data.
        
        The fitting process:
        1. Verifies target is binary
        2. Calculates WOE and IV for each feature
        3. Removes features with low IV
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series or pd.DataFrame
            Binary target variable (0/1)
            
        Returns:
        --------
        self
            Fitted transformer
            
        Raises:
        -------
        ValueError
            If target variable is not binary
        """
        # Ensure y is a Series
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        
        # Verify target is binary
        unique_values = y.nunique()
        if self.task != 'binary_classification':
            raise ValueError(f"Target must be binary. Found {unique_values} unique values.")
                
        # Calculate WOE and IV for each feature
        for column in X.columns:
            self._calculate_woe_iv(X[column], y, column)
                    
        # Select features based on IV threshold 
        self.selected_features_ = [col for col, iv in self.iv_values_.items() 
                                 if iv > self.threshold_iv]
        return self
    
    def transform(self, X):
        """
        Transform the data by selecting features and applying WOE transformation.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        pd.DataFrame
            WOE-transformed DataFrame containing only selected features
            
        Raises:
        -------
        ValueError
            If transformer hasn't been fitted
        """
        
        X_transformed = X.copy()
        return X_transformed[self.selected_features_]
    
    def _calculate_woe_iv(self, feature, target, feature_name):
        """
        Calculate Weight of Evidence and Information Value for a feature.
        
        Parameters:
        -----------
        feature : pd.Series
            Feature to calculate WOE and IV for
        target : pd.Series
            Binary target variable
        feature_name : str
            Name of the feature
            
        Notes:
        ------
        WOE = ln(% of positive class / % of negative class)
        IV = Σ (% of positive class - % of negative class) * WOE
        """
        # Create cross table of feature vs target
        cross_tab = pd.crosstab(feature, target, normalize='columns')
        
        # Calculate WOE and IV
        woe_dict = {}
        iv = 0
        
        for category in cross_tab.index:
            pos_rate = cross_tab.loc[category, 1]
            neg_rate = cross_tab.loc[category, 0]
            
            # Handle zero rates
            if pos_rate == 0:
                pos_rate = 0.0001
            if neg_rate == 0:
                neg_rate = 0.0001
            
            woe = np.log(pos_rate / neg_rate)
            iv += (pos_rate - neg_rate) * woe
            woe_dict[category] = woe
        
        self.woe_encoders_[feature_name] = woe_dict
        self.iv_values_[feature_name] = iv
    
    def _transform_to_woe(self, X):
        """
        Transform features to their WOE values.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to transform
            
        Returns:
        --------
        pd.DataFrame
            WOE-transformed features
        """
        X_woe = X.copy()
        
        for column in X_woe.columns:
            woe_dict = self.woe_encoders_[column]
            X_woe[column] = X_woe[column].map(woe_dict).fillna(0)
        
        return X_woe
    
    