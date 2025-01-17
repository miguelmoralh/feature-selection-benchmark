from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
import pandas as pd
import numpy as np
from typing import List

class CatBoostFeatureImportanceSelector(BaseEstimator, TransformerMixin):
    """
    A feature selector that uses CatBoost's LossFunctionChange importance
    to select the most relevant features.
    
    Parameters:
    -----------
    model
        A CatBoost model (Classifier or Regressor).
    threshold
        Threshold to select features
    """
    def __init__(self,                
                 model: BaseEstimator, 
                 threshold: float = 0.01):
        
        self.model = model
        self.threshold = threshold
        
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the feature selector to the data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable

        Returns:
        --------
        self : object
            Fitted transformer
        """
        # Convert y to Series if it's a DataFrame
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        
        # Store original datatypes
        self.feature_dtypes_ = X.dtypes
        
        # Identify categorical columns
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create a copy of X to avoid modifying the original
        X_processed = X.copy()
        
        # Create Pool with explicit categorical features
        data_pool = Pool(
            data=X_processed,
            label=y,
            cat_features=categorical_features
        )
        
        # Train model using the Pool
        self.model.fit(data_pool)
        
        # Calculate feature importance using the same Pool
        raw_importances = self.model.get_feature_importance(
            type='LossFunctionChange',
            data=data_pool
        )
        normalized_importances = raw_importances / raw_importances.sum()
        
        # Store normalized importances
        self.feature_importances_ = pd.Series(
            normalized_importances,
            index=X.columns
        )
         
        # Select features
        self.selected_features_ = list(
            self.feature_importances_[
                self.feature_importances_ >= self.threshold
            ].index
        )
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by selecting only the chosen features.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features

        Returns:
        --------
        pd.DataFrame
            Transformed DataFrame containing only the selected features
        """
        if self.selected_features_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        
        # Create a copy to avoid modifying the original
        X_transformed = X[self.selected_features_].copy()
        
        # Restore original datatypes
        for col in X_transformed.columns:
            X_transformed[col] = X_transformed[col].astype(self.feature_dtypes_[col])
            
        return X_transformed
    
    
    def get_feature_importances(self) -> pd.Series:
        """
        Get the normalized feature importances sorted in descending order.
        
        Returns:
        --------
        pd.Series
            Normalized feature importances sorted in descending order
        """
        if self.feature_importances_ is None:
            raise ValueError("Transformer has not been fitted yet.")
            
        return self.feature_importances_.sort_values(ascending=False)