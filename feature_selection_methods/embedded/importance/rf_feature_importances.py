from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class RFFearureImportanceSelector(BaseEstimator, TransformerMixin):
    """
    A feature selector that uses Random Forest's MDI (Mean Decrease in Impurity) 
    feature importance to select the most relevant features. Only works with
    Random Forest

    The selector follows these steps:
       
    Feature Selection Process:
       - Fits a Random Forest (Classifier or Regressor based on the task)
       - Calculates MDI feature importance for each feature
       - Selects features with importance above the threshold
    

    Parameters:
    -----------
    rf_model
        Random forest model. Regressor or Classifier
        depending on the dataset. 
    threshold
        Threshold to select features
    """
    def __init__(self,                
                 rf_model: BaseEstimator, 
                 threshold: float = 0.01):
        
        self.rf_model = rf_model
        self.threshold = threshold
        
    def fit(self, X, y):
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
                
        # Fit model and get feature importances
        self.rf_model.fit(X, y)
        
        # Get raw importances and normalize them to sum to 1
        raw_importances = self.rf_model.feature_importances_
        total_importance = np.sum(raw_importances)
        
        if total_importance > 0:
            normalized_importances = raw_importances / total_importance
        else:
            print("Warning: Total importance is zero. Using uniform weights.")

        self.feature_importances_ = pd.Series(
            normalized_importances,
            index=X.columns
        )
        
        # Select features based on threshold
        self.selected_features_ = list(
            self.feature_importances_[
                self.feature_importances_ >= self.threshold
            ].index
        )
        
        return self
    
    def transform(self, X):
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
            
        return X[self.selected_features_]
    
    def fit_transform(self, X, y):
        """
        Fit the transformer and return the transformed data.
        
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
        
    def get_feature_importances(self):
        """
        Get the feature importances sorted in descending order.
        
        Returns:
        --------
        pd.Series
            Feature importances sorted in descending order
        """
        if self.feature_importances_ is None:
            raise ValueError("Transformer has not been fitted yet.")
            
        return self.feature_importances_.sort_values(ascending=False)