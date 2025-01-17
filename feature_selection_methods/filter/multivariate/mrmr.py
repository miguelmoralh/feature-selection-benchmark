from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd

class MRMRSelector(BaseEstimator, TransformerMixin):
    """
    A feature selector that implements the Minimum Redundancy Maximum Relevance (MRMR) algorithm.
    
    The selector follows these steps:
           
    Feature Selection Process:
       a) Calculates mutual information between each feature and target (relevance)
       b) Iteratively selects features that maximize relevance and minimize redundancy
       c) Uses the MRMR criterion: max[MI(f,y) - 1/|S| * sum(MI(f,fi))]
          where S is the set of already selected features
    
    Parameters:
    -----------
    threshold : float, default=0.01
        MRMR score threshold for feature selection.
        Score ranges from -1 (high redundancy, low relevance) to 1 (high relevance, low redundancy).
        
        The threshold of 0.01 is recommended because:
        - Scores > 0 indicate relevance exceeds redundancy
        - 0.01 provides a safety margin above pure positive correlation
        - Ensures selected features have meaningfully more relevance than redundancy
        
        This threshold value works well across diverse datasets by:
        - Removing redundant features effectively
        - Keeping features with clear unique contributions
        - Balancing feature set size and informativeness
        - Adapting well to different dataset characteristics
    task : str, default='regression'
        Type of machine learning task
    """
    
    def __init__(self, threshold=0.01):
        self.threshold = threshold
        self.selected_features_ = None
        self.relevance_scores_ = None
        
    def fit(self, X, y):
        """
        Fit the MRMR selector to the data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series or array-like
            Target variable
        
        Returns:
        --------
        self : object
            Returns the instance itself
        """
                
        # Calculate relevance scores
        self.relevance_scores_ = self._calculate_relevance(X, y)
        
        # Select features directly using MRMR
        self.selected_features_ = self._select_features_mrmr(X)
        
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
            Transformed data with only selected features
        """
        if self.selected_features_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        
        return X[self.selected_features_]
    
    def fit_transform(self, X, y):
        """
        Fit the selector and transform the data in one step.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series or array-like
            Target variable
            
        Returns:
        --------
        pd.DataFrame
            Transformed data with only selected features
        """
        return self.fit(X, y).transform(X)
    
    
    def _calculate_relevance(self, X, y):
        """
        Calculate normalized mutual information between each feature and the target.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Target variable
            
        Returns:
        --------
        dict : Feature names to normalized mutual information scores
        """
        relevance_scores = {}
        for feature in X.columns:
            # Use normalized mutual information instead of raw mutual information
            nmi = normalized_mutual_info_score(X[feature], y, average_method='arithmetic')
            relevance_scores[feature] = nmi
            
        return relevance_scores
    
    
    def _calculate_mrmr_score(self, feature, X, selected_features):
        """
        Calculate the MRMR score using normalized mutual information.
        
        Parameters:
        -----------
        feature : str
            Feature to calculate score for
        X : pd.DataFrame
            Input features
        selected_features : list
            Already selected features
            
        Returns:
        --------
        float : MRMR score using normalized MI
        """
        # Relevance term (already normalized)
        relevance = self.relevance_scores_[feature]
        
        # Redundancy term (using normalized MI)
        redundancy = 0
        if selected_features:
            for selected in selected_features:
                nmi = normalized_mutual_info_score(
                    X[feature], 
                    X[selected],
                    average_method='arithmetic'
                )
                redundancy += nmi
            redundancy /= len(selected_features)
        
        return relevance - redundancy


    def _select_features_mrmr(self, X):
        """
        Select features using the MRMR criterion with a threshold.
        Each feature is selected if its MRMR score exceeds the threshold.
        
        The selection process:
        1. First feature is the one with highest relevance
        2. For each remaining feature:
        - Calculate MRMR score (relevance - redundancy)
        - If score > threshold, select feature
        
        Parameters:
        -----------
        X : pd.DataFrame
            Input features
            
        Returns:
        --------
        list 
            Selected feature names that exceed the MRMR threshold
        """
        # Initialize lists
        selected_features = []
        remaining_features = set(X.columns)
        
        scores = list(self.relevance_scores_.values())
        
        # Select first feature (highest relevance)
        first_feature = max(self.relevance_scores_.items(), key=lambda x: x[1])[0]
        selected_features.append(first_feature)
        remaining_features.remove(first_feature)
        
        # Check remaining features directly with MRMR score
        for feature in list(remaining_features):
            mrmr_score = self._calculate_mrmr_score(feature, X, selected_features)
            
            if mrmr_score > self.threshold:
                selected_features.append(feature)
                remaining_features.remove(feature)
        
        return selected_features
            
    def get_feature_scores(self):
        """
        Get the mutual information scores for selected features.
        
        Returns:
        --------
        dict
            Dictionary mapping selected features to their relevance scores
        """
        if self.selected_features_ is None:
            raise ValueError("Transformer has not been fitted yet.")
            
        return {
            feature: self.relevance_scores_[feature] 
            for feature in self.selected_features_
        }