from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import normalized_mutual_info_score
import pandas as pd
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics.cluster._supervised')

class NormalizedMutualInfoSelector(BaseEstimator, TransformerMixin):
    """
    A feature selector that uses Normalized Mutual Information (NMI) to select the most relevant
    features. The selector follows these steps:

    Feature Selection Process:
       Removes features with low information content:
          - Calculates NMI between each feature and the target
          - Removes features below the target threshold
          
    The selection process prioritizes:
    - Features strongly correlated with the target (high NMI with target)
    - Non-redundant features (low NMI between features)
    
    Threshold guidelines and recommendations:
    - < 0.01: Weak relationship
    - 0.01-0.05: Weak but potentially useful relationship
    - 0.05-0.15: Moderate relationship
    - > 0.15: Strong relationship
    
    Recommended threshold settings:
    - 0.01: Very permissive, keeps most features
    - 0.05: Recommended default for diverse datasets
    - 0.1: More selective, focuses on stronger relationships
    - 0.15: Very selective, might miss useful features
    
    For multiple diverse datasets, a threshold of 0.05 is recommended to:
    - Keep features with modest information content
    - Be suitable for both categorical and numerical features
    - Not be too aggressive in filtering
    - Allow for potential feature interactions
    
    Parameters:
    -----------

    threshold : float, default=0.05
        NMI threshold for considering a feature relevant to the target.
    average_method : str, default='arithmetic'
        Method for averaging in NMI calculation. Options: 'arithmetic', 'geometric', 'min', 'max'.
    
    Attributes:
    -----------
    selected_features_ : list
        List of selected feature names after fitting.

    """
    def __init__(self, threshold=0.01, average_method='arithmetic'):

        self.threshold = threshold
        self.average_method = average_method
        self.selected_features_ = None
            
    
    def fit(self, X, y):
        """
        Fit the feature selector to the data.
        
        The fitting process:
        - Removes features with low NMI with target

        Parameters:
        -----------
        X : pd.DataFrame
            Input features.
        y : pd.Series or pd.DataFrame
            Target variable - will be converted to 1D array if necessary.

        Returns:
        --------
        self
            Fitted transformer.
        """
        
        # Calculate mutual information with target and remove low-info features
        mutual_info_target = self.calculate_mutual_info_with_target(X, y)
        selected_features = {col for col, mi in mutual_info_target.items() 
                           if mi > self.threshold}

        # Final selected features
        self.selected_features_ = list(selected_features)
        return self
    
    def transform(self, X):
        """
        Transform the data by selecting only the chosen features.
        
        The transformation process just selects the features chosen during fitting

        Parameters:
        -----------
        X : pd.DataFrame
            Input features to transform.

        Returns:
        --------
        pd.DataFrame
            Transformed DataFrame containing only the selected features.

        Raises:
        -------
        ValueError
            If the transformer has not been fitted yet.
        """
        X_transformed = X.copy()
        if self.selected_features_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        
        return X_transformed[self.selected_features_]
    
    def fit_transform(self, X, y):
        """
        Fit the transformer to the data and return the transformed data.
        
        This is a convenience method that calls fit() and transform() in sequence.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features.
        y : pd.Series
            Target variable.

        Returns:
        --------
        pd.DataFrame
            Transformed DataFrame containing only the selected features.
        """
        return self.fit(X, y).transform(X)

    
    def calculate_mutual_info_with_target(self, X, y):
        """
        Calculate Normalized Mutual Information between each feature and the target variable.

        Parameters:
        -----------
        X : pd.DataFrame
            Input features.
        y : pd.Series
            Target variable.

        Returns:
        --------
        dict
            Dictionary mapping feature names to their NMI scores with the target.
            Format: {feature_name: nmi_score}
        """
        # Convert y to 1D array if it's a DataFrame or Series
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]
        elif isinstance(y, pd.Series):
            y = y.values
        
        mutual_info_dict = {}
        for column in X.columns:
            mi_score = normalized_mutual_info_score(X[column], y, average_method=self.average_method)
            mutual_info_dict[column] = mi_score
        return mutual_info_dict
    
