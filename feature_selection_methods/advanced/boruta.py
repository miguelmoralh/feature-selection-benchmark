import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Dict, Union, Optional
from scipy import stats
from catboost import CatBoost, Pool

class CatBoostBoruta(BaseEstimator, TransformerMixin):
    """
    Boruta Feature Selection specifically designed for CatBoost,
    handling both numerical and categorical features natively.
    """
    
    def __init__(self, 
                 model_params: dict,
                 n_iterations: int = 100,
                 alpha: float = 0.05,
                 random_state: Optional[int] = None,
                 categorical_features: Optional[List[str]] = None) -> None:
        """
        Initialize CatBoost-specific Boruta Selector.
        
        Parameters:
            model_params: Dictionary of CatBoost model parameters
            n_iterations: Number of iterations for Boruta
            alpha: Significance level for feature selection
            random_state: Random state for reproducibility
            categorical_features: List of categorical feature names
        """
        self.model_params = model_params
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.random_state = random_state
        self.categorical_features = categorical_features if categorical_features else []
        np.random.seed(random_state)
    
    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> 'CatBoostBoruta':
        """
        Fit the Boruta selector to the data.
        """
        
        # If categorical_features not provided, detect them automatically
        if not self.categorical_features:
            self.categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Store original feature names and their types
        self.feature_types_ = {
            col: 'categorical' if col in self.categorical_features else 'numerical'
            for col in X.columns
        }
        
        # Run Boruta algorithm
        feature_hits = self._run_boruta(X, y)
        
        # Select significant features
        self._select_features(feature_hits)
        
        self._print_results(X)
        return self
    
    def _run_boruta(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, int]:
        """
        Run Boruta iterations with proper handling of categorical features.
        """
        n_samples, n_features = X.shape
        feature_hits = {feature: 0 for feature in X.columns}
        
        # Get original categorical feature indices
        cat_feature_indices = [list(X.columns).index(col) for col in self.categorical_features]
        
        for iteration in range(self.n_iterations):
            # Create shadow features by permuting original features
            X_shadow = X.apply(np.random.permutation).copy()
            shadow_columns = [f"shadow_{col}" for col in X.columns]
            X_shadow.columns = shadow_columns
            
            # Combine original and shadow features
            X_combined = pd.concat([X, X_shadow], axis=1)
            
            # Update categorical features indices for combined dataset
            combined_cat_indices = (
                cat_feature_indices +  # Original categorical indices
                [i + n_features for i in cat_feature_indices]  # Shadow categorical indices
            )
            
            # Create a fresh model instance for each iteration
            model = CatBoost(self.model_params)
            
            # Create Pool object with categorical features
            train_pool = Pool(
                data=X_combined,
                label=y,
                cat_features=combined_cat_indices
            )
            
            # Fit model and get feature importances
            model.fit(train_pool)
            importances = model.get_feature_importance()
            
            # Compare original vs shadow importances
            orig_imp = importances[:n_features]
            shadow_imp = importances[n_features:]
            shadow_max = np.max(shadow_imp)
            
            # Update hits for features beating shadow
            for idx, feature in enumerate(X.columns):
                if orig_imp[idx] > shadow_max:
                    feature_hits[feature] += 1
            
            if (iteration + 1) % 10 == 0:
                print(f"Completed iteration {iteration + 1}/{self.n_iterations}")
                
        return feature_hits
    
    def _select_features(self, feature_hits: Dict[str, int]) -> None:
        """
        Select significant features using binomial test.
        """
        # Calculate hit rates
        self.hit_rates_ = {
            feature: hits / self.n_iterations 
            for feature, hits in feature_hits.items()
        }
        
        # Perform binomial test
        self.p_values_ = {
            feature: 1 - stats.binom.cdf(hits - 1, self.n_iterations, 0.5)
            for feature, hits in feature_hits.items()
        }
        
        # Select features based on significance
        self.selected_features_ = [
            feature for feature, p_value in self.p_values_.items()
            if p_value < self.alpha
        ]
        
        # Store selected feature types
        self.selected_feature_types_ = {
            feature: self.feature_types_[feature]
            for feature in self.selected_features_
        }
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input data by selecting only the chosen features.
        """
        return X[self.selected_features_].copy()
    
    def _print_results(self, X: pd.DataFrame) -> None:
        """
        Print the feature selection results with feature type information.
        """
        print(f"\nCatBoost Boruta Feature selection completed:")
        print(f"- Started with {len(X.columns)} features")
        print(f"- Removed {len(X.columns) - len(self.selected_features_)} features")
        print(f"- Retained {len(self.selected_features_)} features")