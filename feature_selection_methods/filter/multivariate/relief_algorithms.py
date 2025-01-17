from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel, delayed
from tqdm import tqdm

class ReliefSelector(BaseEstimator, TransformerMixin):
    """
    A feature selector that automatically applies the appropriate Relief algorithm variant
    based on the target variable characteristics.
    
    The algorithm variant is selected as follows:
    - Binary classification: Original Relief
    - Multiclass: ReliefF
    - Regression: RReliefF
    
    The Relief family of algorithms select features based on their ability to distinguish
    between similar instances. For classification, it rewards features that differentiate
    between different classes while penalizing those that differ within the same class.
    For regression, it rewards features that have similar values when target values are
    similar.
    
    Parameters:
    -----------
    k : int, default=10
        Number of nearest neighbors
    num_instances : int, default=None
        Number of instances to use (None = all)
    threshold : float, default=0.1
        Feature weight threshold for selection.
        Weights are normalized to [-1, 1] range.
        
        The threshold of 0.1 is recommended because:
        - Negative weights indicate potentially irrelevant features
        - Weights > 0 suggest useful features
        - 0.1 provides good separation from noise
        
        This threshold value is effective across diverse datasets by:
        - Keeping features with meaningful discriminative power
        - Adapting well to different data distributions
        - Working effectively with different Relief variants
        - Balancing feature reduction and information retention
    task : str, default='regression'
        Type of machine learning task
    sigma : float, default=0.5
        Bandwidth parameter for RReliefF
    n_jobs : int, default=-1
        Number of parallel jobs
    
    Attributes
    ----------
    feature_weights_ : dict
        Dictionary containing the calculated weights for each feature
    selected_features_ : list
        List of features that exceeded the selection threshold
    
    """
    
    def __init__(self, k=10, num_instances=None, threshold=0.1, task='regression', sigma=0.5, n_jobs=-1):
        self.k = k
        self.num_instances = num_instances
        self.threshold = threshold
        self.task = task
        self.sigma = sigma
        self.n_jobs = n_jobs  
        self.feature_weights_ = None
        self.selected_features_ = None
        
    def fit(self, X, y):
        """
        Fit the Relief selector to the data by identifying target type and calculating weights.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : array-like
            Target variable (can be binary, multiclass, or continuous)
        
        Returns
        -------
        self : object
            Returns the instance itself
        
        Raises
        ------
        ValueError
            If target type is not supported (must be binary, multiclass <=10 classes, or numeric)
        """
        
        # Handle different types of y input and convert to numpy
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0].values  # Convert to numpy immediately
        else:
            y = pd.Series(y).values if not isinstance(y, np.ndarray) else y
            
        # Store column names for later reference
        self.feature_names_ = X.columns.tolist()
        
        # Convert to numpy array for faster computation
        X_values = X.values
        
        # Calculate weights based on target type
        self.feature_weights_ = self._get_weights_by_target_type(X_values, y)
        
        # Select features based on threshold
        self.selected_features_ = [
            feature for feature, weight in self.feature_weights_.items() 
            if weight > self.threshold
        ]
        
        return self
    
    def transform(self, X):
        """
        Transform the data by selecting only the features that exceeded the threshold.
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
        
        Returns
        -------
        pd.DataFrame
            Transformed data containing only the selected features
        
        Raises
        ------
        ValueError
            If the selector has not been fitted yet
        """
        if self.selected_features_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        return X[self.selected_features_]
    
    def _calculate_relief_binary(self, X, y):
        """
        Calculate feature weights for binary classification using original Relief algorithm.
        
        For each randomly selected instance, finds k nearest neighbors and updates
        feature weights based on their values. Features get:
        - Penalized when they differ for instances of the same class
        - Rewarded when they differ for instances of different class
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Binary target variable
            
        Returns
        -------
        dict
            Dictionary mapping feature names to their weights
        """
        n_samples, n_features = X.shape
        
        # Initialize nearest neighbors finder
        nn = NearestNeighbors(n_neighbors=self.k + 1)
        nn.fit(X)
        
        # Select instances
        instances_idx = self._get_instances_idx(n_samples)
        
        def process_instance(idx):
            # Get k nearest neighbors
            distances, indices = nn.kneighbors(X[idx:idx+1])
            indices = indices[0][1:]  # Remove self
            
            # Calculate all feature differences at once
            instance_diff = np.abs(X[indices] - X[idx])
            
            # Calculate hit/miss differences
            same_class = y[indices] == y[idx]
            hit_diffs = np.sum(instance_diff[same_class], axis=0)
            miss_diffs = np.sum(instance_diff[~same_class], axis=0)
            
            return (-hit_diffs + miss_diffs) / self.k
        
        # Process instances in parallel with progress bar
        weights = np.array(Parallel(n_jobs=self.n_jobs)(
            delayed(process_instance)(idx) 
            for idx in tqdm(instances_idx, desc="Computing binary Relief weights")
        ))
        
        # Average weights across instances
        final_weights = np.mean(weights, axis=0)
        return dict(zip(self.feature_names_, self._normalize_weights(final_weights)))
    
    def _calculate_relief_multiclass(self, X, y):
        """
        Calculate feature weights for multiclass classification using ReliefF algorithm.
        
        Extension of Relief for multiclass problems. Handles multiple classes by:
        - Finding nearest neighbors from each class
        - Weighting class contributions by their prior probabilities
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Multiclass target variable
            
        Returns
        -------
        dict
            Dictionary mapping feature names to their weights
        """
        n_samples, n_features = X.shape
        
        # Precompute class probabilities
        classes, class_counts = np.unique(y, return_counts=True)
        class_probs = dict(zip(classes, class_counts / n_samples))
        
        nn = NearestNeighbors(n_neighbors=self.k + 1)
        nn.fit(X)
        
        instances_idx = self._get_instances_idx(n_samples)
        
        def process_instance(idx):
            # Get k nearest neighbors
            distances, indices = nn.kneighbors(X[idx:idx+1])
            indices = indices[0][1:]
            instance_class = y[idx]
            
            # Calculate all feature differences at once
            instance_diff = np.abs(X[indices] - X[idx])
            
            # Process neighbors
            neighbor_classes = y[indices]
            same_class = neighbor_classes == instance_class
            
            # Calculate hit differences
            hit_diffs = np.sum(instance_diff[same_class], axis=0)
            
            # Calculate miss differences with probability weighting
            miss_diffs = np.zeros(n_features)
            prob_sum = sum(p for c, p in class_probs.items() if c != instance_class)
            
            if prob_sum > 0:
                for c in classes[classes != instance_class]:
                    class_mask = neighbor_classes == c
                    if np.any(class_mask):
                        miss_weight = class_probs[c] / prob_sum
                        miss_diffs += miss_weight * np.sum(instance_diff[class_mask], axis=0)
            
            return (-hit_diffs + miss_diffs) / self.k
        
        # Process instances in parallel with progress bar
        weights = np.array(Parallel(n_jobs=self.n_jobs)(
            delayed(process_instance)(idx) 
            for idx in tqdm(instances_idx, desc="Computing multiclass Relief weights")
        ))
        
        # Average weights across instances
        final_weights = np.mean(weights, axis=0)
        return dict(zip(self.feature_names_, self._normalize_weights(final_weights)))
    
    def _calculate_rrelieff(self, X, y):
        """
        Calculate feature weights for regression using RReliefF algorithm.
        
        Adapts Relief to regression by:
        - Using continuous target differences instead of class matches
        - Weighting instances based on target value similarity
        - Using a continuous similarity function (exponential kernel)
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : pd.Series
            Continuous target variable
            
        Returns
        -------
        dict
            Dictionary mapping feature names to their weights
        """
        n_samples, n_features = X.shape
        
        # Precompute normalizations
        y_range = np.ptp(y) or 1
        feature_ranges = np.ptp(X, axis=0)
        feature_ranges[feature_ranges == 0] = 1  # Avoid division by zero
        
        nn = NearestNeighbors(n_neighbors=self.k + 1)
        nn.fit(X)
        
        instances_idx = self._get_instances_idx(n_samples)
        
        def process_instance(idx):
            # Get k nearest neighbors
            distances, indices = nn.kneighbors(X[idx:idx+1])
            indices = indices[0][1:]
            
            # Calculate normalized target differences
            target_diffs = np.abs(y[indices] - y[idx]) / y_range
            instance_weights = np.exp(-target_diffs / self.sigma)
            
            # Calculate normalized feature differences for all features at once
            feature_diffs = np.abs(X[indices] - X[idx]) / feature_ranges
            
            # Weight contributions
            weighted_diffs = instance_weights[:, np.newaxis] * feature_diffs
            
            return np.sum(weighted_diffs, axis=0), np.sum(instance_weights)
        
        # Process instances in parallel with progress bar
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_instance)(idx) 
            for idx in tqdm(instances_idx, desc="Computing RReliefF weights")
        )
        
        # Aggregate results
        diff_weights = np.zeros(n_features)
        diff_counts = np.zeros(n_features)
        
        for weight, count in results:
            diff_weights += weight
            diff_counts += count
        
        # Calculate final weights
        weights = np.zeros(n_features)
        nonzero_mask = diff_counts != 0
        weights[nonzero_mask] = -(diff_weights[nonzero_mask] / diff_counts[nonzero_mask])
        
        return dict(zip(self.feature_names_, self._normalize_weights(weights)))
    
    def _normalize_weights(self, weights):
        """
        Normalize feature weights to the range [-1, 1], centered around 0.
        
        Parameters
        ----------
        weights : array-like
            Raw feature weights
                
        Returns
        -------
        array-like
            Normalized weights in [-1, 1] range, centered at 0
        """
    
        weights = np.asarray(weights)
        
        # np.ptp(weights) calculates the range of values (peak-to-peak)
        if np.ptp(weights) != 0: # Check if weights are not all identical
            
            # Center weights by subtracting mean
            # This makes the weights centered around 0
            weights_centered = weights - np.mean(weights)
            
            # Scale to [-1, 1] by dividing by maximum absolute value
            return weights_centered / np.max(np.abs(weights_centered))
        return weights

        
    def _get_weights_by_target_type(self, X, y):
        """
        Identify target type and calculate appropriate Relief weights.
        
        Parameters
        ----------
        X : pd.DataFrame
            Preprocessed input features
        y : pd.Series
            Target variable
            
        Returns
        -------
        dict
            Dictionary of feature weights calculated using the appropriate Relief variant
            
        Raises
        ------
        ValueError
            If target type is not supported
        """
        n_unique = len(np.unique(y))
        is_numeric = pd.api.types.is_numeric_dtype(y)
        
        if self.task == 'binary_classification':
            return self._calculate_relief_binary(X, y)
        elif self.task == 'multiclass_classification':
            return self._calculate_relief_multiclass(X, y)
        else:
            return self._calculate_rrelieff(X, y)
        
    def get_feature_weights(self):
        """
        Get the calculated feature weights.
        
        Returns
        -------
        dict
            Dictionary mapping feature names to their Relief weights
            
        Raises
        ------
        ValueError
            If the selector has not been fitted yet
        """
        if self.feature_weights_ is None:
            raise ValueError("Transformer has not been fitted yet.")
        return self.feature_weights_
    
    def _get_instances_idx(self, n_samples):
        """
        Get indices of instances to be used for Relief weight calculations.
        
        This method determines which instances from the dataset will be used as reference 
        points for calculating feature weights. It either returns all instances or a random
        subset based on the num_instances parameter.
        
        Parameters
        ----------
        n_samples : int
            Total number of instances in the dataset.
            
        Returns
        -------
        array-like
            If self.num_instances is None:
                Returns range(n_samples) to use all instances
            If self.num_instances is set:
                Returns random sample of indices of size self.num_instances
                
        """
        return (range(n_samples) if self.num_instances is None 
                else np.random.choice(n_samples, self.num_instances, replace=False))