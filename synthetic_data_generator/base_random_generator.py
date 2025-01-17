import random
import numpy as np

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass

@dataclass
class DistributionParams:
    """Store parameters for different distributions"""
    name: str
    params: Dict
    generator: callable

class RandomTabularDataGenerator:
    def __init__(self, random_seed: int = None):
        """Initialize the generator with distribution definitions"""
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
            
        # Define available distributions with their parameters
        self.distributions = {
            'normal': DistributionParams(
                name='normal',
                params={'loc': 0, 'scale': 1},
                generator=np.random.normal
            ),
            'lognormal': DistributionParams(
                name='lognormal',
                params={'mean': 0, 'sigma': 0.5},
                generator=np.random.lognormal
            ),
            'exponential': DistributionParams(
                name='exponential',
                params={'scale': 1.0},
                generator=np.random.exponential
            ),
            'beta': DistributionParams(
                name='beta',
                params={'a': 2, 'b': 5},
                generator=np.random.beta
            ),
            'gamma': DistributionParams(
                name='gamma',
                params={'shape': 2, 'scale': 2},
                generator=np.random.gamma
            ),
            'weibull': DistributionParams(
                name='weibull',
                params={'a': 1.5},
                generator=np.random.weibull
            ),
            'chi_square': DistributionParams(
                name='chi_square',
                params={'df': 3},
                generator=np.random.chisquare
            )
        }
    
    def _assign_random_distributions(self, n_features: int) -> List[str]:
        """
        Randomly assign distributions to features.
        
        Returns list of distribution names for each feature.
        """
        # Generate random weights for each distribution
        weights = np.random.dirichlet(np.ones(len(self.distributions)))
        n_features_per_dist = np.random.multinomial(n_features, weights)
        
        # Create list of distribution assignments
        distribution_assignments = []
        for dist_name, n_feat in zip(self.distributions.keys(), n_features_per_dist):
            distribution_assignments.extend([dist_name] * n_feat)
            
        # Shuffle the assignments
        random.shuffle(distribution_assignments)
        return distribution_assignments
    
    def generate_base_features(
        self, 
        n_samples: int, 
        n_features: int,
        return_dist_info: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Generate base feature matrix with diverse statistical distributions.
        
        Parameters:
            n_samples (int): Number of samples to generate
            n_features (int): Number of features to generate
            return_dist_info (bool): Whether to return distribution information
            
        Returns:
            np.ndarray: Feature matrix of shape (n_samples, n_features)
            Dict: Distribution information for each feature (if return_dist_info=True)
        """
        # Initialize feature matrix
        X = np.zeros((n_samples, n_features))
        
        # Get random distribution assignments
        distributions = self._assign_random_distributions(n_features)
        
        # Generate features
        dist_info = {}
        for i, dist_name in enumerate(distributions):
            dist = self.distributions[dist_name]
            X[:, i] = dist.generator(size=n_samples, **dist.params)
            dist_info[f'feature_{i}'] = {
                'distribution': dist_name,
                'parameters': dist.params
            }
        
        if return_dist_info:
            return X, dist_info
        return X

    def get_distribution_summary(self, X: np.ndarray, dist_info: Dict) -> Dict:
        """
        Generate summary statistics for the generated features.
        
        Parameters:
            X (np.ndarray): Feature matrix
            dist_info (Dict): Distribution information
            
        Returns:
            Dict: Summary statistics for each feature
        """
        summary = {}
        for i in range(X.shape[1]):
            feature_data = X[:, i]
            summary[f'feature_{i}'] = {
                'distribution': dist_info[f'feature_{i}']['distribution'],
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'skewness': np.mean((feature_data - np.mean(feature_data))**3) / np.std(feature_data)**3,
                'min': np.min(feature_data),
                'max': np.max(feature_data)
            }
        return summary
