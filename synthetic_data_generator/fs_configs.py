import random
import numpy as np
import pandas as pd

from generator import AdvancedDatasetConfig
from typing import List, Tuple, Optional, Dict, Callable
from config import dimensions, tasks, nonlinear_transforms, interaction_types

def get_informative_ratio(n_features: int) -> float:
    """Calculate informative ratio based on number of features."""
    if n_features <= 10:
        return random.uniform(0.45, 0.5)
    elif n_features > 10 and n_features <= 20:
        return random.uniform(0.35, 0.45)
    elif n_features > 100:
        return np.clip(np.random.lognormal(mean=np.log(0.25), sigma=0.1), 0.2, 0.3)
    else:
        return min(np.random.lognormal(mean=np.log(0.3), sigma=0.2), 0.45)


def create_dataset_configs(n_datasets: int = 50) -> List[AdvancedDatasetConfig]:
    """
    Generate a balanced list of dataset configurations.
    
    Creates configurations ensuring even distribution of:
    - Dimensionality (high/mid/low)
    - Task types (regression/binary/multiclass)
    - Data type, always primarily numerical (no categorical)
    - Class balance (for classification tasks)
    
    Parameters:
        n_datasets (int): Number of dataset configurations to generate
        
    Returns:
        List[AdvancedDatasetConfig]: List of configuration objects for dataset generation
    """
    configs = []
    
    # Initialize distribution counters
    dim_counts = {dim[0]: 0 for dim in dimensions}
    task_counts = {task: 0 for task in tasks}
    
    # Calculate target counts for balanced distribution
    target_per_dim = -(n_datasets // -len(dimensions))
    target_per_task = -(n_datasets // -len(tasks))
    
    while len(configs) < n_datasets:
        # Select characteristics ensuring balance
        dim_category, dim_func = random.choice(
            [dim for dim in dimensions if dim_counts[dim[0]] < target_per_dim]
        )
        task = random.choice(
            [t for t in tasks if task_counts[t] < target_per_task]
        )

        
        # Handle class balance for classification tasks
        is_balanced = None
        if task != "regression":
            classification_count = sum(1 for c in configs 
                                    if c.n_classes is not None)
            if classification_count > 0:
                balanced_ratio = sum(1 for c in configs 
                                    if c.n_classes is not None and c.is_balanced) / classification_count
                # Ensure approximately half are balanced
                is_balanced = balanced_ratio < 0.5
            else:
                is_balanced = random.choice([True, False])
        
        # Update distribution counters
        dim_counts[dim_category] += 1
        task_counts[task] += 1
        
        # Generate basic parameters
        n_features = dim_func()
        n_classes = None if task == "regression" else (
            2 if task == "binary" else random.randint(3, 7)
        )
                
        # Create configuration
        configs.append(AdvancedDatasetConfig(
            n_samples=int(np.random.randint(1000, 50000)), 
            n_features=n_features,
            n_classes=n_classes,
            is_balanced=is_balanced,
            informative_ratio = get_informative_ratio(n_features), 
            redundant_ratio = min(np.random.lognormal(mean=np.log(0.3), sigma=0.2), 0.6),
            feature_types= {'numerical': int(n_features), 'categorical': 0}, # All features are numerical
            interaction_ratio = min(np.random.lognormal(mean=np.log(0.2), sigma=0.1), 0.25),
            categorical_ratio=0.0,
            nonlinear_prob=0.25
        ))
    
    return configs