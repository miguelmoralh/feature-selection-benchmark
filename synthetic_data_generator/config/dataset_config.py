import random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Union, Optional, Dict

# Type hints for better code documentation
DimensionTuple = Tuple[str, Callable[[], int]]
TaskType = str

dimensions: List[DimensionTuple] = [
    ("high", lambda: random.randint(100, 130)),
    ("mid", lambda: random.randint(11, 99)),
    ("low", lambda: random.randint(4, 10))
]

tasks: List[TaskType] = [
    "regression",
    "binary",
    "multiclass"
]

@dataclass
class AdvancedDatasetConfig:
    """
    Configuration class for dataset generation with advanced relationships.
    
    Attributes:
        n_samples (int): Number of samples in the dataset
        n_features (int): Total number of features
        n_classes (Optional[int]): Number of classes for classification tasks (None for regression)
        is_balanced (bool): Whether classes should be balanced (for classification)
        informative_ratio (float): Proportion of features that are informative 
        feature_types (Dict[str, int]): Dictionary specifying number of features of each type
        nonlinear_ratio (float): Proportion of informative features with nonlinear relationships
        interaction_ratio (float): Proportion of informative features involved in interactions
        categorical_ratio (float): Proportion of categorical features (0-1)
            - ≤0.2: Primarily numerical
            - ≥0.8: Primarily categorical
            - 0.2-0.8: Mixed-type
    """
    n_samples: int
    n_features: int
    n_classes: Optional[int]
    is_balanced: bool
    informative_ratio: float
    redundant_ratio: float
    feature_types: Dict[str, int]
    interaction_ratio: float
    categorical_ratio: float
    nonlinear_prob: float
