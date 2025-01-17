import random
import numpy as np
import pandas as pd

from config import nonlinear_transforms, interaction_types
from typing import Union, List, Tuple, Dict

### Sample and tracking functions

def sample_list(base_list: list, n_samples: int) -> list:
    """
    Randomly sample a specified number of unique elements from a base list.
    
    Args:
        base_list (list): The source list to sample from
        n_samples (int): Number of elements to sample. If 0 or negative,
                        returns an empty set
    
    Returns:
        set: A set containing the randomly sampled unique elements.
              Returns empty set if n_samples <= 0
    """
    sampled_list = set(np.random.choice(
            list(base_list), n_samples, replace=False
        )) if n_samples > 0 else set()
    return sampled_list


def sample_interactive_features(features: list, informative_features: list, n_interactive: int) -> list:
    """
    Sample features for interaction effects, ensuring they are selected only from 
    non-informative features to avoid confounding with main effects.
    
    Args:
        features (list): Complete list of all available features
        informative_features (list): List of features already selected as informative
                                   (having main effects)
        n_interactive (int): Number of features to select for interactions
    
    Returns:
        set: A set of selected features for interactions. May contain fewer features
             than requested if there aren't enough non-informative features available.
    
    """
    interactive_features = set()
    if n_interactive > 0 and informative_features:
        # We can only select interactive features from non-informative ones
        non_informative = features - informative_features
        if non_informative:  # Only if we have non-informative features available
            n_interactive = min(n_interactive, len(non_informative))  # Ensure we don't request more than available
            interactive_features = set(np.random.choice(list(non_informative), n_interactive, replace=False))
    return interactive_features


def sample_non_linear_transformation_func(probability: float):
    """Sample a transformation function based on probability.
    
    Args:
        probability (float): Probability of getting a non-linear transformation
        
    Returns:
        Tuple[str, Callable]: 
            - Empty string and identity function if linear
            - Name and function if non-linear
    """
    # Decide if we'll use a non-linear transform based on probability
    if random.random() < probability:
        # Get a random non-linear transformation
        transformation_name, transformation_func = random.choice(nonlinear_transforms)
    else:
        # For linear case: empty name and identity function (linear transformation)
        transformation_name, transformation_func = "", lambda x: x
    
    return transformation_name, transformation_func

def track_non_linear_features(transform_name: str, nonlinear_features: list, feature: str) -> list:
    """
    Track nonlinear transformation if applied.
    
    Args:
        transform_name: Name of the transformation applied (empty for linear)
        nonlinear_features: List of features with nonlinear transformations
        feature: Name of the feature being transformed
    
    Returns:
        Updated list of nonlinear features
    """
    if transform_name and feature not in nonlinear_features:
        nonlinear_features.append(feature)
    return nonlinear_features

### Functions to add relations and interactions between informativ and interactive features

def add_feature_target_relation(
        X: pd.DataFrame, 
        y: np.ndarray, 
        feature: str, 
        categorical: bool, 
        non_linear_transformation_prob: float,
        formula: List[str],
        coefficients: Dict[str, float],
        nonlinear_features:List[str]
    ) -> Tuple[np.ndarray, List[str], Dict[str, float], List[str], List[str]]:
    """
    Add a relationship between a feature and the target variable, potentially applying
    nonlinear transformations and handling both categorical and numerical features.

    Args:
        X (pd.DataFrame): Input features dataframe
        y (np.ndarray): Target variable array to be modified
        feature (str): Name of the feature to add relationship for
        categorical (bool): Whether the feature is categorical
        non_linear_transformation_prob (float): Probability of applying nonlinear transformation
        formula (List[str]): List to track the mathematical formula of relationships
        coefficients (Dict[str, float]): Dictionary to store feature coefficients
        nonlinear_features (List[str]): Dictionary to track which features got nonlinear transformations

    Returns:
        Tuple containing:
            np.ndarray: Modified target variable with new relationship
            List[str]: Updated formula with new relationship
            Dict[str, float]: Updated coefficients dictionary
            List[str]: Updated nonlinear features tracking dictionary

    """
    
    # Sample a nonlinear transformation function based on given probability
    transform_name, transform_func = sample_non_linear_transformation_func(
        non_linear_transformation_prob)
    
    # Update the nonlinear_features tracking dictionary if transformation was applied
    track_non_linear_features(transform_name, nonlinear_features, feature)

    if categorical:
        # Handle categorical features by creating dummy variables
        dummies = pd.get_dummies(X[feature], prefix=feature)
        for col in dummies.columns:
            # Generate random coefficient for each category
            # Clipped to range [0.5, 4.5] and randomly assigned positive
            coefficient = np.clip(np.random.normal(loc=2.5, scale=1), 1.0, 5.0) * np.random.choice([-1, 1]) # Normal distribution centered at 2.5, standard deviation 1

            # Apply transformation and add relationship to target
            y += coefficient * transform_func(dummies[col].values)
            
            # Track the mathematical formula and coefficient
            formula.append(f"{coefficient:.2f}*{transform_name}({col})")
            coefficients[col] = abs(coefficient)
    else:
        # Handle numerical features directly
        coefficient = np.clip(np.random.normal(loc=2.5, scale=1), 1.0, 5.0) * np.random.choice([-1, 1]) # Generate random coefficient with same distribution as categorical

        
        # Apply transformation and add relationship to target
        y += coefficient * transform_func(X[feature].values)
        
        # Track the mathematical formula and coefficient
        formula.append(f"{coefficient:.2f}*{transform_name}({feature})")
        coefficients[feature] = abs(coefficient)

    return y, formula, coefficients, nonlinear_features


def add_interaction_target_relation(
        X: pd.DataFrame, 
        y: np.ndarray, 
        feature_partner: str,
        feature_interactive: str, 
        categorical_informative: bool, 
        categorical_interactive: bool,
        non_linear_transformation_prob: float,
        interaction_parts: List[str],
        nonlinear_features: List[str]
    ) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Add interaction effects between two features to the target variable, handling various 
    combinations of categorical and numerical features with optional nonlinear transformations.

    Args:
        X (pd.DataFrame): Input features dataframe
        y (np.ndarray): Target variable array to be modified
        feature_partner (str): Name of the first feature (main effect feature). Can be either an informative or other interactive feature.
        feature_interactive (str): Name of the second feature (interaction feature)
        categorical_informative (bool): Whether the first feature is categorical
        categorical_interactive (bool): Whether the second feature is categorical
        non_linear_transformation_prob (float): Probability of applying nonlinear transformation
        interaction_parts (List[str]): List to track the mathematical formulas of interactions
        nonlinear_features (Dict[str, List[str]]): Dictionary tracking features with nonlinear transformations

    Returns:
        Tuple containing:
            np.ndarray: Modified target variable with new interaction effects
            List[str]: Updated interaction formulas
            Dict[str, List[str]]: Updated nonlinear features tracking dictionary

    Notes:
        Handles three cases of interactions:
        1. Categorical-Categorical: Creates interactions between all dummy variables
           - Uses smaller coefficients (0.2-1.0)
           - Randomly transforms one of the dummies
        
        2. Categorical-Numerical: Creates interactions between each dummy and the numerical variable
           - Uses medium coefficients (0.5-2.0)
           - Randomly transforms either the dummy or numerical variable
        
        3. Numerical-Numerical: Creates direct interaction between numerical variables
           - Uses larger coefficients (0.5-4.5) from normal distribution
           - Randomly transforms one of the numerical variables
        
        Each interaction can use either multiplication (*) or division (/) as the base operation,
        combined with optional nonlinear transformations on one of the interacting features.
    """

                
    # Case 1: Both features are categorical
    if categorical_informative and categorical_interactive:
        # Get all dummy columns for both categorical variables
        dummies_informative = pd.get_dummies(X[feature_partner], prefix=feature_partner)
        dummies_interactive = pd.get_dummies(X[feature_interactive], prefix=feature_interactive)
        
        interactive_dummies = dummies_interactive.columns
        informative_dummies = dummies_informative.columns
        
        # Each dummy from one categorical variable interacts with each dummy from the other
        for i_dummy in interactive_dummies:
            for j_dummy in informative_dummies:
                # Small coefficient (1.0-2.0) for dummy-dummy interaction
                coef = np.random.uniform(1.0, 2.0) * np.random.choice([-1, 1])
                
                # Get the values for both dummies
                val1 = dummies_interactive[i_dummy].values.astype(float)
                val2 = dummies_informative[j_dummy].values.astype(float)

                # interaction operator
                base_interaction_name, base_interaction_func = random.choice(interaction_types)
                operator = '*' if base_interaction_name == 'multiply' else '/'
                
                # non-linear transformation
                transform_name, transform_func = sample_non_linear_transformation_func(
                    non_linear_transformation_prob
                )
                
                # Randomly choose which dummy to transform
                if random.random() < 0.5:
                    val1 = transform_func(val1)
                    formula = f"{coef:.2f}*({transform_name}({i_dummy}) {operator} {j_dummy})"
                    nonlinear_features = track_non_linear_features(transform_name, nonlinear_features, feature_interactive)

                else:
                    val2 = transform_func(val2)
                    formula = f"{coef:.2f}*({i_dummy} {operator} {transform_name}({j_dummy}))"
                    nonlinear_features = track_non_linear_features(transform_name, nonlinear_features, feature_partner)

                
                # Calculate interaction and add to target
                interaction_result = coef * base_interaction_func(val1, val2)
                y += coef * base_interaction_func(val1, val2)
                interaction_parts.append(formula)

    
    # Case 2: One is categorical, the other is numerical
    elif categorical_informative or categorical_interactive:

        numerical_feature = feature_interactive if categorical_informative else feature_partner
        categorical_feature = feature_partner if categorical_informative else feature_interactive
        dummies = pd.get_dummies(X[categorical_feature], prefix=categorical_feature)
        
        features_dummies = dummies.columns
        val_numerical = X[numerical_feature].values
        
        # Each dummy interacts with the numerical variable
        for dummy in features_dummies:
            coef = np.random.uniform(1.0, 2.0) * np.random.choice([-1, 1])
            val_dummy = dummies[dummy].values.astype(float)

            # interaction operator
            base_interaction_name, base_interaction_func = random.choice(interaction_types)
            operator = '*' if base_interaction_name == 'multiply' else '/'
            
            # non-linear transformation
            transform_name, transform_func = sample_non_linear_transformation_func(
                non_linear_transformation_prob
            )

            # Randomly choose which variable to transform
            if random.random() < 0.5:
                val_dummy_trans = transform_func(val_dummy)
                formula = f"{coef:.2f}*({transform_name}({dummy}) {operator} {numerical_feature})"
                interaction_result = coef * base_interaction_func(val_dummy_trans, val_numerical)
                nonlinear_features = track_non_linear_features(transform_name, nonlinear_features, categorical_feature)

            else:
                val_numerical_trans = transform_func(val_numerical.copy())  # Copy to avoid modifying original
                formula = f"{coef:.2f}*({dummy} {operator} {transform_name}({numerical_feature}))"
                interaction_result = coef * base_interaction_func(val_dummy, val_numerical_trans)
                nonlinear_features = track_non_linear_features(transform_name, nonlinear_features, numerical_feature)

            y += interaction_result
            interaction_parts.append(formula)

    # Case 3: Both features are numerical
    else:
        # Normal coefficient for numerical-numerical interaction
        coef = np.clip(np.random.normal(loc=2.5, scale=1), 1.0, 5.0) * np.random.choice([-1, 1])
        val1 = X[feature_interactive].values
        val2 = X[feature_partner].values

        # interaction operator
        base_interaction_name, base_interaction_func = random.choice(interaction_types)
        operator = '*' if base_interaction_name == 'multiply' else '/'

        # non-linear transformation
        transform_name, transform_func = sample_non_linear_transformation_func(
            non_linear_transformation_prob
        )
        
        if random.random() < 0.5:
            val1 = transform_func(val1)
            formula = f"{coef:.2f}*({transform_name}({feature_interactive}) {operator} {feature_partner})"
            nonlinear_features = track_non_linear_features(transform_name, nonlinear_features, feature_interactive)

        else:
            val2 = transform_func(val2)
            formula = f"{coef:.2f}*({feature_interactive} {operator} {transform_name}({feature_partner}))"
            nonlinear_features = track_non_linear_features(transform_name, nonlinear_features, feature_partner)

        
        interaction_result = coef * base_interaction_func(val1, val2)
        y += interaction_result
        interaction_parts.append(formula)

    
    return y, interaction_parts, nonlinear_features

### Functions to get a discrete (binary or multiclass) target from a continuous one.

def get_binary_threshold(y: np.ndarray, is_balanced: bool) -> float:
    """Calculate the threshold value for converting continuous values into binary classes.
    
    This function determines the cutoff point for binary classification based on whether
    the desired classes should be balanced or imbalanced:
    - For balanced classes: One class will have between 30-70% of samples (randomly chosen)
      and the other class will have the remaining percentage
    - For imbalanced classes: Class 1 will always be the minority class, having between 
      1-20% of samples (randomly chosen) and class 0 will have the remaining percentage
    
    Args:
        y (np.ndarray): Array of continuous values to be converted to binary
        is_balanced (bool): If True, create balanced classes; if False, create imbalanced classes
    
    Returns:
        float: The threshold value for binary classification
    """
    if is_balanced:
        # For balanced case, randomly select a split point between 30-70%
        percentile = random.uniform(30, 70)
    else:
        # For imbalanced case, ensure class 1 is minority (1-20% of samples)
        # Using a higher threshold means fewer samples will be above it (class 1)
        percentile = random.uniform(80, 99)
            
    return np.percentile(y, percentile)
        

def get_multiclass_balanced_thresholds(y: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Calculate thresholds for balanced multiclass classification where no class exceeds 60% of samples.
    Uses a random distribution approach where splits are generated ensuring no class gets more than
    60% of the data.
    
    Args:
        y (np.ndarray): Continuous target values to be discretized
        n_classes (int): Number of desired classes
        
    Returns:
        np.ndarray: Array of threshold values that will create the desired class distribution
    """
    while True:
        # Generate random splits between 0 and 100
        splits = sorted([random.uniform(0, 100) for _ in range(n_classes - 1)])
        
        # Calculate the size of each class
        class_sizes = []
        prev = 0
        for split in splits:
            class_sizes.append(split - prev)
            prev = split
        # Add the last class size
        class_sizes.append(100 - prev)
        
        # If no class exceeds 50%, we have a valid distribution
        if max(class_sizes) <= 50:
            # Convert to thresholds using percentiles
            return np.array([np.percentile(y, split) for split in splits])

def get_imbalanced_multiclass_thresholds(y: np.ndarray, n_classes: int) -> List[float]:
    """
    Calculate thresholds for imbalanced multiclass distribution with guaranteed minimum samples
    and exact number of classes.
    """
    n_samples = len(y)
    min_samples_per_class = 20
    
    # Calculate dominant class size (70-85%)
    dominant_percentage = random.uniform(70, 85)
    dominant_samples = int(dominant_percentage * n_samples / 100)
    
    # Calculate samples for other classes
    remaining_samples = n_samples - dominant_samples
    min_total_required = min_samples_per_class * (n_classes - 1)
    
    # Adjust if not enough remaining samples
    if remaining_samples < min_total_required:
        dominant_samples = n_samples - min_total_required
        remaining_samples = min_total_required
    
    # Distribute remaining samples
    non_dominant_class_sizes = np.full(n_classes - 1, min_samples_per_class)
    extra_samples = remaining_samples - min_total_required
    
    if extra_samples > 0:
        proportions = np.random.dirichlet(np.ones(n_classes-1))
        extra_per_class = np.floor(proportions * extra_samples).astype(int)
        extra_per_class[0] += extra_samples - extra_per_class.sum()
        non_dominant_class_sizes += extra_per_class
    
    # Always ensure n_classes thresholds
    dominant_class = random.randint(0, n_classes - 1)
    thresholds = []
    cumsum = 0
    
    for i in range(n_classes):
        if i == dominant_class:
            cumsum += dominant_samples
        else:
            idx = i - (1 if i > dominant_class else 0)
            cumsum += non_dominant_class_sizes[idx]
            
        if i < n_classes - 1:  # Don't add threshold for last class
            thresholds.append(np.percentile(y, (cumsum/n_samples)*100))
    
    return thresholds

def apply_thresholds(y: np.ndarray, thresholds: Union[float, List[float], np.ndarray], binary: bool = False) -> np.ndarray:
    """Apply thresholds to create discrete classes.
    
    Args:
        y (np.ndarray): Continuous values to be discretized
        thresholds (Union[float, List[float], np.ndarray]): Single threshold for binary classification
            or list/array of thresholds for multiclass
        binary (bool): If True, creates binary classes using a single threshold
    
    Returns:
        np.ndarray: Discretized array with class labels
    """
    if binary or np.isscalar(thresholds):
        # Handle binary classification case or single threshold
        threshold_value = float(thresholds)  # Convert numpy scalar to float if needed
        y_new = (y > threshold_value).astype(int)
        
        # Verify we have at least one sample of each class
        unique_classes = np.unique(y_new)
        if len(unique_classes) < 2:
            # If we somehow ended up with only one class, force create another
            if len(y_new) > 0:
                # Change a small portion of samples to the other class
                n_samples_to_change = max(1, int(len(y_new) * 0.15))  # At least 1 sample, up to 15%
                indices_to_change = np.random.choice(len(y_new), n_samples_to_change, replace=False)
                y_new[indices_to_change] = 1 - y_new[0]  # Flip to opposite class
        return y_new
    
    # Handle multiclass case
    y_new = np.zeros(len(y), dtype=int)  # Initialize as integer array
    # Ensure thresholds is a 1D array and contains no None values
    thresholds_array = np.asarray(thresholds, dtype=float).ravel()
    if len(thresholds_array) > 0:  # Only proceed if we have valid thresholds
        for i, threshold in enumerate(thresholds_array, 1):
            y_new[y >= threshold] = i
    
    return y_new

def discretize_target(
    y: np.ndarray,
    n_classes: int,
    is_balanced: bool = True,
) -> Tuple[np.ndarray, Union[float, List[float]]]:
    """
    Discretize a continuous target variable into discrete classes.
    
    Parameters
    ----------
    y : np.ndarray
        Continuous target variable to be discretized
    n_classes : int
        Number of desired classes (must be >= 2)
    is_balanced : bool, optional
        Whether to create balanced classes, by default True
    
    Returns
    -------
    Tuple[np.ndarray, Union[float, List[float]]]
        Tuple containing:
        - Discretized target variable
        - Threshold(s) used for discretization
    
    Raises
    ------
    ValueError
        If n_classes is less than 2
    """
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    
    # Binary classification case
    if n_classes == 2:
        threshold = get_binary_threshold(y, is_balanced)
        return apply_thresholds(y, threshold, binary=True), threshold
    
    # Multiclass classification case
    if is_balanced:
        thresholds = get_multiclass_balanced_thresholds(y, n_classes)
    else:
        thresholds = get_imbalanced_multiclass_thresholds(y, n_classes)
    return apply_thresholds(y, thresholds, binary=False), thresholds


### Functions to generate redundant features

def _add_noise_to_probabilities(original_probs: List[float]) -> np.ndarray:
    """
    Add random Gaussian noise to a probability distribution while maintaining valid probabilities.
    
    Args:
        original_probs (List[float]): Original probability distribution that sums to 1
        
    Returns:
        np.ndarray: Modified probability distribution with added noise, still summing to 1
        
    Notes:
        - Adds random Gaussian noise with scale randomly chosen between 20% and 80%
        - Ensures probabilities remain positive by clipping to minimum of 0.1
        - Normalizes result to maintain valid probability distribution summing to 1
        - Useful for creating more diverse and realistic probability distributions
    """
    # Random noise scale between 0.2 and 0.8
    noise_scale = random.uniform(0.2, 0.8)
    noise = np.random.normal(0, noise_scale, len(original_probs))
    new_probs = np.array(original_probs) + noise
    new_probs = np.clip(new_probs, 0.1, None)  # Ensure positive probabilities
    return new_probs / new_probs.sum()  # Normalize to sum to 1

def _generate_coefficient(min_val: float, max_val: float) -> float:
    """
    Generate a random coefficient within a specified range with random sign.
    
    Args:
        min_val (float): Minimum absolute value for the coefficient
        max_val (float): Maximum absolute value for the coefficient
        
    Returns:
        float: Random coefficient between min_val and max_val with random sign (+/-)
        
    Notes:
        - Uses uniform distribution for value generation
        - Randomly assigns positive or negative sign
        - Useful for generating balanced positive and negative relationships in data
    """
    return np.random.uniform(min_val, max_val) * np.random.choice([-1, 1])

def _add_scaled_noise(values: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Add noise to values scaled relative to their standard deviation.
    
    Args:
        values (np.ndarray): Array of values to add noise to
        n_samples (int): Number of samples in the output array
        
    Returns:
        np.ndarray: Array with added noise scaled to the data's variability
        
    Notes:
        - Noise scale is randomly chosen between 20% and 80% of the data's standard deviation
        - Uses Gaussian (normal) distribution for noise generation
    """
    # Random noise scale between 0.2 and 0.8 (20% to 80%)
    noise_percentage = random.uniform(0.2, 0.8)
    noise_scale = np.std(values) * noise_percentage
    return values + np.random.normal(0, noise_scale, n_samples)

def create_single_redundant_feature(
    X_df: pd.DataFrame,
    feature_to_replace: str,
    base_feature: str,
    is_noisy_categorical: bool,
    is_base_categorical: bool,
    n_samples: int
) -> Tuple[pd.Series, str]:
    """
    Create a single redundant feature based on a base feature.
    
    Args:
        X_df: DataFrame containing original features
        feature_to_replace: Name of the feature to be replaced with redundant feature
        base_feature: Name of the base feature used for generating redundancy
        is_noisy_categorical: Whether the redundant feature should be categorical
        is_base_categorical: Whether the base feature is categorical
        n_samples: Number of samples in the dataset
        
    Returns:
        Tuple containing:
        - Series with the new redundant feature values
        - String describing the generation formula
    """
    if is_noisy_categorical:
        if is_base_categorical:
            # Case 1: Categorical Informative - Categorical Noisy
            dummies = pd.get_dummies(X_df[base_feature], prefix=base_feature)
            
            # Get base categories by removing the prefix from dummy column names
            prefix = f"{base_feature}_"
            base_categories = [col.replace(prefix, '') for col in dummies.columns]
            
            # Calculate original probabilities and add noise
            original_probs = [dummies[col].mean() for col in dummies.columns]
            new_probs = _add_noise_to_probabilities(original_probs)
            
            # Generate new categories with modified distribution
            categories = [f'cat_{i}' for i in range(len(base_categories))]
            redundant_values = np.random.choice(categories, size=n_samples, p=new_probs)
            mapping = f"categorical_redundant_of={base_feature}"
        
        else:
            # Case 2: Numerical Informative - Categorical Noisy
            coefficient = _generate_coefficient(1.0, 2.0)
            base_values = coefficient * X_df[base_feature].values
            base_values = _add_scaled_noise(base_values, n_samples)
            
            n_categories = random.randint(3, 6)
            categories = [f'cat_{i}' for i in range(n_categories)]
            redundant_values = pd.qcut(base_values, n_categories, labels=categories)
            mapping = f"categorical_from_numerical={coefficient:.2f}*{base_feature}"
    
    else:
        if is_base_categorical:
            # Case 3: Categorical Informative - Numerical Noisy
            dummies = pd.get_dummies(X_df[base_feature], prefix=base_feature)
            redundant_values = np.zeros(n_samples)
            coefficients = []
            
            for dummy_col in dummies.columns:
                coef = _generate_coefficient(1.0, 2.0)
                redundant_values += coef * dummies[dummy_col].values
                coefficients.append(f"{coef:.2f}*{dummy_col}")
            
            redundant_values = _add_scaled_noise(redundant_values, n_samples)
            mapping = f"numerical_from_categorical={' + '.join(coefficients)}"
        
        else:
            # Case 4: Numerical Informative - Numerical Noisy
            coefficient = _generate_coefficient(0.5, 3.0)
            redundant_values = coefficient * X_df[base_feature].values
            redundant_values = _add_scaled_noise(redundant_values, n_samples)
            mapping = f"numerical_from_numerical={coefficient:.2f}*{base_feature}"
    
    return pd.Series(redundant_values, name=feature_to_replace), mapping

