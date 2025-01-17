import random
import numpy as np
import pandas as pd

from typing import List, Tuple, Optional, Dict, Callable
from base_random_generator import RandomTabularDataGenerator
from utils import (
    sample_list, 
    sample_interactive_features, 
    add_feature_target_relation, 
    add_interaction_target_relation, 
    discretize_target,
    create_single_redundant_feature
)
from config import AdvancedDatasetConfig


class AdvancedDataGenerator:
    """
    Generator class for creating synthetic datasets with complex relationships.
    
    This class provides functionality to generate datasets with:
    - Nonlinear relationships between features and target
    - Feature interactions
    - Mixed feature types (numerical and categorical)
    - Redundant features correlated with informative features
    - Controlled class balance for classification tasks
    - Support for both regression and classification
    """
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize the generator with predefined transformations and interactions.
        
        Defines two main types of relationships:
        1. Nonlinear transformations: Applied to individual features
        2. Interaction types: Applied between pairs of features
        
        Each transformation and interaction is defined with:
        - A name for identification in metadata
        - A lambda function implementing the transformation
        
        Parameters:
            seed (int): Seed value for random number generators
        
        """
        # Set random seeds
        self.seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    def create_categorical_features(
        self, X: np.ndarray, 
        n_categorical: int
    ) -> Tuple[pd.DataFrame, List[int], Dict]:
        
        """
        Convert specified number of features to categorical.
        
        Parameters:
            X (np.ndarray): Original feature matrix
            n_categorical (int): Number of features to convert to categorical
            
        Returns:
            Tuple[pd.DataFrame, List[int]]:
                - DataFrame with mixed numerical and categorical features
                - List of indices of categorical features
        """
        X_df = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])
        features_to_convert = np.random.choice(range(X.shape[1]), size=n_categorical, replace=False)
        
        for idx in features_to_convert:
            feature_name = f'X{idx}'
            n_categories = random.randint(3, 6)
            categories = [f'cat_{i}' for i in range(n_categories)]
            
            # Generate categorical values with controlled probabilities
            probabilities = np.random.dirichlet(np.ones(n_categories))  # Random but sum to 1
            X_df[feature_name] = np.random.choice(categories, size=len(X_df), p=probabilities)
        
        return X_df, list(features_to_convert)

    def generate_target(
            self, 
            X_df: pd.DataFrame, 
            config: AdvancedDatasetConfig, 
            categorical_features: List[int], 
        ) -> Tuple[np.ndarray, Dict]:
        """
        Generate target variable with complex relationships using mixed DataFrame.
        
        This method creates a target variable by:
        1. Using informative features with linear/nonlinear relationships
        2. Creating interactions between informative and interactive features
        3. Handling both numerical and categorical features
        4. Applying random nonlinear transformations in interactions
        5. Using different coefficient scales for numerical vs categorical interactions
        
        Parameters:
            X_df (pd.DataFrame): DataFrame with both numerical and categorical features
            config (AdvancedDatasetConfig): Configuration parameters
            categorical_features (List[int]): List of indices of categorical features
                
        Returns:
            Tuple[np.ndarray, Dict]: 
                - Generated target values
                - Metadata dictionary with formulas and feature information
        """
        # Calculate n_informative ensuring at least 2 features
        n_informative = max(2, int(config.n_features * config.informative_ratio))
        
        # Calculate how many features will be used for interactions
        n_interactive = int(n_informative * config.interaction_ratio) 
        
        # Create a set of all possible feature indices
        all_features = set(range(config.n_features))
        
        # Randomly select which features will be informative (directly contribute to target)
        informative_features = sample_list(all_features, n_informative)
        
        # Randomly select features that will participate in interactions
        interactive_features = sample_interactive_features(all_features, informative_features, n_interactive)
        
        # Initialize arrays and tracking variables
        y = np.zeros(len(X_df))  # Target variable
        formula_parts = []       # Will store the linear/nonlinear terms
        interaction_parts = []   # Will store interaction terms separately
        feature_importance = {}  # Track feature coefficients
        nonlinear_features = [] # Track which features have nonlinear relationships

        # Process the informative features - these directly contribute to the target
        for i, feat_idx in enumerate(informative_features):
            feature_name = f'X{feat_idx}'
            y, formula_parts, feature_importance, nonlinear_features = add_feature_target_relation(
                X=X_df, 
                y=y,
                feature=feature_name,
                categorical=True if feat_idx in categorical_features else False,
                non_linear_transformation_prob=config.nonlinear_prob,
                formula=formula_parts,
                coefficients=feature_importance, 
                nonlinear_features=nonlinear_features
            )
        
        # Process interactions between features (interactive-informative or interactive-interactive)
        interactions = False
        if interactive_features:
            
            # Convert sets to lists once for better random selection
            informative_list = list(informative_features)
            interactive_list = list(interactive_features)
            used_interactive = set()  # Track which interactive features have been used

            for interactive_idx in interactive_list:
                # Skip if this interactive feature has already been used in an interaction
                if interactive_idx in used_interactive:
                    continue
                    
                interactive_name = f'X{interactive_idx}'

                # Get available interactive partners (excluding used ones and current)
                available_interactive = [idx for idx in interactive_list 
                                    if idx != interactive_idx 
                                    and idx not in used_interactive]
                
                # Decide whether to use interactive or informative partner
                if not available_interactive:
                    # If no interactive partners available, use random informative
                    partner_idx = np.random.choice(informative_list)  # Using np.random.choice instead of random.choice
                else:
                    # 70% chance of informative, 30% chance of interactive
                    if random.random() < 0.3:
                        partner_idx = random.choice(available_interactive)
                        # Mark both features as used
                        used_interactive.add(interactive_idx)
                        used_interactive.add(partner_idx)
                    else:
                        partner_idx = np.random.choice(informative_list)  # Using np.random.choice instead of random.choice
                        # Mark only the current feature as used
                        used_interactive.add(interactive_idx)
                            
                partner_name = f'X{partner_idx}'
                
                y, interaction_parts, nonlinear_features = add_interaction_target_relation(
                    X=X_df, 
                    y=y,
                    feature_partner=partner_name,
                    feature_interactive=interactive_name,
                    categorical_informative=True if partner_idx in categorical_features else False,
                    categorical_interactive=True if interactive_idx in categorical_features else False,
                    non_linear_transformation_prob=config.nonlinear_prob,
                    interaction_parts=interaction_parts,
                    nonlinear_features=nonlinear_features
                )
            interactions = True

        # Handle classification tasks if specified
        class_distribution = None
        if config.n_classes is not None:
            y, thresholds = discretize_target(
                y = y,
                n_classes= config.n_classes,
                is_balanced= config.is_balanced
            )

            # Calculate class distribution
            unique, counts = np.unique(y, return_counts=True)
            class_distribution = {int(cls): int(count) for cls, count in zip(unique, counts)}

        # Create ground truth features list (informative + interactive)
        ground_truth_features = list(informative_features.union(interactive_features))
        
        # Compile complete metadata
        metadata = {
            'informative_features': [f'X{i}' for i in informative_features],
            'interactive_features': [f'X{i}' for i in interactive_features],
            'ground_truth_features': [f'X{i}' for i in ground_truth_features],
            'target_formula': ' + '.join(formula_parts + interaction_parts),  # Complete formula
            'total_interaction_formula': '(' + ' , '.join(interaction_parts) + ')' if interaction_parts else '0',  # Just interactions
            'nonlinear_features': nonlinear_features,
            'class_distribution': class_distribution,
            'interactions': interactions,
        }
        
        return y, metadata
                
    def create_redundant_features(
            self, 
            X_df: pd.DataFrame, 
            informative_features: List[str],
            interactive_features: List[str], 
            categorical_features: List[str],
            n_redundant: int
        ) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Generate redundant features that are correlated with informative features while preserving data types.
        
        The function handles four cases of redundant feature generation:
        1. Categorical from categorical: Creates new categories with similar distribution + noise
        2. Categorical from numerical: Discretizes transformed numerical values into categories
        3. Numerical from categorical: Creates weighted sum of dummy variables + noise
        4. Numerical from numerical: Applies linear transformation + noise
        
        Args:
            X_df: DataFrame containing original features
            informative_features: List of features used in target generation
            interactive_features: Features involved in interactions
            categorical_features: Names of categorical features
            dummy_vars_dict: Mapping of categorical features to their dummy variables
            n_redundant: Number of redundant features to generate
            
        Returns:
            Tuple containing:
            - DataFrame with added redundant features
            - Dictionary mapping redundant features to their generation formulas
        """
        # Early exit if no informative features or redundant features needed
        if not informative_features or n_redundant <= 0:
            return X_df, {}

        X_with_redundant = X_df.copy()
        redundant_mapping = {}
        
        # Find features that can be replaced with redundant ones
        all_features = set(X_df.columns)
        noise_features = list(all_features - set(informative_features) - set(interactive_features))
        
        # Limit redundant features to available noise features
        n_redundant = min(n_redundant, len(noise_features))
        features_to_replace = random.sample(noise_features, n_redundant)
        
        for feature_to_replace in features_to_replace:
            base_feature = random.choice(informative_features)
            
            # Create redundant feature using utility function
            redundant_values, mapping = create_single_redundant_feature(
                X_df,
                feature_to_replace,
                base_feature,
                is_noisy_categorical=True if feature_to_replace in categorical_features else False,
                is_base_categorical=True if base_feature in categorical_features else False,
                n_samples = len(X_df)
            )
            
            X_with_redundant[feature_to_replace] = redundant_values
            redundant_mapping[feature_to_replace] = mapping

        return X_with_redundant, redundant_mapping
        

    
    def generate_dataset(self, config: AdvancedDatasetConfig) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Generate complete dataset with mixed numerical and categorical features, redundant features,
        target variable, and comprehensive metadata.
        
        The generation process follows these steps:
        1. Generate base numerical features
        2. Create categorical features from some of the base features
        3. Generate target variable using informative and interactive features
        4. Create redundant features based on informative features
        5. Compile complete metadata

        Parameters:
            config (AdvancedDatasetConfig): Configuration parameters including:
                - n_samples: Number of samples
                - n_features: Total number of initial features
                - informative_ratio: Proportion of features used in target generation
                - redundant_ratio: Proportion of non-informative features that will be redundant
                - feature_types: Dict specifying number of numerical/categorical features
                - other configuration parameters...
            
        Returns:
            Tuple[pd.DataFrame, pd.Series, Dict]:
                - Feature DataFrame containing numerical, categorical and redundant features
                - Target Series
                - Metadata dictionary with complete generation details
        """
        
        # Generate base numerical features
        random_generator = RandomTabularDataGenerator(random_seed=self.seed)
        X_base, dist_info = random_generator.generate_base_features(
            n_samples=config.n_samples,
            n_features=config.n_features,
            return_dist_info=True
        )
        
        # Convert features to a DataFrame and create categorical features
        X_transformed, categorical_features = self.create_categorical_features(
            X_base, config.feature_types['categorical']
        )
        
        # Generate target using transformed DataFrame and get informative features
        y, target_metadata = self.generate_target(
            X_transformed,
            config,
            categorical_features,
        )
        
        # Calculate numbers of different feature types
        n_informative = len(target_metadata['informative_features'])
        n_interactive = len(target_metadata['interactive_features'])
        n_non_informative_and_interactive = config.n_features - n_informative - n_interactive
        n_redundant = int(n_non_informative_and_interactive * config.redundant_ratio)
        n_noise = n_non_informative_and_interactive - n_redundant
        
        # Generate redundant features only if we have both informative and noise features
        if n_redundant > 0 and n_informative > 0:
            X_final, redundant_mapping = self.create_redundant_features(
                X_transformed,
                target_metadata['informative_features'],
                target_metadata['interactive_features'],
                [f'X{i}' for i in categorical_features],
                n_redundant
            )
        else:
            X_final = X_transformed.copy()
            redundant_mapping = {}
            n_redundant = 0  # Reset n_redundant if we couldn't create any
            n_noise = n_non_informative_and_interactive  # All non-informative and non-interactive features become noise
    
        
        # Compile metadata
        metadata = {
            'n_samples': config.n_samples,
            'n_features': config.n_features,  # Total remains constant
            'n_informative': n_informative,
            'n_interactive': n_interactive,
            'n_noise': n_noise,
            'n_redundant': len(redundant_mapping),  # Use actual number of redundant features created
            'n_categorical': len(categorical_features),
            'n_numerical': config.n_features - len(categorical_features),
            'n_nonlinear': len(target_metadata['nonlinear_features']),
            'categorical_features': [f'X{i}' for i in categorical_features],
            'numerical_features': [f'X{i}' for i in range(config.n_features) 
                                if i not in categorical_features],
            'informative_features': target_metadata['informative_features'],
            'interactive_features': target_metadata['interactive_features'],
            'redundant_features': list(redundant_mapping.keys()),
            'redundant_mapping': redundant_mapping,
            'task_type': 'regression' if config.n_classes is None else 
                        'binary_classification' if config.n_classes == 2 else 
                        'multiclass_classification',
            **target_metadata  # Expands all fields from target_metadata
        }
        
        return X_final, pd.Series(y, name='target'), metadata
        