import os
import json
import numpy as np
import pandas as pd

from typing import List, Tuple, Optional, Dict, Callable
from tqdm import tqdm
from generator import AdvancedDataGenerator
from fs_configs import create_dataset_configs
from feature_importances import FeatureImportanceAnalyzer
        

def generate_dataset_summary(metadata_list: List[Dict]) -> Dict:
    """
    Generate comprehensive summary statistics of all generated datasets.
    
    This function analyzes a list of dataset metadata and compiles statistics about:
    - Dataset dimensions (high/mid/low dimensional)
    - Task types distribution (regression/binary/multiclass)
    - Data types distribution (numerical/categorical/mixed)
    - Class balance in classification tasks
    - Average feature statistics across all datasets
    
    Parameters:
        metadata_list (List[Dict]): List of metadata dictionaries from each dataset
        
    Returns:
        Dict: A dictionary containing summary statistics with the following structure:
            - dimensionality: Counts of datasets by dimension category
            - task_types: Counts of datasets by task type
            - data_types: Counts of datasets by predominant data type
            - class_balance: Counts of balanced vs imbalanced classification datasets
            - feature_stats: Average statistics about different feature types
    """
    summary = {
        "dimensionality": {
            "high": sum(1 for m in metadata_list if m["n_features"] >= 100),
            "mid": sum(1 for m in metadata_list if 11 <= m["n_features"] < 100),
            "low": sum(1 for m in metadata_list if m["n_features"] < 11)
        },
        "task_types": {
            "regression": sum(1 for m in metadata_list if m["task_type"] == "regression"),
            "binary_classification": sum(1 for m in metadata_list if m["task_type"] == "binary_classification"),
            "multiclass_classification": sum(1 for m in metadata_list if m["task_type"] == "multiclass_classification")
        },
        "data_types": {
            "primarily_numerical": sum(1 for m in metadata_list if m["data_type"] == "primarily_numerical"),
            "primarily_categorical": sum(1 for m in metadata_list if m["data_type"] == "primarily_categorical"),
            "mixed": sum(1 for m in metadata_list if m["data_type"] == "mixed")
        },
        "class_balance": {
            "balanced": sum(1 for m in metadata_list if m["task_type"] != "regression" and m["is_balanced"]),
            "imbalanced": sum(1 for m in metadata_list if m["task_type"] != "regression" and not m["is_balanced"])      
        },
        "feature_stats": {
            "avg_features_per_dataset": np.mean([m["n_features"] for m in metadata_list]),
            "avg_informative_features": np.mean([m["n_informative"] for m in metadata_list]),
            "avg_interactive_features": np.mean([m["n_interactive"] for m in metadata_list]),
            "avg_redundant_features": np.mean([m["n_redundant"] for m in metadata_list]),
            "avg_noise_features": np.mean([m["n_noise"] for m in metadata_list]),
            "avg_numerical_features": np.mean([m["n_numerical"] for m in metadata_list]),
            "avg_categorical_features": np.mean([m["n_categorical"] for m in metadata_list]),
            "avg_nonlinear_features": np.mean([len(m["nonlinear_features"]) for m in metadata_list]),
        }   
    }
    
    return summary


def print_dataset_info(metadata: Dict):
    """
    Print detailed information about a single dataset in a structured and readable format.
    
    This function organizes and displays dataset information in several sections:
    1. Basic Information: Task type, number of samples, and data type
    2. Feature Distribution: Breakdown of numerical vs categorical features
    3. Feature Roles: How features contribute to the target (informative, interactive, etc.)
    4. Feature Relationships: Nonlinear transformations and interactions
    5. Classification Details: Class distribution (if applicable)
    
    Parameters:
        metadata (Dict): Dictionary containing all dataset metadata including:
            - Basic properties (dataset_id, task_type, n_samples, etc.)
            - Feature counts and distributions
            - Relationship information
            - Classification details (if applicable)
    """
    # Print header with dataset identifier
    print(f"\n{'='*50}")
    print(f"Dataset {metadata['dataset_id']} Summary:")
    print(f"{'='*50}")
    
    # Section 1: Basic Information about the dataset
    print(f"\nBasic Information:")
    print(f"  Task Type: {metadata['task_type']}")  
    print(f"  Samples: {metadata['n_samples']:,}")  
    print(f"  Data Type: {metadata['data_type']}")  
    
    # Section 2: Distribution of feature types
    # Store total for percentage calculations
    total_features = metadata['n_features']
    print("\nFeature Distribution:")
    print(f"  Total Features: {total_features}")
    # Calculate and display percentages of numerical and categorical features
    print(f"  └─ Numerical: {metadata['n_numerical']} "
          f"({metadata['n_numerical']/total_features*100:.1f}%)")
    print(f"  └─ Categorical: {metadata['n_categorical']} "
          f"({metadata['n_categorical']/total_features*100:.1f}%)")
    
    # Section 3: How features are used in the target variable
    print("\nFeature Roles:")
    print(f"  Informative: {metadata['n_informative']} "
          f"({metadata['n_informative']/total_features*100:.1f}%)")
    print(f"  Interactive: {metadata['n_interactive']} "
          f"({metadata['n_interactive']/total_features*100:.1f}%)")
    print(f"  Redundant: {metadata['n_redundant']} "
          f"({metadata['n_redundant']/total_features*100:.1f}%)")
    print(f"  Noise: {metadata['n_noise']} "
          f"({metadata['n_noise']/total_features*100:.1f}%)")
    
    # Section 4: Complex relationships between features
    print(f"\nFeature Relationships:")
    print(f"  Nonlinear Features: {len(metadata['nonlinear_features'])}")
    print(f"  Total Interaction Formula:")
    print(f"    {metadata['total_interaction_formula']}")
    
    # Section 5: Classification-specific information (if applicable)
    if metadata.get('class_distribution'):
        print(f"\nClassification Details:")
        print(f"  Balance: {'Balanced' if metadata['is_balanced'] else 'Imbalanced'}")
        print("  Class Distribution:")
        # Number and percentage of instances in each class
        for cls, count in metadata['class_distribution'].items():
            print(f"    Class {cls}: {count:,} "
                  f"({count/metadata['n_samples']*100:.1f}%)")

def generate_and_save_datasets(n_datasets: int = 50, base_dir: str = 'data/synthetic_datasets'):
    """
    Generate and save multiple datasets with detailed metadata.
    
    This function:
    1. Creates synthetic datasets with varied characteristics
    2. Saves each dataset's features and target variables
    3. Generates and saves comprehensive metadata for each dataset
    4. Creates summary statistics across all datasets
    
    The generated files include:
    - Individual dataset directories containing:
        * features.csv: Feature matrix
        * target.csv: Target variable values
        * metadata.json: Detailed dataset characteristics
    - summary_metadata.json: Complete metadata for all datasets
    - dataset_summary_stats.json: Statistical summary across datasets
    
    Parameters:
        n_datasets (int): Number of datasets to generate
        base_dir (str): Base directory for saving the datasets
    """
    os.makedirs(base_dir, exist_ok=True)
    generator = AdvancedDataGenerator(random_seed=42)
    configs = create_dataset_configs(n_datasets)
    analyzer = FeatureImportanceAnalyzer()
    metadata_list = []
    
    for i, config in tqdm(enumerate(configs), total=n_datasets, desc="Generating datasets"):
        # Generate dataset
        X, y, generation_metadata = generator.generate_dataset(config)
        
        # Calculate feature importances
        _, feature_importance = analyzer.analyze_importance(
            X=X,
            target_formula=generation_metadata['target_formula']
        )
        
        # Filter importances to include only ground truth features
        ground_truth_features = generation_metadata['ground_truth_features']
        ground_truth_importances = {
            feat: feature_importance[feat] 
            for feat in ground_truth_features 
            if feat in feature_importance
        }
        
        # Sort importances by value in descending order
        ground_truth_importances = dict(sorted(
            ground_truth_importances.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        # Create metadata
        dataset_metadata = {
            # Basic Information
            "dataset_id": i + 1,
            "task_type": generation_metadata['task_type'],
            "n_samples": config.n_samples,
            "n_features": config.n_features,
            "n_classes": config.n_classes,
            "is_balanced": config.is_balanced,
            "class_distribution": generation_metadata.get('class_distribution'),
            
            # Feature counts and types
            "n_informative": generation_metadata['n_informative'],
            "n_interactive": generation_metadata['n_interactive'],
            "n_redundant": generation_metadata['n_redundant'],
            "n_noise": generation_metadata['n_noise'],
            "n_categorical": generation_metadata['n_categorical'],
            "n_numerical": generation_metadata['n_numerical'],
            "n_nonlinear": generation_metadata['n_nonlinear'],

            
            # Feature lists and mappings
            "informative_features": generation_metadata['informative_features'],
            "interactive_features": generation_metadata['interactive_features'],
            "numerical_features": generation_metadata['numerical_features'],
            "redundant_features": generation_metadata['redundant_features'],
            "redundant_mapping": generation_metadata['redundant_mapping'],
            "categorical_features": generation_metadata['categorical_features'],
            "numerical_features": generation_metadata['numerical_features'],
            "ground_truth_features": generation_metadata['ground_truth_features'],
            "feature_importances": ground_truth_importances,
            
            # Feature relationships
            "nonlinear_features": generation_metadata['nonlinear_features'],
            "total_interaction_formula": generation_metadata['total_interaction_formula'],
            "target_formula": generation_metadata['target_formula'],
            
            # Generation parameters
            "informative_ratio": config.informative_ratio,
            "redundant_ratio": config.redundant_ratio,
            "interaction_ratio": config.interaction_ratio,
            "categorical_ratio": config.categorical_ratio,
            
            # Dataset category
            "data_type": ("primarily_numerical" if config.categorical_ratio <= 0.2 else
                         "primarily_categorical" if config.categorical_ratio >= 0.8 else
                         "mixed")
        }
        
        # Save dataset files
        dataset_dir = os.path.join(base_dir, f'dataset_{i+1}')
        os.makedirs(dataset_dir, exist_ok=True)
        X.to_csv(os.path.join(dataset_dir, 'features.csv'), index=False)
        pd.Series(y).to_csv(os.path.join(dataset_dir, 'target.csv'), index=False)
        with open(os.path.join(dataset_dir, 'metadata.json'), 'w') as f:
            json.dump(dataset_metadata, f, indent=4)
            
        metadata_list.append(dataset_metadata)
        print_dataset_info(dataset_metadata)
    
    # Generate and save summary information
    summary = generate_dataset_summary(metadata_list)
    with open(os.path.join(base_dir, 'metadata/datasets_metadata.json'), 'w') as f:
        json.dump({'n_datasets': n_datasets, 'datasets': metadata_list}, f, indent=4)
    with open(os.path.join(base_dir, 'metadata/metadata_summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    

generate_and_save_datasets(n_datasets=50)

