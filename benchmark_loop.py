import os
os.environ['OMP_NUM_THREADS'] = '1'
import json
import pandas as pd
from typing import Dict, List, Optional
from sklearn.model_selection import train_test_split

# Import preprocessing utilities
from utils.utils_preprocessing import (
    encode_categorical_features, encode_binary_target, 
    handle_categorical_missing, remove_constant_features
)
from utils.utils_datasets import load_file
from constants import dataset_filenames, target_columns
from execution_functions import (
    evaluate_synthetic_dataset,
    evaluate_real_dataset,
    initialize_and_run_feature_selection,
    specific_dataset_cleanings
)

def get_applicable_methods(n_features: int, task_type: str) -> List[str]:
    """
    Determine which feature selection methods are applicable based on dataset characteristics.
    
    Parameters:
    -----------
    n_features : int
        Number of features in the dataset
    task_type : str
        Type of task ('regression', 'binary_classification', 'multiclass_classification')
        
    Returns:
    --------
    list
        Names of applicable feature selection methods
    """
    base_methods = [
        'chi_squared',
        'mutual_info',
        'mrmr',
        'fcbf',
        'relief',
        'rf_fimportances',
        'cb_fimportances', 
        'permutation_importance',
        'shap',
        'boruta',
        'hybrid_rfe',
        'hybrid_nmi_sfs',
        'hybrid_fcbf_sfs',
        'hybrid_shap_sfs'
    ]
    
    if n_features <= 100:
        base_methods.extend(['seq_backward', 'seq_forward', 'seq_forward_floating', 'seq_backward_floating'])
                    
    if task_type == 'binary_classification':
        base_methods.append('information_value')
    
    if task_type != 'multiclass_classification':
        base_methods.append('correlation')
    
    return base_methods

class IncrementalFeatureSelection:
    """
    A class for performing incremental feature selection on both synthetic and real-world datasets.
    
    This class implements a pipeline for systematically evaluating different feature selection
    methods across multiple datasets. It handles both synthetic datasets (with known ground truth)
    and real-world datasets, managing the entire process from data loading to result storage.
    
    Attributes:
        methods (List[str], optional): List of feature selection methods to evaluate.
            If None, all applicable methods will be used.
        synthetic_results_file (str): Path to store results for synthetic datasets.
        real_results_file (str): Path to store results for real-world datasets.
    """
    
    def __init__(self, methods: Optional[List[str]] = None):
        """
        Initialize the incremental feature selection pipeline.
        
        Args:
            methods (List[str], optional): Specific feature selection methods to evaluate.
                If None, all applicable methods will be used based on dataset characteristics.
        """
        self.methods = methods
        self.synthetic_results_file = 'logs/benchmark/synthetic_benchmark_results.json'
        self.real_results_file = 'logs/benchmark/real_world_benchmark_results.json'
        os.makedirs('logs', exist_ok=True)

    def load_existing_results(self, file_path: str) -> Dict:
        """
        Load previously saved results from a JSON file.
        
        Args:
            file_path (str): Path to the JSON file containing previous results.
            
        Returns:
            Dict: Dictionary containing previous results or empty dict if file doesn't exist.
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def save_results(self, results: Dict, file_path: str):
        """
        Save feature selection results to a JSON file.
        
        Args:
            results (Dict): Results to save, typically containing performance metrics.
            file_path (str): Path where the results should be saved.
        """
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)

    def update_results(self, results: Dict, dataset_name: str, method: str, new_result: Dict):
        """
        Update the results dictionary with new evaluation results.
        
        Args:
            results (Dict): Existing results dictionary to update.
            dataset_name (str): Name of the dataset being evaluated.
            method (str): Name of the feature selection method.
            new_result (Dict): New evaluation results to add.
            
        Returns:
            Dict: Updated results dictionary.
        """
        if dataset_name not in results:
            results[dataset_name] = {}
        results[dataset_name][method] = new_result
        return results

    def process_synthetic_dataset(self, dataset_idx: int) -> None:
        """
        Process and evaluate feature selection methods on a synthetic dataset.
        
        This method handles the complete pipeline for synthetic datasets:
        1. Loads dataset and its metadata
        2. Determines applicable feature selection methods
        3. Runs each method and evaluates performance
        4. Saves results incrementally
        
        Args:
            dataset_idx (int): Index of the synthetic dataset to process.
            
        Note:
            Results are saved to the synthetic_results_file path.
            Errors during processing are caught and logged but don't halt execution.
        """
        print(f"\n{'=' * 50}\nProcessing synthetic dataset {dataset_idx}\n{'=' * 50}")
        
        # Load and validate dataset files
        dataset_path = f'data/synthetic_datasets/datasets/dataset_{dataset_idx}'
        try:
            X = pd.read_csv(os.path.join(dataset_path, 'features.csv'))
            y = pd.read_csv(os.path.join(dataset_path, 'target.csv')).iloc[:, 0]
            
            with open(os.path.join(dataset_path, 'metadata.json'), 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error loading dataset {dataset_idx}: {str(e)}")
            return

        # Initialize results tracking
        results = self.load_existing_results(self.synthetic_results_file)
        dataset_name = f'dataset_{dataset_idx}'

        # Determine which methods to run
        applicable_methods = get_applicable_methods(X.shape[1], metadata['task_type'])
        methods_to_run = self.methods if self.methods else applicable_methods
        methods_to_run = [m for m in methods_to_run if m in applicable_methods]

        balanced_status = 'Balanced' if metadata['is_balanced'] else 'Imbalanced'

        # Run each feature selection method
        for method_name in methods_to_run:

            try:
                print(f"Running {method_name} on {dataset_name}")
                
                # Run feature selection and time execution
                X_selected, execution_time = initialize_and_run_feature_selection(
                    method_name, X, y, metadata['task_type'],
                    metadata.get('categorical_features', []), balanced_status
                )

                if X_selected is not None:
                    # Evaluate selected features against ground truth
                    result = evaluate_synthetic_dataset(
                        X_selected, metadata['ground_truth_features'],
                        metadata['interactive_features'], metadata['informative_features'],
                        metadata['feature_importances'], execution_time, X
                    )
                    
                    # Save results incrementally
                    results = self.update_results(results, dataset_name, method_name, result)
                    self.save_results(results, self.synthetic_results_file)
                    print(f"Successfully completed {method_name} on {dataset_name}")
                    
            except Exception as e:
                print(f"Error processing {method_name} on {dataset_name}: {str(e)}")
                continue

    def process_real_dataset(self, dataset_idx: int) -> None:
        """
        Process and evaluate feature selection methods on a real-world dataset.
        
        This method implements the complete pipeline for real-world datasets:
        1. Loads and preprocesses the dataset
        2. Handles preprocessing steps
        3. Splits data into train/test sets
        4. Runs and evaluates feature selection methods
        5. Saves results incrementally
        
        Args:
            dataset_idx (int): Index of the real-world dataset to process.
            
        Note:
            Results are saved to the real_results_file path.
            Includes specific data cleaning steps for certain datasets.
            Errors during processing are caught and logged but don't halt execution.
        """
        # Initialize dataset processing
        file = dataset_filenames[dataset_idx]
        target_column = target_columns[dataset_idx]
        dataset_name = os.path.splitext(file)[0]
        
        print(f"\n{'=' * 50}\nProcessing dataset: {dataset_name}\n{'=' * 50}")

        try:
            # Load dataset and metadata
            dataset_path = os.path.join("data/real_world_datasets/datasets", file)
            with open('data/real_world_datasets/metadata/datasets_metadata.json', 'r') as f:
                metadata = json.load(f)[dataset_name]

            results = self.load_existing_results(self.real_results_file)

            # Prepare features and target
            df = load_file(dataset_path)
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Apply dataset-specific preprocessing
            X_preprocessed, y_preprocessed = specific_dataset_cleanings(X, dataset_name, y)

            # Handle target variable encoding
            if metadata['task'] == 'multiclass_classification':
                y_preprocessed = encode_categorical_features(y_preprocessed.copy())
            elif metadata['task'] == 'binary_classification':
                y_preprocessed = encode_binary_target(y_preprocessed.copy(), metadata['balance_status'])

            # Preprocess features
            X_transformed = handle_categorical_missing(X_preprocessed)
            X_cleaned = remove_constant_features(X_transformed)

            # Create train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_cleaned, y_preprocessed, test_size=0.2, random_state=42
            )

            # Determine applicable methods
            applicable_methods = get_applicable_methods(X_train.shape[1], metadata['task'])
            methods_to_run = self.methods if self.methods else applicable_methods
            methods_to_run = [m for m in methods_to_run if m in applicable_methods]

            categorical_features = X_train.select_dtypes(
                include=['object', 'category']
            ).columns.tolist()

            # Run each feature selection method
            for method_name in methods_to_run:
                
                
                try:
                    print(f"Running {method_name} on {dataset_name}")
                    
                    # Run feature selection
                    X_train_selected, execution_time = initialize_and_run_feature_selection(
                        method_name, X_train, y_train, metadata['task'],
                        categorical_features, metadata['balance_status']
                    )

                    if X_train_selected is not None:
                        # Prepare selected features for evaluation
                        categorical_features_selected = X_train_selected.select_dtypes(
                            include=['object', 'category']
                        ).columns.tolist()
                        X_test_selected = X_test[X_train_selected.columns]
                        
                        # Evaluate performance
                        result = evaluate_real_dataset(
                            X_train_selected, X_test_selected, y_train, y_test,
                            metadata['task'], execution_time,
                            categorical_features_selected, metadata['balance_status']
                        )
                        
                        # Save results incrementally
                        results = self.update_results(results, dataset_name, method_name, result)
                        self.save_results(results, self.real_results_file)
                        print(f"Successfully completed {method_name} on {dataset_name}")

                except Exception as e:
                    print(f"Error processing {method_name} on {dataset_name}: {str(e)}")
                    continue

        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")