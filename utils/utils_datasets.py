import pandas as pd
import os
import arff
import json


def load_file(path, sheet_name=0, delimiter='\t', orient='records', key=None):
    """
    Load file in different formats to a Dataframe, depending on the extension.
    
    Parameters:
    - path: Path to file.
    - sheet_name: Name or index of the sheet for excel files (the first, by default).
    - delimiter: Delimiter for text files (by default, tabulator).
    - orient: Orientation for JSON files  ('records', by default).
    
    Returns:
    - pandas DataFrame with loaded data.
    """
    
    # Get the file extensiomn
    ext = os.path.splitext(path)[1].lower()
    
    # Manaje the file based on the extension
    if ext == '.csv':
        return pd.read_csv(path)
    
    elif ext in ['.xlsx', '.xls']:
        return pd.read_excel(path, sheet_name=sheet_name)
    
    elif ext == '.json':
        return pd.read_json(path, orient=orient)
    
    elif ext == '.parquet':
        return pd.read_parquet(path)
    
    elif ext == '.pkl':
        return pd.read_pickle(path)
    
    elif ext in ['.txt', '.tsv']:
        return pd.read_csv(path, delimiter=delimiter)
    
    elif ext == '.xml':
        return pd.read_xml(path)
    
    elif ext == '.arff':
        with open(path) as f:
            dataset = arff.load(f)
        data = dataset['data']
        column_names = [attr[0] for attr in dataset['attributes']]
        return pd.DataFrame(data, columns=column_names)
    
    else:
        raise ValueError(f"The file format {ext} is not compatible.")


def categorize_dataset(df, target_column, data_name, imbalance_threshold_binary=0.80, imbalance_threshold_multi=0.60):
    """
    Categorizes a dataset based on:
    1. The number of variables (dimension):
    
    2. The type of variables (categorical or numerical):
       
    3. Problem type based on the target variable:
    
    4. Identification of imbalance in classification problems:

    Arguments:
    - df: pandas DataFrame.
    - target_column: Name of the target column (dependent variable).
    - data_name: Name to identify the dataset
    - imbalance_threshold_binary: Minimum proportion to consider a class Imbalanced in binary classification (default 80%).
    - imbalance_threshold_multi: Minimum proportion to consider a class Imbalanced in multiclass classification (default 60%).
    
    
    Returns:
    - A dictionary with the categoized info of the dataset including:
        - dimensionality: dimensional category
        - var_type: Main type of the variables
        - problem_type: Type of task
        - imbalance: Balance state
        - n_classes: number of classes (for classification tasks)
        - n_categorical: number of categorical variables
        - n_numerical: number of numerical variables
        - n_samples: number of samples
    """

    # Get DataFrame excluding target column
    df_features = df.drop(columns=[target_column])
    
    # Get the number of variables (columns) excluding the target
    num_vars = df_features.shape[1]
    
    # Dimension classification (low, mid, high)
    if num_vars <= 10:
        dimension_category = 'Low-Dimensional'
    elif 11 <= num_vars <= 100:
        dimension_category = 'Mid-Dimensional'
    else:
        dimension_category = 'High-Dimensional'
    
    # Count the number of categorical and numerical variables, excluding the target
    num_categorical = sum(df_features.dtypes == 'object') + sum(pd.api.types.is_categorical_dtype(df_features[col]) for col in df_features.columns)
    num_numerical = sum(pd.api.types.is_numeric_dtype(df_features[col]) for col in df_features.columns)

    # Proportions of categorical and numerical variables
    proportion_categorical = num_categorical / num_vars
    proportion_numerical = num_numerical / num_vars

    # Variable type classification
    if proportion_categorical >= 0.80:
        type_category = 'Primarily Categorical'
    elif proportion_numerical >= 0.80:
        type_category = 'Primarily Numerical'
    else:
        type_category = 'Mixed-Type'

    # Problem type classification based on the target
    target = df[target_column]
    n_classes = None
    
    if pd.api.types.is_numeric_dtype(target):
        unique_values = target.nunique()
        
        if unique_values > 20:  # Regression
            problem_type = 'regression'
            imbalance_info = 'N/A'
            n_classes = 'N/A'
        elif unique_values == 2:  # Binary Classification
            problem_type = 'binary_classification'
            n_classes = 2
            
            class_counts = target.value_counts(normalize=True)
            if any(class_counts > imbalance_threshold_binary):
                imbalance_info = 'Imbalanced'
            else:
                imbalance_info = 'Balanced'
        else:  # Multi-Class Classification
            problem_type = 'multiclass_classification'
            n_classes = unique_values
            
            class_counts = target.value_counts(normalize=True)
            if any(class_counts > imbalance_threshold_multi):
                imbalance_info = 'Imbalanced'
            else:
                imbalance_info = 'Balanced'
    else:
        if target.nunique() == 2:
            problem_type = 'binary_classification'
            n_classes = 2
            
            class_counts = target.value_counts(normalize=True)
            if any(class_counts > imbalance_threshold_binary):
                imbalance_info = 'Imbalanced'
            else:
                imbalance_info = 'Balanced'
        else:
            problem_type = 'multiclass_classification'
            n_classes = target.nunique()
            
            class_counts = target.value_counts(normalize=True)
            if any(class_counts > imbalance_threshold_multi):
                imbalance_info = 'Imbalanced'
            else:
                imbalance_info = 'Balanced'

    # Create dictionary with all information
    dataset_info = {
        'dimensionality': dimension_category,
        'var_type': type_category,
        'problem_type': problem_type,
        'imbalance': imbalance_info,
        'n_classes': n_classes,
        'n_categorical': num_categorical,
        'n_numerical': num_numerical,
        'n_samples': len(df)
    }

    return dataset_info


def generate_individual_metadata(dataset_info):
    """
    Generate detailed metadata for each individual dataset in the collection.
    
    This function processes the raw dataset information and creates a structured
    metadata dictionary for each dataset, including its characteristics and
    detailed statistics.
    
    Parameters
    ----------
    dataset_info : dict
        A dictionary containing the raw information for each dataset with structure:
        {
            'dataset_name': {
                'dimensionality': str ('High/Mid/Low-Dimensional'),
                'var_type': str ('Primarily Categorical/Numerical' or 'Mixed-Type'),
                'problem_type': str ('Regression/Binary Classification/Multi-Class Classification'),
                'imbalance': str ('Balanced/Imbalanced/N/A'),
                'n_classes': int or 'N/A',
                'n_categorical': int,
                'n_numerical': int,
                'n_samples': int
            }
        }
    
    Returns
    -------
    dict
        A dictionary containing structured metadata for each dataset with format:
        {
            'dataset_name': {
                'dimensionality': str,
                'variable_types': str,
                'task': str,
                'balance_status': str,
                'details': {
                    'num_classes': int or 'N/A',
                    'num_categorical_vars': int,
                    'num_numerical_vars': int,
                    'num_samples': int,
                    'total_variables': int
                }
            }
        }

    """
    return {
        dataset_name: {
            "dimensionality": info["dimensionality"],
            "variable_types": info["var_type"],
            "task": info["problem_type"],
            "balance_status": info["imbalance"] if info["problem_type"] != "Regression" else "N/A",
            "details": {
                "n_classes": info["n_classes"],
                "n_categorical_vars": info["n_categorical"],
                "n_numerical_vars": info["n_numerical"],
                "n_samples": info["n_samples"],
                "n_features": info["n_categorical"] + info["n_numerical"]
            }
        }
        for dataset_name, info in dataset_info.items()
    }

def generate_summary_metadata(dataset_info):
    """
    Generate summary statistics and counts for the entire dataset collection.
    
    This function analyzes the complete collection of datasets and produces
    aggregate statistics including counts of different types of datasets
    and average characteristics across the collection.
    
    Parameters
    ----------
    dataset_info : dict
        The raw dataset information dictionary with the same structure as in
        generate_individual_metadata()
    
    Returns
    -------
    dict
        A dictionary containing summary statistics with structure:
        {
            'dataset_counts': {
                'total_datasets': int,
                'dimensionality': {
                    'high_dimensional': int,
                    'mid_dimensional': int,
                    'low_dimensional': int
                },
                'variable_types': {...},
                'tasks': {...},
                'balance_status': {...}
            },
            'averages': {
                'avg_samples': float,
                'avg_variables': float,
                'avg_categorical_vars': float,
                'avg_numerical_vars': float
            }
        }
    """
    return {
        "dataset_counts": {
            "total_datasets": len(dataset_info),
            "dimensionality": {
                "high_dimensional": sum(1 for info in dataset_info.values() if info["dimensionality"] == "High-Dimensional"),
                "mid_dimensional": sum(1 for info in dataset_info.values() if info["dimensionality"] == "Mid-Dimensional"),
                "low_dimensional": sum(1 for info in dataset_info.values() if info["dimensionality"] == "Low-Dimensional")
            },
            "variable_types": {
                "primarily_categorical": sum(1 for info in dataset_info.values() if info["var_type"] == "Primarily Categorical"),
                "primarily_numerical": sum(1 for info in dataset_info.values() if info["var_type"] == "Primarily Numerical"),
                "mixed_type": sum(1 for info in dataset_info.values() if info["var_type"] == "Mixed-Type")
            },
            "tasks": {
                "regression": sum(1 for info in dataset_info.values() if info["problem_type"] == "regression"),
                "binary_classification": sum(1 for info in dataset_info.values() if info["problem_type"] == "binary_classification"),
                "multiclass_classification": sum(1 for info in dataset_info.values() if info["problem_type"] == "multiclass_classification")
            },
            "balance_status": {
                "balanced": sum(1 for info in dataset_info.values() if info["imbalance"] == "Balanced"),
                "imbalanced": sum(1 for info in dataset_info.values() if info["imbalance"] == "Imbalanced")
            }
        },
        "averages": {
            "avg_samples": sum(info["n_samples"] for info in dataset_info.values()) / len(dataset_info),
            "avg_variables": sum(info["n_categorical"] + info["n_numerical"] for info in dataset_info.values()) / len(dataset_info),
            "avg_categorical_vars": sum(info["n_categorical"] for info in dataset_info.values()) / len(dataset_info),
            "avg_numerical_vars": sum(info["n_numerical"] for info in dataset_info.values()) / len(dataset_info)
        }
    }

def print_summary(summary_metadata):
    """
    Print a formatted summary of the dataset collection metadata.
    
    This function takes the summary metadata and presents it in a
    human-readable format with clear sections and formatting.
    
    Parameters
    ----------
    summary_metadata : dict
        The summary metadata dictionary generated by generate_summary_metadata()
    """
    counts = summary_metadata["dataset_counts"]
    avgs = summary_metadata["averages"]
    
    print("\nDATASET COLLECTION SUMMARY")
    print("=" * 50)
    
    print(f"\nTotal Datasets: {counts['total_datasets']}")
    
    print("\nDimensionality Distribution:")
    for dim, count in counts['dimensionality'].items():
        print(f"- {dim}: {count}")
    
    print("\nVariable Types Distribution:")
    for var_type, count in counts['variable_types'].items():
        print(f"- {var_type}: {count}")
    
    print("\nTask Distribution:")
    for task, count in counts['tasks'].items():
        print(f"- {task}: {count}")
    
    print("\nBalance Status Distribution (Classification only):")
    for status, count in counts['balance_status'].items():
        print(f"- {status}: {count}")
    
    print("\nAverages:")
    print(f"- Average samples per dataset: {avgs['avg_samples']:.2f}")
    print(f"- Average variables per dataset: {avgs['avg_variables']:.2f}")
    print(f"- Average categorical variables: {avgs['avg_categorical_vars']:.2f}")
    print(f"- Average numerical variables: {avgs['avg_numerical_vars']:.2f}")