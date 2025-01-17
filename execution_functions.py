
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    roc_auc_score,  
    average_precision_score, 
    make_scorer, 
    mean_absolute_error
)
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from utils.utils_preprocessing import encode_categorical_features, CustomLabelEncoder, ImputeMissing
from params_config import (
    fast_cb_bin_params, 
    fast_cb_regression_params, 
    fast_cb_multi_params, 
    rf_params
)
# Import preprocessing utilities
from utils.utils_preprocessing import (
    encode_categorical_features, kmeans_discretize, ImputeMissing, target_encode_variables
)
from utils.utils_methods import get_predictions

# Import feature selection methods (same imports as original)
from feature_selection_methods.filter.bivariate.chi_squared import ChiSquaredSelector
from feature_selection_methods.filter.bivariate.correlation import CorrelationSelector
from feature_selection_methods.filter.bivariate.information_value import WOEInformationValueSelector
from feature_selection_methods.filter.bivariate.norm_mutual_info import NormalizedMutualInfoSelector
from feature_selection_methods.filter.multivariate.mrmr import MRMRSelector
from feature_selection_methods.filter.multivariate.relief_algorithms import ReliefSelector
from feature_selection_methods.filter.multivariate.fcbf import FCBFSelector
from feature_selection_methods.wrapper.backward_elimination.sequential_backward_selection import SeqBackSelectorCV
from feature_selection_methods.wrapper.forward_selection.sequential_forward_selection import SeqForwardSelectorCV
from feature_selection_methods.wrapper.bidirectional.sequential_forward_floating_selection import SeqForwardFloatingSelectorCV
from feature_selection_methods.wrapper.bidirectional.sequential_backward_floating_selection import SeqBackFloatingSelectorCV
from feature_selection_methods.hybrid.embedded_wrapper.recursive_feature_elimination import HybridRFE
from feature_selection_methods.hybrid.filter_wrapper.fcbf_sfs import HybridFcbfSfs
from feature_selection_methods.hybrid.filter_wrapper.nmi_sfs import HybridNmiSfs
from feature_selection_methods.hybrid.advanced_wrapper.shap_sfs import HybridShapSfs
from feature_selection_methods.embedded.importance.permutation_feature_importance import PermutationImportanceSelector
from feature_selection_methods.embedded.importance.rf_feature_importances import RFFearureImportanceSelector
from feature_selection_methods.embedded.importance.cb_feature_importances import CatBoostFeatureImportanceSelector
from feature_selection_methods.advanced.shap import ShapFeatureImportanceSelector
from feature_selection_methods.advanced.boruta import CatBoostBoruta

def evaluate_synthetic_dataset(X_selected, ground_truth_features, interactive_features, informative_features, feature_importances, execution_time, X_original):
    """
    Evaluate feature selection on synthetic datasets using weighted accuracy.
    
    Parameters:
    -----------
    X_selected : pd.DataFrame
        DataFrame containing only the selected features
    ground_truth_features : list
        List of feature names that are truly relevant
    interactive_features : list
        List of interactive features
    informative_features : list
        List of informative features 
    feature_importances : dict
        Dictionary mapping ground truth features to their importance (0-100)
    execution_time : float
        Time taken to run the feature selection
    X_original : pd.DataFrame
        Original DataFrame with all features
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    selected_features = set(X_selected.columns)
    ground_truth = set(ground_truth_features)
    all_features = set(X_original.columns)
    
    # Split evaluation into ground truth and non-ground truth parts
    n_total_features = len(all_features)
    n_ground_truth = len(ground_truth)
    n_non_ground_truth = n_total_features - n_ground_truth
    
    # Calculate weights for both parts
    ground_truth_weight = n_ground_truth / n_total_features
    non_ground_truth_weight = n_non_ground_truth / n_total_features
    
    # Evaluate ground truth features (weighted by importance)
    ground_truth_score = 0
    total_importance = sum(feature_importances.values())  # Should be 100
    
    # True Positives: Selected features that are actually ground truth
    for feature in ground_truth:
        if feature in selected_features:
            ground_truth_score += feature_importances[feature] / total_importance
            
    # Evaluate non-ground truth features
    non_ground_truth_features = all_features - ground_truth
    
    # True Negatives: Correctly not selected non-ground truth features
    true_negatives = len(non_ground_truth_features - selected_features)
        
    # Calculate non-ground truth score accounting for both TN and FP
    if n_non_ground_truth > 0:
        non_ground_truth_accuracy = true_negatives / n_non_ground_truth
    else:
        non_ground_truth_accuracy = 0
    
    # Combine both parts with their respective weights
    weighted_accuracy = (
        (ground_truth_weight * ground_truth_score) + 
        (non_ground_truth_weight * non_ground_truth_accuracy)
    )
    
    # Calculate traditional accuracy for comparison
    traditional_accuracy = (
        len(selected_features & ground_truth) + 
        len(all_features - selected_features - ground_truth)
    ) / n_total_features
    
    # Calculate metrics for interactive and informative features
    n_interactive_total = len(interactive_features)
    # Check if there are any interactive features
    if n_interactive_total > 0:
        selected_interactive = selected_features.intersection(interactive_features)
        interactive_selection_rate = len(selected_interactive) / n_interactive_total
    else:
        interactive_selection_rate = 'N/A'

    # For informative features, exclude interactive features to avoid double counting
    n_informative_total = len(informative_features)
    selected_informative = selected_features.intersection(informative_features)
    informative_selection_rate = len(selected_informative) / n_informative_total if n_informative_total > 0 else 0
    
    return {
        'weighted_accuracy': weighted_accuracy,
        'ground_truth_accuracy': ground_truth_score,
        'non_ground_truth_accuracy': non_ground_truth_accuracy,
        'interactive_selection_rate': interactive_selection_rate,
        'informative_selection_rate': informative_selection_rate,
        'execution_time': execution_time,
        'n_selected_features': len(selected_features)
    }

def get_model_and_scorer(problem_type, balanced_status, categorical_features, n_samples):
   """
   Create an appropriately configured CatBoost model and scoring metric based on the problem type.
   
   This function configures CatBoost models with appropriate parameters and scoring metrics
   for different machine learning tasks (regression, binary and multiclass classification).
   It handles both balanced and imbalanced datasets.
   
   Args:
       problem_type (str): Type of machine learning problem - 'regression', 
           'binary_classification', or 'multiclass_classification' 
       balanced_status (str): Whether the dataset is 'Balanced' or 'Imbalanced'
       categorical_features (list): List of categorical feature column names
       n_samples (int): Number of samples in the dataset, used to set min_data_in_leaf
           
   Returns:
       tuple: Contains:
           - model: Configured CatBoost model instance
           - params: Dictionary of model parameters
           - scorer: Sklearn scorer object with appropriate metric
           - cv: Cross-validation splitter (KFold or StratifiedKFold)
           - str: Direction of optimization ('minimize' or 'maximize')
   """
   # Set base parameters common to all model types
   # min_data_in_leaf is set to 1% of 2/3 of total samples (training set size)
   base_params = {
       'min_data_in_leaf': round(n_samples * 2/3 * 0.01),
       'cat_features': categorical_features
   }
   
   # Handle regression case
   if problem_type == 'regression':
       model = CatBoostRegressor(**fast_cb_regression_params, **base_params)
       scorer = make_scorer(mean_absolute_error, greater_is_better=False)
       cv = KFold(n_splits=3, shuffle=True, random_state=42)
       params = fast_cb_regression_params
       return model, params, scorer, cv, 'minimize'
   
   # For classification problems, use stratified cross-validation
   cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
   
   # Handle binary classification
   if problem_type == 'binary_classification':
       model = CatBoostClassifier(**fast_cb_bin_params, **base_params)
       
       # Use average precision for imbalanced data, ROC-AUC for balanced
       if balanced_status == 'Imbalanced':
           scorer = make_scorer(average_precision_score, needs_proba=True)
       else: 
           scorer = make_scorer(roc_auc_score, needs_proba=True)
       params = fast_cb_bin_params

   # Handle multiclass classification
   else:  
       model = CatBoostClassifier(**fast_cb_multi_params, **base_params)
       
       # Use average precision for imbalanced data, ROC-AUC (one-vs-rest) for balanced
       if balanced_status == 'Imbalanced':
           scorer = make_scorer(average_precision_score, needs_proba=True)
       else:
           scorer = make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
       params = fast_cb_multi_params

   return model, params, scorer, cv, 'maximize'

def evaluate_real_dataset(X_train_selected, X_test, y_train, y_test, problem_type, execution_time, categorical_features, balanced_status):
    """
    Evaluate feature selection on real-world datasets using the test set.
    
    Parameters:
    -----------
    X_train_selected : pd.DataFrame
        Training data with selected features
    X_test : pd.DataFrame
        Test data with selected features (already filtered to selected features)
    y_train : array-like
        Target variable for training
    y_test : array-like
        Target variable for test set
    problem_type : str
        Type of problem ('regression', 'binary_classification', 'multiclass_classification')
    execution_time : float
        Time taken by feature selection
    categorical_features : list
        List of categorical feature names
    balanced_status : str
        Whether the dataset is 'Balanced' or 'Imbalanced'
        
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics
    """
    # Check if any features were selected
    if X_train_selected.empty or len(X_train_selected.columns) == 0:
        if problem_type == 'regression':
            mean_prediction = y_train.mean()
            worst_mae = np.mean(np.abs(y_test - mean_prediction))
            return {
                'models_performance': {'catboost': worst_mae, 'random_forest': worst_mae},
                'execution_time': execution_time,
                'n_selected_features': 0
            }
        else:
            return {
                'models_performance': {'catboost': 0, 'random_forest': 0},
                'execution_time': execution_time,
                'n_selected_features': 0
            }
    
    # Filter categorical features to only include those present in X_train_selected
    valid_cat_features = [col for col in categorical_features if col in X_train_selected.columns]
    
    # Get appropriate scorer
    _, params, scorer, _, _ = get_model_and_scorer(
        problem_type, 
        balanced_status,
        valid_cat_features,
        X_train_selected.shape[0]
    )
    model_scores = {'catboost': None, 'random_forest': None}
    
    # Initialize preprocessors for Random Forest
    imputer = ImputeMissing()
    label_encoder = CustomLabelEncoder()
    
    # Initialize models based on problem type with appropriate parameters
    if problem_type == 'regression':
        models = {
            'catboost': CatBoostRegressor(cat_features=valid_cat_features, **params),
            'random_forest': RandomForestRegressor(**rf_params)
        }
    else: 
        models = {
            'catboost': CatBoostClassifier(cat_features=valid_cat_features, **params),
            'random_forest': RandomForestClassifier(**rf_params)
        }
    
    # Evaluate each model type
    for model_name, model in models.items():
        try:
            if model_name == 'random_forest':
                # For Random Forest: apply imputation and encoding
                X_train_processed = label_encoder.fit_transform(
                    imputer.fit_transform(X_train_selected)
                )
                X_test_processed = label_encoder.transform(
                    imputer.transform(X_test)
                )
            else:
                # For CatBoost: use original data with missing values
                X_train_processed = X_train_selected
                X_test_processed = X_test
            
            # Train and evaluate model
            model.fit(X_train_processed, y_train)
            y_pred = get_predictions(model, X_test_processed, problem_type)
            
            # Handle multiclass imbalanced case specially
            if (problem_type == 'multiclass_classification' and 
                balanced_status == 'Imbalanced' and 
                'average_precision_score' in str(scorer._score_func)):
                y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
                score = scorer._score_func(y_test_bin, y_pred, **scorer._kwargs)
            else:
                score = scorer._score_func(y_test, y_pred, **scorer._kwargs)
                
            model_scores[model_name] = score
            
        except Exception as e:
            print(f"Warning: Error evaluating {model_name}: {str(e)}")
            if problem_type == 'regression':
                mean_prediction = y_train.mean()
                worst_mae = np.mean(np.abs(y_test - mean_prediction))
                model_scores[model_name] = worst_mae
            else:
                model_scores[model_name] = 0
    
    return {
        'models_performance': model_scores,
        'execution_time': execution_time,
        'n_selected_features': len(X_train_selected.columns)
    }

def initialize_and_run_feature_selection(method_name, X, y, task_type, categorical_features, is_balanced):
   """
   Initialize and execute a specified feature selection method on a dataset.
   
   This function handles the complete feature selection pipeline:
   1. Preprocessing data for the specific method
   2. Creating and configuring the feature selector
   3. Fitting and transforming the data
   4. Measuring execution time
   
   Args:
       method_name (str): Name of the feature selection method to use
       X (pd.DataFrame): Input features to perform selection on
       y (array-like): Target variable
       task_type (str): Type of ML task ('regression', 'binary_classification' or 'multiclass_classification)
       categorical_features (list): List of categorical feature names
       is_balanced (str): Dataset balance status ('Balanced' or 'Imbalanced')
       
   Returns:
       tuple: Contains:
           - pd.DataFrame or None: Selected features if successful, None if failed
           - float: Execution time in seconds (0 if failed)
   """
   # Preprocess data according to method-specific requirements
   X_processed, y_processed = preprocess_methods(
       X, y, task_type, method_name
   )
   
   # Start timing the feature selection process
   start_time = time.time()
   
   # Identify categorical columns in processed data
   categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns.tolist()
   
   # Initialize the appropriate feature selector
   selector = initialize_selector(
       method_name, 
       task_type, 
       X_processed.shape[0], 
       categorical_cols, 
       is_balanced
   )
   
   # Early return if selector initialization failed
   if selector is None:
       return None, 0
       
   try:
       # Apply feature selection
       X_selected_processed = selector.fit_transform(X_processed, y_processed)
       
       # Map selected features back to original dataset
       selected_columns = X_selected_processed.columns
       X_selected = X[selected_columns]
       
       # Return selected features and execution time
       return X_selected, time.time() - start_time
       
   except Exception as e:
       # Handle any errors during feature selection
       print(f"Error in {method_name}: {str(e)}")
       return None, 0

def initialize_selector(method_name, task_type, n_samples, categorical_features, balanced_status):
   """
   Initialize and configure a feature selection method based on specified parameters.
   
   This function serves as a factory for creating feature selectors, configuring them
   with appropriate parameters based on the task type and data characteristics.
   
   Args:
       method_name (str): Name of the feature selection method to initialize
       task_type (str): Type of ML task (regression/classification)
       n_samples (int): Number of samples in dataset
       categorical_features (list): List of categorical feature names
       balanced_status (str): Whether dataset is 'Balanced' or 'Imbalanced'
       
   Returns:
       object: Initialized feature selector instance configured for the specified task
   """
   # Get appropriate model, scoring metric and CV strategy for the task
   model, params, scorer, cv, metric_direction = get_model_and_scorer(
       task_type, 
       balanced_status,
       categorical_features, 
       n_samples
   )
   
   # Initialize Random Forest model for embedded method rf_fimportances
   rf_embedded = (RandomForestRegressor if task_type == 'regression' else RandomForestClassifier)(
       **rf_params,
       min_samples_leaf=max(1, round(n_samples*2/3*0.01))
   )
   
   # Dictionary mapping method names to their initialization functions
   selector_map = {
       
       # Filter
       'chi_squared': lambda: ChiSquaredSelector(
           alpha=0.05  
       ),
       'information_value': lambda: WOEInformationValueSelector(
           threshold_iv=0.02, 
           task=task_type
       ),
       'correlation': lambda: CorrelationSelector(
           threshold=0.1,  
           task=task_type
       ),
       'mutual_info': lambda: NormalizedMutualInfoSelector(
           threshold=0.01,  
       ),
       'mrmr': lambda: MRMRSelector(
           threshold=0.01  
       ),
       'fcbf': lambda: FCBFSelector(
           threshold=0.01  
       ),
       'relief': lambda: ReliefSelector(
           k=10, 
           threshold=0.1,  
           sigma=0.5,  
           n_jobs=-1, 
           task=task_type
       ),
       
       # Wrapper 
       'seq_backward': lambda: SeqBackSelectorCV(
           model=model, scorer=scorer, metric_direction=metric_direction,
           cv=cv, categorical_features=categorical_features, task=task_type,
           loss_threshold=0.001,  
           min_features=1  
       ),
       'seq_forward': lambda: SeqForwardSelectorCV(
           model=model, scorer=scorer, metric_direction=metric_direction,
           cv=cv, categorical_features=categorical_features, task=task_type,
           min_improvement=0.001  
       ),
       'seq_forward_floating': lambda: SeqForwardFloatingSelectorCV(
           model=model, scorer=scorer, metric_direction=metric_direction,
           cv=cv, categorical_features=categorical_features, task=task_type,
           min_improvement=0.001
       ),
       'seq_backward_floating': lambda: SeqBackFloatingSelectorCV(
           model=model, scorer=scorer, metric_direction=metric_direction,
           loss_threshold=0.001, min_features=1, cv=cv, 
           categorical_features=categorical_features, task=task_type
       ),
       
       # Embedded 
       'hybrid_rfe': lambda: HybridRFE(
           model=model, scorer=scorer, metric_direction=metric_direction,
           cv=cv, categorical_features=categorical_features, task=task_type,
           score_threshold=0.015, 
           min_features=1  
       ),
       'rf_fimportances': lambda: RFFearureImportanceSelector(
           rf_model=rf_embedded, 
           threshold=0.01 
       ),
       'cb_fimportances': lambda: CatBoostFeatureImportanceSelector(
           model=model, 
           threshold=0.01  
       ),
       'permutation_importance': lambda: PermutationImportanceSelector(
           model=model, scorer=scorer, metric_direction=metric_direction,
           n_repeats=5, 
           cv=cv, task=task_type, threshold=0.01
       ),
       
       # Advanced
       'shap': lambda: ShapFeatureImportanceSelector(
           model=model, cv=cv, threshold=0.01, task=task_type, 
           operation_timeout=300  
       ),
       'boruta': lambda: CatBoostBoruta(
           model_params=params, n_iterations=100, alpha=0.05, random_state=42, 
           categorical_features=categorical_features
       ),
       
       # Hybrid 
       'hybrid_fcbf_sfs': lambda: HybridFcbfSfs(
           model=model, scorer=scorer, metric_direction=metric_direction,
           cv=cv, categorical_features=categorical_features, task=task_type,
           min_improvement=0.001, fcbf_threshold=0.001
       ),
       'hybrid_nmi_sfs': lambda: HybridNmiSfs(
           model=model, scorer=scorer, metric_direction=metric_direction,
           cv=cv, categorical_features=categorical_features, task=task_type,
           min_improvement=0.001, threshold=0.001
       ),
       'hybrid_shap_sfs': lambda: HybridShapSfs(
           model=model, scorer=scorer, metric_direction=metric_direction,
           cv=cv, categorical_features=categorical_features, task=task_type,
           min_improvement=0.001, operation_timeout=300, threshold=0.005
       )
   }
   
   # Return initialized selector or None if method not found
   return selector_map.get(method_name, lambda: None)()

def specific_dataset_cleanings(X, dataset_name, y):
    """
    Apply dataset-specific cleaning operations based on the dataset name.
    This function handles known issues or requirements for specific datasets.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        The input dataset to be cleaned
    dataset_name : str
        Name of the dataset to determine which specific cleaning operations to apply
        
    Returns:
    --------
    pandas.DataFrame
        The cleaned dataset with dataset-specific transformations applied
        
    Notes:
    ------
    Current dataset-specific operations:
    - speeddating: Removes 'decision' and 'decision_o' columns due to data leakage
      These columns contain information that would not be available at prediction time
      and could lead to overly optimistic model performance
      
    - weatherAUS: Removes samples where there the target is a missing value
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    X_cleaned = X.copy()
    y_cleaned = y.copy()
    
    # Handle speeddating dataset
    if dataset_name == 'speeddating':
        # Remove columns that could cause data leakage
        leakage_columns = ['decision', 'decision_o']
        
        # Check if the columns exist before trying to drop them
        existing_leakage_columns = [col for col in leakage_columns if col in X_cleaned.columns]
        if existing_leakage_columns:
            X_cleaned = X_cleaned.drop(columns=existing_leakage_columns)
            print(f"Removed data leakage columns from {dataset_name}: {existing_leakage_columns}")
            
    elif dataset_name == 'weatherAUS':
        # Get indices where target is not NaN
        valid_indices = ~y_cleaned.isna()
        
        # Filter both X and y to keep only samples where target is not NaN
        X_cleaned = X_cleaned[valid_indices]
        y_cleaned = y_cleaned[valid_indices]
        
        n_removed = sum(~valid_indices)
        print(f"Removed {n_removed} samples with NaN targets from {dataset_name}")
        print(f"Remaining samples: {len(y_cleaned)}")
    
        # Handle speeddating dataset
    elif dataset_name == 'irish':
        # Remove columns that could cause data leakage
        leakage_columns = ['Educational_level']
        
        # Check if the columns exist before trying to drop them
        existing_leakage_columns = [col for col in leakage_columns if col in X_cleaned.columns]
        if existing_leakage_columns:
            X_cleaned = X_cleaned.drop(columns=existing_leakage_columns)
            print(f"Removed data leakage columns from {dataset_name}: {existing_leakage_columns}")
    
    return X_cleaned, y_cleaned


def preprocess_methods(X, y, task_type, method_name, num_bins=5):
    """
    Preprocess data specifically for different feature selection methods.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input features
    y : pd.Series or pd.DataFrame
        Target variable
    task_type : str
        Type of task ('regression', 'binary_classification', 'multiclass_classification')
    num_bins : int, default=5
        Number of bins for discretizing continuous features
            
    Returns:
    --------
    tuple : (pd.DataFrame, pd.Series)
        Preprocessed features and target
    """
    X_transformed = X.copy()
    y_transformed = y.copy()
    
    sampe_process_list = ['mutual_info', 'fcbf', 'mrmr']

    
    if method_name == 'chi_squared':
        
        # ensure it's a Series
        if isinstance(y_transformed, pd.DataFrame):
            y_transformed = y_transformed.iloc[:, 0]
        else:
            y_transformed = pd.Series(y_transformed)
            
        # Impute if there are missing values
        if X_transformed.isna().any().any():
            X_transformed = ImputeMissing().fit_transform(X_transformed) 
                
        # Discretize continuous features
        X_transformed = kmeans_discretize(X_transformed, num_bins)
            
        if task_type == 'regression':
            y_transformed = kmeans_discretize(y_transformed, num_bins)
    
    elif method_name == 'correlation':
        
        # Ensure y is a Series
        if isinstance(y_transformed, pd.DataFrame):
            y_transformed = y_transformed.iloc[:, 0]
        
        # Impute if there are missing values
        if X_transformed.isna().any().any():
            X_transformed = ImputeMissing().fit_transform(X_transformed)
            
        # Get encoded DataFrame with categorical features transformed
        X_transformed = target_encode_variables(X_transformed, y_transformed)
        
    elif method_name == 'information_value':
        
        # Ensure y is a Series
        if isinstance(y_transformed, pd.DataFrame):
            y_transformed = y_transformed.iloc[:, 0]
            
        # Impute if there are missing values
        if X_transformed.isna().any().any():
            X_transformed = ImputeMissing().fit_transform(X_transformed)
                
        # Discretize numerical variables
        X_transformed = kmeans_discretize(X_transformed, num_bins)
        
    elif method_name in sampe_process_list:
                
        # Handle target variable - ensure it's a Series
        if isinstance(y_transformed, pd.DataFrame):
            y_transformed = y_transformed.iloc[:, 0]  # Take first column if DataFrame
        else:
            y_transformed = pd.Series(y_transformed)
            
        # Impute if there are missing values
        if X_transformed.isna().any().any():
            X_transformed = ImputeMissing().fit_transform(X_transformed)
        
        # Encode categorical variables
        X_transformed = encode_categorical_features(X_transformed)
        
        # Discretize numeric variables
        X_transformed = kmeans_discretize(X_transformed, num_bins)
        
        if task_type == 'regression':
            y_transformed = kmeans_discretize(y_transformed, num_bins)
            
    elif method_name == 'relief':
        
        # Impute if there are missing values
        if  X_transformed.isna().any().any():
            X_transformed = ImputeMissing().fit_transform(X_transformed)
        
        X_transformed = encode_categorical_features(X_transformed)
        
        # Handle different types of y input and convert to numpy
        if isinstance(y_transformed, pd.DataFrame):
            y_transformed = y_transformed.iloc[:, 0].values  # Convert to numpy immediately
        else:
            y_transformed = pd.Series(y_transformed).values if not isinstance(y_transformed, np.ndarray) else y
            
    elif method_name == 'rf_fimportances':
        
        # Random forest does not accept missings and categorical features
        
        # Impute if there are missing values
        if  X_transformed.isna().any().any():
            X_transformed = ImputeMissing().fit_transform(X_transformed)
        
        X_transformed = encode_categorical_features(X_transformed)
        
        # Handle target variable - ensure it's a Series
        if isinstance(y_transformed, pd.DataFrame):
            y_transformed = y_transformed.iloc[:, 0]  # Take first column if DataFrame
        else:
            y_transformed = pd.Series(y_transformed)
        
        
    return X_transformed, y_transformed







