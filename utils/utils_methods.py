import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import label_binarize


def get_predictions(model, X, task, classes=None):
    """
    Get predictions using the appropriate prediction method.
    
    Parameters:
        model: fitted sklearn-compatible estimator
        X: Feature matrix
        prediction_type: str, type of prediction needed
        classes: array-like, optional, unique classes for multiclass problems
        
    Returns:
        Model predictions in the appropriate format
    """
    if task == 'multiclass_classification':
        # Get probability predictions for each class
        return model.predict_proba(X)
    elif task == 'binary_classification':
        # For binary classification, return probability of positive class
        return model.predict_proba(X)[:, 1]
    else:  # 'predict' or 'regression'
        return model.predict(X)

def check_classification_targets(y):
    """
    Check if target variable has enough classes in the dataset.
    
    Parameters:
        y: Target variable
        
    Returns:
        bool: True if targets are valid for classification
    """
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        raise ValueError("Target variable must have at least two unique classes for classification.")
    return True

def compute_cv_score(X, y, model, cv, scorer, task):
    """
    Compute the cross-validated score for the current feature set.
    
    This method handles all types of prediction tasks (binary classification,
    multiclass classification, and regression) by using the appropriate
    prediction method based on the detected prediction type.
    
    Parameters:
        X: Feature matrix with current set of features
        y: Target variable (binary, multiclass, or continuous)
        model: sklearn-compatible estimator
        cv: cross-validation splitter
        scorer: scoring function
        prediction_type: str, type of prediction needed
            
    Returns:
        float: Mean cross-validated score across all folds
    """
    scores = []
    
    # For classification, verify we have enough classes
    if task in ['binary_classification', 'multiclass_classification']:
        check_classification_targets(y)
        classes = np.unique(y)
    else:
        classes = None
    
    # Iterate through cross-validation folds
    for train_idx, val_idx in cv.split(X, y):
        # Split data into training and validation sets for this fold
        if isinstance(X, pd.DataFrame):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
            
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        else:
            y_train, y_val = y[train_idx], y[val_idx]
        
        # Verify we have all classes in both training and validation sets
        if task in ['binary_classification', 'multiclass_classification']:
            train_classes = np.unique(y_train)
            val_classes = np.unique(y_val)
            if len(train_classes) < len(classes) or len(val_classes) < len(classes):
                continue  # Skip this fold if not all classes are present
        
        # Train the model on the training data
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        try:
            y_pred = get_predictions(model_clone, X_val, task, classes)
            
            if task == 'multiclass_classification' and 'average_precision_score' in str(scorer._score_func):
                y_val_bin = label_binarize(y_val, classes=classes)
                score = scorer._score_func(y_val_bin, y_pred, **scorer._kwargs)
            else:
                score = scorer._score_func(y_val, y_pred, **scorer._kwargs)
                
            scores.append(score)
            
        except ValueError as e:
            print(f"Warning: Scoring error in fold: {str(e)}")
            continue
    
    if not scores:
        raise ValueError("No valid folds found for scoring. Ensure your dataset has enough samples of each class.")
    
    return np.mean(scores)