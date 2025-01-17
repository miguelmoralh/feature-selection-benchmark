import os
os.environ['OMP_NUM_THREADS'] = '5'
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder as SklearnLabelEncoder
from sklearn.preprocessing import KBinsDiscretizer

class ImputeMissing(BaseEstimator, TransformerMixin):
    """
    A custom transformer for handling missing values in both numerical and categorical features.
    
    This transformer inherits from scikit-learn's BaseEstimator and TransformerMixin to ensure
    compatibility with scikit-learn pipelines. It performs selective imputation by only
    processing features that contain missing values, making it more efficient than imputing
    all columns.
    
    Attributes:
        numerical_features (Index): Column names of numerical features in the dataset.
        categorical_features (Index): Column names of categorical features in the dataset.
        median_values (dict): Dictionary storing median values for numerical features with missing values.
    """
    def __init__(self):
        """
        Initialize the imputer with empty attributes. These will be populated during fitting.
        """
        self.numerical_features = None
        self.categorical_features = None
        self.median_values = {}
        
    def fit(self, X, y=None):
        """
        Fit the imputer by identifying numerical and categorical features and calculating
        necessary median values.
        
        Args:
            X (pd.DataFrame): Input features to fit the imputer on.
            y (array-like, optional): Target values, not used in this transformer.
            
        Returns:
            self: Returns the instance itself for method chaining.
        """
       # Separate features by data type
        self.numerical_features = X.select_dtypes(include=['number']).columns
        self.categorical_features = X.select_dtypes(include=['object', 'category']).columns
        
        # Calculate and store median values for ALL numerical features
        for feature in self.numerical_features:
            self.median_values[feature] = X[feature].median()
            
        return self

    def transform(self, X):
        """
        Transform the dataset by imputing missing values where they exist.
        
        For numerical features, missing values are replaced with the median calculated
        during fitting. For categorical features, missing values are replaced with
        the string 'Missing'.
        
        Args:
            X (pd.DataFrame): Dataset to transform.
            
        Returns:
            pd.DataFrame: Transformed dataset with imputed values.
        """
        X_imputed = X.copy()
        
        # Handle numerical features - use stored median for any missing values
        for feature in self.numerical_features:
            if X_imputed[feature].isnull().any():
                X_imputed[feature] = X_imputed[feature].fillna(self.median_values[feature])
            
        # Handle categorical features
        for feature in self.categorical_features:
            if X_imputed[feature].isnull().any():
                X_imputed[feature] = X_imputed[feature].fillna('Missing')
    
        return X_imputed
        
    def fit_transform(self, X, y=None):
        """        
        Args:
            X (pd.DataFrame): Input features to fit and transform.
            y (array-like, optional): Target values, not used in this transformer.
            
        Returns:
            pd.DataFrame: Transformed dataset with imputed values.
        """
        return self.fit(X).transform(X)


class CustomLabelEncoder(BaseEstimator, TransformerMixin):
   """
   A custom transformer for automatic label encoding of categorical variables.
   
   This transformer extends scikit-learn's LabelEncoder functionality by:
   - Automatically detecting and encoding all categorical columns
   - Handling unseen categories during transformation by replacing them with the mode
   - Providing compatibility with scikit-learn pipelines
   
   Attributes:
       encoders_ (dict): Dictionary mapping column names to their corresponding 
           LabelEncoder objects. Structure: {column_name: LabelEncoder}
       modes_ (dict): Dictionary storing the encoded mode value for each column,
           used for handling unseen categories. Structure: {column_name: encoded_mode}
       categorical_cols_ (list): List of detected categorical column names in the dataset
   """
   def __init__(self):
       """
       Initialize the custom encoder with empty storage for encoders, modes,
       and categorical column names.
       """
       # Store a LabelEncoder instance for each categorical column
       self.encoders_ = {}
       
       # Store the encoded mode value for each column to handle unseen categories
       self.modes_ = {}
       
       # Will store list of categorical columns detected during fit
       self.categorical_cols_ = None
       
   def fit(self, X, y=None):
       """
       Detect categorical columns and fit a LabelEncoder for each one.
       
       For each detected categorical column:
       1. Creates and fits a LabelEncoder
       2. Stores the encoded mode value for handling unseen categories
       
       Args:
           X (pd.DataFrame): Input dataset containing categorical columns to encode
           y (None): Ignored. Included for scikit-learn compatibility
               
       Returns:
           self: Returns the instance itself for method chaining
           
       Note:
           All values are converted to strings before encoding to handle mixed types
       """
       # Find all categorical and object columns in the dataset
       self.categorical_cols_ = list(X.select_dtypes(include=['object', 'category']).columns)
       
       # Early return if no categorical columns found
       if not self.categorical_cols_:
           print("No categorical columns found in the dataset.")
           return self
           
       # Process each categorical column
       for col in self.categorical_cols_:
           # Create and fit a new LabelEncoder for this column
           le = SklearnLabelEncoder()
           le.fit(X[col].astype(str))  # Convert to string for consistent handling
           self.encoders_[col] = le
           
           # Store encoded mode value for handling unseen categories later
           most_frequent_category = X[col].mode()[0]
           self.modes_[col] = le.transform([str(most_frequent_category)])[0]
           
       return self
   
   def transform(self, X):
       """
       Transform categorical columns using their fitted LabelEncoders.
       
       Applies the label encoding to each categorical column, handling unseen
       categories by replacing them with the encoded mode value from training.
       
       Args:
           X (pd.DataFrame): Dataset containing categorical columns to transform
               
       Returns:
           pd.DataFrame: Transformed dataset with encoded categorical columns
           
       Raises:
           ValueError: If any categorical columns found during fit are missing
               from the input DataFrame
       """
       # Create copy to avoid modifying original data
       X_transformed = X.copy()
       
       # Return unmodified data if no categorical columns were found during fit
       if not self.categorical_cols_:
           return X_transformed
       
       # Validate all expected columns are present
       missing_cols = [col for col in self.categorical_cols_ if col not in X.columns]
       if missing_cols:
           raise ValueError(f"Columns {missing_cols} not found in the input DataFrame")
       
       # Transform each categorical column
       for col in self.categorical_cols_:
           # Ensure consistent string type before transformation
           X_transformed[col] = X_transformed[col].astype(str)
           
           # Apply transformation with mode fallback for unseen categories
           X_transformed[col] = X_transformed[col].apply(
               lambda x: self.encoders_[col].transform([x])[0] 
               if x in self.encoders_[col].classes_ 
               else self.modes_[col]
           )
       
       return X_transformed
    

def handle_categorical_missing(df):
    """
    Fill missing values only in categorical columns that contain them.
    Preserves CatBoost's native missing value handling for numeric features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with missing values in categorical columns replaced with 'Missing'
    """
    # Create a copy to avoid modifying the original DataFrame
    df_processed = df.copy()
    
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Only process categorical columns that have missing values
    for col in categorical_cols:
        if df[col].isna().any():
            df_processed[col] = df[col].fillna('Missing')
            
    return df_processed 

def remove_constant_features(df):
    """
    Removes constant features (columns) from a DataFrame. A constant feature is one
    that has the same value across all rows.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame to process
        
    Returns:
    --------
    pandas.DataFrame
        A new DataFrame with constant features removed
        
    
    """
    # Calculate number of unique values for each column, excluding NaN
    unique_counts = df.nunique(dropna=True)
    
    # Find columns with only one unique value
    constant_columns = unique_counts[unique_counts <= 1].index.tolist()
    
    # Remove these columns from the DataFrame
    df_cleaned = df.drop(columns=constant_columns)
    
    return df_cleaned 

def kmeans_discretize(X, num_bins=5):
    """
    Discretize numeric features using k-means clustering.
    Only discretizes continuous numeric columns.
    
    Parameters:
    -----------
    X : pd.DataFrame or pd.Series
        Input data
    num_bins : int, default=5
        Number of bins
        
    Returns:
    --------
    pd.DataFrame or pd.Series
        Discretized data
    """
    def _kmeans_bin(series):

        kbd = KBinsDiscretizer(n_bins=num_bins, encode='ordinal', strategy='kmeans', subsample=None)
        discretized = kbd.fit_transform(series.values.reshape(-1, 1))
        return pd.Series(discretized.astype(int).ravel(), index=series.index)
            
    
    # Handle Series input
    if isinstance(X, pd.Series):
        return _kmeans_bin(X)
        
    # Handle DataFrame input    
    X_disc = X.copy()
    num_cols = X.select_dtypes(include=['number']).columns
    
    for col in num_cols:
        if len(X[col].unique()) > 10:
            X_disc[col] = _kmeans_bin(X[col])
            
    return X_disc

def encode_categorical_features(data):
    """
    Encode categorical features in a DataFrame or Series using LabelEncoder.
    Handles both DataFrame and Series inputs, preserving the input type in the output.
    Does not handle missing values - data should be preprocessed beforehand.
    
    Parameters:
    -----------
    data : pandas.DataFrame or pandas.Series
        Input data containing categorical features
        
    Returns:
    --------
    pandas.DataFrame or pandas.Series
        Data with categorical features encoded, maintaining the same type as input
    """
    # Handle Series input
    if isinstance(data, pd.Series):
        # Only encode if the Series is categorical or object type
        if data.dtype in ['object', 'category']:
            encoder = LabelEncoder()
            return pd.Series(encoder.fit_transform(data), index=data.index)
        return data
    
    # Handle DataFrame input
    df_encoded = data.copy()
    
    # Get categorical columns (object and category dtypes)
    categorical_columns = df_encoded.select_dtypes(include=['object', 'category']).columns
    
    # If no categorical columns, return original df
    if len(categorical_columns) == 0:
        return df_encoded
    
    # Apply LabelEncoder to each categorical column
    for column in categorical_columns:
        encoder = LabelEncoder()
        df_encoded[column] = encoder.fit_transform(df_encoded[column])
    
    return df_encoded


def target_encode_variables(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Apply target encoding to all categorical variables in a dataset.
    
    Parameters
    ----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
        
    Returns
    -------
    pd.DataFrame
        DataFrame with categorical variables encoded based on target means
    
    Example
    -------
    >>> X = pd.DataFrame({
    ...     'cat1': ['A', 'B', 'A', 'C'],
    ...     'cat2': ['X', 'X', 'Y', 'Z']
    ... })
    >>> y = pd.Series([1, 2, 1, 3])
    >>> X_encoded = target_encode_variables(X, y)
    """
    X_encoded = X.copy()
    
    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    # Calculate and apply mean encoding for each category in each column
    for col in categorical_cols:
        
        # Create a temporary DataFrame with the categorical column and target
        temp_df = pd.DataFrame({'category': X[col], 'target': y})
        
        # Compute mean target value for each category
        category_means = temp_df.groupby('category')['target'].mean()
        
        # Apply encoding to the column
        X_encoded[col] = X[col].map(category_means)
        
        # Fill any missing values with global mean
        if X_encoded[col].isna().any():
            X_encoded[col] = X_encoded[col].fillna(y.mean())

    return X_encoded

def encode_binary_target(y, balanced_status) -> np.ndarray:
    """
    Encode binary target variables ensuring that for imbalanced cases, 
    class 1 is always the minority class. Preserves the data type as pandas Series
    with the same index as the input.
    
    Args:
        y : Target variable to encode. Can be:
            - Already numeric binary (0,1)
            - Categorical/string binary (e.g., ['yes','no'], ['true','false'])
        balanced_status: Whether the target should be balanced or imbalanced.
            For imbalanced case, ensures class 1 is minority.
    
    Returns:
        pd.Series: Encoded target series with values 0 and 1
    """
    # Convert to pandas Series if not already
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    # Get unique values
    unique_values = y.unique()
    if len(unique_values) != 2:
        raise ValueError(f"Target must be binary. Found values: {unique_values}")
    
    # If already numeric binary (0,1)
    if set(unique_values) <= {0, 1}:
        if balanced_status == 'Imbalanced':
            
            # Count occurrences
            value_counts = y.value_counts()
            
            # If 1 is not the minority class, flip the encoding
            if value_counts.get(1, 0) > value_counts.get(0, 0):
                return pd.Series(1 - y, index=y.index)
        return y
    
    # For categorical/string values
    if balanced_status == 'Imbalanced':
        
        # Count occurrences
        value_counts = y.value_counts()
        
        # Create mapping ensuring minority class maps to 1
        encoding_map = {
            value_counts.index[-1]: 1,  # Minority class -> 1
            value_counts.index[0]: 0    # Majority class -> 0
        }
    else:
        
        # For balanced case, arbitrary mapping
        encoding_map = {val: i for i, val in enumerate(unique_values)}
    
    # Apply encoding while preserving index
    return pd.Series([encoding_map[val] for val in y], index=y.index)
