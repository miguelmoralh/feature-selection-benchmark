# Binary Classification parameters for fast catboost computation
fast_cb_bin_params = {
    # GPU Settings - Optimized for RTX 3050 Ti (4GB VRAM)
    'task_type': 'GPU',
    'devices': '0',
    'gpu_ram_part': 0.9,  # Leave some VRAM for system
    
    # Core Training
    'iterations': 200,
    'depth': 6,
    'learning_rate': 0.1,
    
    # GPU Optimizations for 3050 Ti
    'bootstrap_type': 'Bernoulli',
    'sampling_frequency': 'PerTree',
    'max_bin': 128,        # RTX 3050 Ti can handle this well
    
    # Early Stopping - Fixed to avoid conflicts
    'early_stopping_rounds': 30,
    'od_type': 'Iter',
    'use_best_model': True,  # Guardar y usar el mejor modelo encontrado
    'eval_fraction': 0.2,    # Usar 20% de datos para validación
    
    # Binary Classification Specific (CHANGE WHEN MULTICLASS)
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'auto_class_weights': 'Balanced',
    
    # Performance
    'subsample': 0.8,
    'l2_leaf_reg': 1.0,
    'random_strength': 0.5,
    
    # Feature Selection Specific
    'leaf_estimation_method': 'Newton',
    'leaf_estimation_iterations': 1,
    'nan_mode': 'Min',
    
    # Memory Settings for 4GB VRAM
    'used_ram_limit': '3gb',
    
    # Logging
    'verbose': False,
    'random_seed': 42,

    # Thread count - optimized for Ryzen 7 4800H
    'thread_count': 6
}

# Multiclass Classification parameters for fast catboost computations
fast_cb_multi_params = {
    # GPU Settings
    'task_type': 'GPU',
    'devices': '0',
    'gpu_ram_part': 0.9,
    
    # Core Training
    'iterations': 200,
    'depth': 6,
    'learning_rate': 0.1,
    
    # GPU Optimizations - Changed bootstrap settings
    'bootstrap_type': 'Bernoulli',  # Changed to Bernoulli which supports GPU multiclass
    'subsample': 0.8,  # Bernoulli bootstrap supports subsample
    'max_bin': 128,
    
    # Early Stopping
    'early_stopping_rounds': 30,
    'od_type': 'Iter',
    'use_best_model': True,  # Guardar y usar el mejor modelo encontrado
    'eval_fraction': 0.2,    # Usar 15% de datos para validación
    
    
    # Multiclass Specific
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'auto_class_weights': 'Balanced',
    
    # Performance
    'l2_leaf_reg': 1.0,
    'random_strength': 0.5,
    
    # Feature Selection Specific
    'leaf_estimation_method': 'Newton',
    'leaf_estimation_iterations': 1,
    'nan_mode': 'Min',
    
    # Memory Settings
    'used_ram_limit': '3gb',
    
    # Logging
    'verbose': False,
    'random_seed': 42,

    
    # Threading
    'thread_count': 6
}

# Regression parameters for fast catboost computation
fast_cb_regression_params = {
    # GPU Settings - Optimized for RTX 3050 Ti (4GB VRAM)
    'task_type': 'GPU',
    'devices': '0',
    'gpu_ram_part': 0.9,
    
    # Core Training
    'iterations': 200,
    'depth': 6,
    'learning_rate': 0.1,
    
    # GPU Optimizations for 3050 Ti
    'bootstrap_type': 'Bernoulli',
    'sampling_frequency': 'PerTree',
    'max_bin': 128,
    
    # Early Stopping - Fixed to avoid conflicts
    'early_stopping_rounds': 30,
    'od_type': 'Iter',
    'use_best_model': True,  # Guardar y usar el mejor modelo encontrado
    'eval_fraction': 0.2,    # Usar 20% de datos para validación
    
    # Regression Specific
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    
    # Performance
    'subsample': 0.8,
    'l2_leaf_reg': 1.0,
    'random_strength': 0.5,
    
    # Feature Selection Specific
    'leaf_estimation_method': 'Newton',
    'leaf_estimation_iterations': 1,
    'nan_mode': 'Min',
    
    # Memory Settings for 4GB VRAM
    'used_ram_limit': '3gb',
    
    # Logging
    'verbose': False,
    'random_seed': 42,
    
    # Thread count - optimized for Ryzen 7 4800H
    'thread_count': 6
}


rf_params = {
    'n_estimators': 100,        
    'max_depth': 6,             
    'verbose': False,
    'random_state': 42,
    'n_jobs': 4,               
    'max_features': 'sqrt'     
}