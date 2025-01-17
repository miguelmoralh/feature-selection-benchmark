import os
os.environ['OMP_NUM_THREADS'] = '1'
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import train_test_split

from benchmark_loop import IncrementalFeatureSelection

# Import filnames
from constants import dataset_filenames

def execute_final_pipline(methods: Optional[List[str]] = None, 
         synthetic_range: Optional[Tuple[int, int]] = None,
         real_range: Optional[Tuple[int, int]] = None):
    """
    Main function to run the incremental feature selection benchmark.
    
    Parameters:
    -----------
    methods : List[str], optional
        List of specific methods to run. If None, all applicable methods will be run.
    synthetic_range : Tuple[int, int], optional
        Range of synthetic datasets to process (start_idx, end_idx).
        If None, all synthetic datasets will be processed.
    real_range : Tuple[int, int], optional
        Range of real datasets to process (start_idx, end_idx).
        If None, all real datasets will be processed.
    """
    pipeline = IncrementalFeatureSelection(methods)

    # Process synthetic datasets
    if synthetic_range:
        start_idx, end_idx = synthetic_range
    else:
        start_idx, end_idx = 1, 51

    for i in range(start_idx, end_idx):
        pipeline.process_synthetic_dataset(i)

    # Process real-world datasets
    if real_range:
        start_idx, end_idx = real_range
    else:
        start_idx, end_idx = 0, len(dataset_filenames)

    for i in range(start_idx, end_idx):
        pipeline.process_real_dataset(i)

# Example usage:

# Run all methods on all datasets
execute_final_pipline()

# Run specific methods on some datasets
#execute_final_pipline(methods=['mutual_info'], synthetic_range=(1,1), real_range=(21,28))

#Run all methods on specific synthetic datasets
#execute_final_pipline(synthetic_range=(1, 20), real_range=(1,1))
