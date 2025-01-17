import pandas as pd
from utils.utils_datasets import load_file, categorize_dataset, generate_individual_metadata, generate_summary_metadata, print_summary
from constants import dataset_filenames, target_columns
import os
import json

# Store categorization info in a dictionary
dataset_info = {}

for file, target in zip(dataset_filenames, target_columns):
    dataset_path = os.path.join("data/real_world_datasets/datasets", file) 
    dataset_name = os.path.splitext(file)[0]  # Get dataset name without extension
    
    try:
            # Load and categorize dataset
            dataset = load_file(dataset_path)
            dataset_info[dataset_name] = categorize_dataset(
                dataset, 
                target, 
                dataset_name,
                imbalance_threshold_binary=0.80,
                imbalance_threshold_multi=0.60
            )
            print(f"Successfully processed dataset: {dataset_name}")  
    except Exception as e:
            print(f"Error processing dataset {dataset_name}: {str(e)}")
                  
    # Generate and save individual metadata
    individual_metadata = generate_individual_metadata(dataset_info)
    with open("data/real_world_datasets/metadata/datasets_metadata.json", "w", encoding="utf-8") as f:
        json.dump(individual_metadata, f, indent=2)
    print("\nIndividual metadata saved to: data/real_world_datasets/metadata/datasets_metadata.json")
    
    # Generate and save summary metadata
    summary_metadata = generate_summary_metadata(dataset_info)
    with open("data/real_world_datasets/metadata/metadata_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_metadata, f, indent=2)
    print("Summary metadata saved to: data/real_world_datasets/metadata/metadata_summary.json")
    
    # Print summary
    print_summary(summary_metadata)

