import json
import os
import numpy as np
from collections import defaultdict
from utils.utils_results_and_plots import load_metadata, get_rankings, calculate_mean_metrics_overall, calculate_mean_metrics_by_category

# Load metadata
real_world_metadata, synthetic_metadata = load_metadata()

# Load benchmark results
with open('logs/benchmark/real_world_benchmark_results.json', 'r') as f:
    real_world_results = json.load(f)
with open('logs/benchmark/synthetic_benchmark_results.json', 'r') as f:
    synthetic_results = json.load(f)

# Calculate original overall rankings and metrics
rankings = get_rankings(real_world_results, synthetic_results, real_world_metadata)
mean_metrics_overall = calculate_mean_metrics_overall(rankings, real_world_results, synthetic_results)

# Calculate metrics by dimensionality
dimensionality_metrics = {
    'real_world': calculate_mean_metrics_by_category(
        real_world_results, real_world_metadata, 'real_world', 'dimensionality'
    ),
    'synthetic': calculate_mean_metrics_by_category(
        synthetic_results, synthetic_metadata, 'synthetic', 'dimensionality'
    )
}

# Calculate metrics by task
task_metrics = {
    'real_world': calculate_mean_metrics_by_category(
        real_world_results, real_world_metadata, 'real_world', 'task'
    ),
    'synthetic': calculate_mean_metrics_by_category(
        synthetic_results, synthetic_metadata, 'synthetic', 'task'
    )
}

# Save all results
os.makedirs('logs/results', exist_ok=True)
os.makedirs('logs/results/datasets_categories', exist_ok=True)

# Save rankings
with open('logs/results/rankings_by_dataset.json', 'w') as f:
    json.dump(rankings, f, indent=4)
    
# Save overall metrics
with open('logs/results/avg_metrics_by_method.json', 'w') as f:
    json.dump(mean_metrics_overall, f, indent=4)
    
# Save categorized metrics
with open('logs/results/datasets_categories/avg_dimensionality_results.json', 'w') as f:
    json.dump(dimensionality_metrics, f, indent=4)
    
with open('logs/results/datasets_categories/avg_task_results.json', 'w') as f:
    json.dump(task_metrics, f, indent=4)
