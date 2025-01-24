import json
import os
import numpy as np
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

def load_metadata():
    """
    Loads metadata for both real-world and synthetic datasets from JSON files.
    
    Returns:
        tuple: (real_world_metadata, synthetic_datasets)
        - real_world_metadata: Dictionary containing metadata for real-world datasets
        - synthetic_datasets: Dictionary containing metadata for synthetic datasets
    """
    # Load real-world metadata
    with open('data/real_world_datasets/metadata/datasets_metadata.json', 'r') as f:
        real_world_metadata = json.load(f)
    
    # Load synthetic metadata    
    with open('data/synthetic_datasets/metadata/datasets_metadata.json', 'r') as f:
        synthetic_metadata = json.load(f)
        synthetic_datasets = {f"dataset_{i+1}": dataset 
                            for i, dataset in enumerate(synthetic_metadata['datasets'])}
    
    return real_world_metadata, synthetic_datasets

def get_dataset_category(dataset_info, category_type, dataset_type):
    """
    Determines the category (dimensionality or task type) for a given dataset.
    
    Args:
        dataset_info (dict): Dataset metadata
        category_type (str): Type of categorization ('task' or 'dimensionality')
        dataset_type (str): Type of dataset ('synthetic' or 'real_world')
    
    Returns:
        str: Category label for the dataset
    """
    if category_type == 'task':
        if dataset_type == 'synthetic':
            return dataset_info['task_type']
        return dataset_info['task']
    elif category_type == 'dimensionality':
        if dataset_type == 'synthetic':
            n_features = len(dataset_info.get('numerical_features', []))
            if n_features > 100:
                return 'High-Dimensional'
            elif n_features > 10:
                return 'Mid-Dimensional'
            else:
                return 'Low-Dimensional'
        return dataset_info['dimensionality']
    return None

def get_percentage_score_errors(real_world_results, real_world_metadata):
    """
    Calculate percentage score error for each method in each dataset before averaging.
    
    Args:
        real_world_results (dict): Results from real-world datasets
        real_world_metadata (dict): Metadata for real-world datasets
    
    Returns:
        dict: Percentage score errors for each method in each dataset
    """
    errors_by_dataset = {}
    
    # Process each dataset
    for dataset_name, dataset_results in real_world_results.items():
        task = real_world_metadata[dataset_name]['task']
        reverse = task != 'regression'  # True for classification, False for regression
        
        # Find best CatBoost score for this dataset
        best_score = None
        for method_results in dataset_results.values():
            if 'models_performance' not in method_results:
                continue
            score = method_results['models_performance']['catboost']
            if best_score is None:
                best_score = score
            elif reverse and score > best_score:  # classification
                best_score = score
            elif not reverse and score < best_score:  # regression
                best_score = score
        
        # Calculate percentage error for each method
        errors_by_dataset[dataset_name] = {}
        for method, method_results in dataset_results.items():
            if 'models_performance' not in method_results:
                continue
            
            score = method_results['models_performance']['catboost']
            # Calculate distance and normalize to percentage
            distance = abs(score - best_score)
            error_percentage = (distance / abs(best_score)) * 100 if best_score != 0 else 0
            errors_by_dataset[dataset_name][method] = error_percentage
    
    return errors_by_dataset

def get_selection_technique_rankings(dataset_name, dataset_results, dataset_type, task=None):
    """
    Calculates rankings for feature selection techniques based on their performance.
    Only used for synthetic datasets.
    
    Args:
        dataset_name (str): Name of the dataset
        dataset_results (dict): Results for all methods on this dataset
        dataset_type (str): Type of dataset ('synthetic' or 'real_world')
        task (str, optional): Type of task (regression/classification)
    
    Returns:
        dict: Rankings for each feature selection method
    """
    method_scores = {}
    
    # Extract relevant scores based on dataset type
    for method, method_results in dataset_results.items():
        if dataset_type == 'synthetic':
            if 'weighted_accuracy' in method_results:
                method_scores[method] = method_results['weighted_accuracy']
    
    # Sort and assign rankings
    sorted_methods = sorted(method_scores.items(), key=lambda x: x[1], reverse=True)
    rankings = {}
    current_rank = 1
    prev_score = None
    skip_ranks = 0
    
    for method, score in sorted_methods:
        if prev_score is not None and score != prev_score:
            current_rank += skip_ranks + 1
            skip_ranks = 0
        else:
            if prev_score is not None:
                skip_ranks += 1
                
        rankings[method] = current_rank
        prev_score = score
    
    return rankings

def get_rankings(synthetic_results):
    """
    Calculates and returns rankings of feature selection methods for synthetic datasets.

    Args:
        synthetic_results (dict): Results dictionary for synthetic datasets.
    Returns:
        dict: A nested dictionary where rankings are integers starting from 1 (best performing).
    """
    rankings = {
        'synthetic': {}
    }

    # Synthetic rankings
    for dataset_name, dataset_results in synthetic_results.items():
        rankings['synthetic'][dataset_name] = get_selection_technique_rankings(
            dataset_name, dataset_results, 'synthetic'
        )
    
    return rankings

def calculate_mean_metrics_overall(rankings, real_world_results, synthetic_results, real_world_metadata):
    """
    Calculates average performance metrics across all datasets.
    
    Args:
        rankings (dict): Weighted Average synthtic rankings for each method across datasets
        real_world_results (dict): Results from real-world datasets
        synthetic_results (dict): Results from synthetic datasets
        real_world_metadata (dict): Metadata for real-world datasets
    
    Returns:
        dict: Average metrics for each method, separated by dataset type
    """
    mean_metrics = {
        'real_world': {},
        'synthetic': {}
    }

    # Real-world means
    method_metrics = defaultdict(lambda: {'total_time': 0, 'total_features': 0, 'total_error': 0, 'count': 0})
    
    # Process each dataset
    for dataset_name, dataset_results in real_world_results.items():
        task = real_world_metadata[dataset_name]['task']
        reverse = task != 'regression'  # True for classification, False for regression
        
        # Find best CatBoost score for this dataset
        best_score = None
        for method_results in dataset_results.values():
            if 'models_performance' not in method_results:
                continue
            score = method_results['models_performance']['catboost']
            if best_score is None:
                best_score = score
            elif reverse and score > best_score:  # classification
                best_score = score
            elif not reverse and score < best_score:  # regression
                best_score = score
        
        # Calculate percentage error for each method
        for method, method_results in dataset_results.items():
            if 'models_performance' not in method_results:
                continue
            
            score = method_results['models_performance']['catboost']
            # Calculate distance and normalize to percentage
            distance = abs(score - best_score)
            error_percentage = (distance / abs(best_score)) * 100 if best_score != 0 else 0
            
            method_metrics[method]['total_error'] += error_percentage
            method_metrics[method]['total_time'] += method_results['execution_time']
            method_metrics[method]['total_features'] += method_results['n_selected_features']
            method_metrics[method]['count'] += 1

    # Calculate averages for real-world
    for method, metrics in method_metrics.items():
        count = metrics['count']
        mean_metrics['real_world'][method] = {
            'mean_execution_time_minutes': (metrics['total_time'] / count) / 60,
            'mean_selected_features': metrics['total_features'] / count,
            'average_percentage_score_error': metrics['total_error'] / count,
            'datasets_evaluated': count
        }

    # Synthetic means
    method_metrics = defaultdict(lambda: {'total_time': 0, 'total_rank': 0, 'count': 0})
    
    for dataset_name, dataset_results in synthetic_results.items():
        for method, method_results in dataset_results.items():
            if 'weighted_accuracy' not in method_results:
                continue
                
            method_metrics[method]['total_time'] += method_results['execution_time']
            method_metrics[method]['total_rank'] += rankings['synthetic'][dataset_name][method]
            method_metrics[method]['count'] += 1

    for method, metrics in method_metrics.items():
        count = metrics['count']
        mean_metrics['synthetic'][method] = {
            'mean_execution_time_minutes': (metrics['total_time'] / count) / 60,
            'mean_weighted_accuracy_ranking': metrics['total_rank'] / count,
            'datasets_evaluated': count
        }
    
    return mean_metrics

def calculate_mean_metrics_by_category(results, metadata, dataset_type, category_type):
    """
    Calculates average performance metrics for feature selection methods grouped by category.
    Categories can be either dimensionality-based or task-based.

    Args:
        results (dict): Dictionary containing benchmark results for all methods across datasets.
        metadata (dict): Dataset metadata containing categorization information.
        dataset_type (str): Type of dataset, either 'real_world' or 'synthetic'.
        category_type (str): Type of categorization, either 'dimensionality' or 'task'.

    Returns:
        dict: A nested dictionary containing average metrics for each method within each category.
        For dimensionality categories: {'High-Dimensional', 'Mid-Dimensional', 'Low-Dimensional'}
        For task categories: {'regression', 'binary_classification', 'multiclass_classification'}+
    """
    categories = {
        'dimensionality': ['High-Dimensional', 'Mid-Dimensional', 'Low-Dimensional'],
        'task': ['regression', 'binary_classification', 'multiclass_classification']
    }
    
    mean_metrics = {}
    
    # Initialize structure for each category
    for category in categories[category_type]:
        mean_metrics[category] = {}
    
            # Process each dataset
    for dataset_name, dataset_results in results.items():
        if dataset_name not in metadata:
            continue
            
        dataset_info = metadata[dataset_name]
        category = get_dataset_category(dataset_info, category_type, dataset_type)
        
        # Get rankings only for synthetic datasets
        rankings = None
        if dataset_type == 'synthetic':
            rankings = get_selection_technique_rankings(
                dataset_name,
                dataset_results,
                dataset_type,
                task=get_dataset_category(dataset_info, 'task', dataset_type)
            )
        
        if dataset_type == 'real_world':
            # For real-world datasets, find best score first
            task = dataset_info['task']
            reverse = task != 'regression'
            best_score = None
            
            for method_results in dataset_results.values():
                if 'models_performance' not in method_results:
                    continue
                score = method_results['models_performance']['catboost']
                if best_score is None:
                    best_score = score
                elif reverse and score > best_score:
                    best_score = score
                elif not reverse and score < best_score:
                    best_score = score
        
        # Process each method's results
        for method, method_results in dataset_results.items():
            if method not in mean_metrics[category]:
                mean_metrics[category][method] = defaultdict(lambda: {'total': 0, 'count': 0})
            
            metrics = mean_metrics[category][method]
            
            # Skip if missing required metrics
            if dataset_type == 'real_world':
                if 'models_performance' not in method_results:
                    continue
                
                # Calculate error percentage for real-world
                score = method_results['models_performance']['catboost']
                distance = abs(score - best_score)
                error_percentage = (distance / abs(best_score)) * 100 if best_score != 0 else 0
                metrics['error']['total'] += error_percentage
                metrics['error']['count'] += 1
                
                metrics['selected_features']['total'] += method_results['n_selected_features']
                metrics['selected_features']['count'] += 1
            else:  # synthetic
                if 'weighted_accuracy' not in method_results:
                    continue
                if rankings:  # Only if we have rankings (synthetic case)
                    metrics['weighted_accuracy_ranking']['total'] += rankings[method]
                    metrics['weighted_accuracy_ranking']['count'] += 1
            
            metrics['execution_time']['total'] += method_results['execution_time']
            metrics['execution_time']['count'] += 1
    
    # Calculate final means
    final_metrics = {}
    for category in categories[category_type]:
        final_metrics[category] = {}
        for method, metrics in mean_metrics[category].items():
            method_means = {}
            
            # Calculate means for each metric
            for metric_name, values in metrics.items():
                if values['count'] > 0:
                    if metric_name == 'error':
                        method_means['average_percentage_score_error'] = values['total'] / values['count']
                    else:
                        method_means[f'mean_{metric_name}'] = values['total'] / values['count']
            
            if method_means:
                method_means['mean_execution_time_minutes'] = (
                    method_means.pop('mean_execution_time') / 60
                )
                method_means['datasets_evaluated'] = metrics['execution_time']['count']
                final_metrics[category][method] = method_means
    
    return final_metrics

def create_real_world_plot(results, METHOD_ORDER, FAMILY_ORDER, METHOD_FAMILIES, FAMILY_COLORS, METHOD_NAMES, title_suffix="", output_suffix=""):
    """
    Creates a scatter plot visualizing the performance of feature selection methods
    on real-world datasets.
    
    Args:
        results (dict): Dictionary containing performance metrics for each method
        METHOD_ORDER (dict): Mapping of methods to their respective order numbers
        FAMILY_ORDER (list): List defining the order of method families for display
        METHOD_FAMILIES (dict): Mapping of methods to their respective families
        FAMILY_COLORS (dict): Color mapping for each method family
        METHOD_NAMES (dict): Display names for each method
        title_suffix (str): Additional text to append to plot title
        output_suffix (str): Suffix for output filename
    
    The plot shows:
    - X-axis: Average model scores ranking (lower is better)
    - Y-axis: Average number of selected features
    - Point size: Indicates execution time (larger = longer runtime)
    - Color: Indicates method family
    """
    methods_data = []
    
    # Process methods in consistent order
    for method_name, order_num in METHOD_ORDER.items():
        if method_name in results:
            metrics = results[method_name]
            methods_data.append({
                'method': method_name,
                'family': METHOD_FAMILIES[method_name],
                'error': metrics['average_percentage_score_error'],
                'features': metrics.get('mean_selected_features', 0),
                'time': metrics['mean_execution_time_minutes'],
                'index': order_num
            })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot points
    for data in sorted(methods_data, key=lambda x: x['index']):
        color = FAMILY_COLORS[data['family']]
        
        # Size calculation
        log_time = np.log1p(data['time'])
        min_log = np.log1p(min(d['time'] for d in methods_data))
        max_log = np.log1p(max(d['time'] for d in methods_data))
        
        normalized_size = (log_time - min_log) / (max_log - min_log) if max_log != min_log else 0
        size = 300 + normalized_size * 1700
        
        # Plot point
        ax.scatter(data['error'], data['features'], s=size, c=color, alpha=0.6)
        ax.text(data['error'], data['features'], str(data['index']),
                color='white', ha='center', va='center', fontweight='bold', fontsize=8)
    
    # Create legend
    legend_elements = []
    legend_labels = []
    
    # Add execution time explanation
    legend_elements.append(Line2D([], [], color='none'))
    legend_labels.append('Point size indicates \nexecution time: bigger \npoints = longer runtime')
    legend_elements.append(Line2D([], [], color='none'))
    legend_labels.append(' ')
    
    # Add methods by family
    for family in FAMILY_ORDER:
        legend_elements.append(Line2D([], [], color='none'))
        legend_labels.append(family.upper())
        
        family_methods = [m for m in methods_data if m['family'] == family]
        for data in sorted(family_methods, key=lambda x: x['index']):
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=FAMILY_COLORS[family],
                                       markersize=8, alpha=0.6))
            legend_labels.append(f"{data['index']}. {METHOD_NAMES[data['method']]}")
    
    # Customize plot
    ax.legend(legend_elements, legend_labels,
             loc='center left', bbox_to_anchor=(1, 0.5),
             fontsize=10, frameon=True,
             handletextpad=1,
             borderpad=1,
             labelspacing=0.5)
    
    ax.set_xlabel('Average Percentage Score Error\n(lower is better)', fontsize=15)
    ax.set_ylabel('Average number of selected features', fontsize=15)
    ax.set_title(f'Average results in real-world datasets{title_suffix}', pad=20, fontsize=20)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Set x-axis limit based on plot type
    if 'dimensional' in output_suffix.lower():
        ax.set_xlim(0, 70)
    elif any(task in output_suffix.lower() for task in ['regression', 'classification']):
        ax.set_xlim(0, 70)
    else:
        ax.set_xlim(0, 40)
        
    # Set y-axis limit based on dimensionality
    if 'high_dimensional' in output_suffix.lower():
        ax.set_ylim(0, 200)  # Higher limit for high dimensional
    elif 'low_dimensional' in output_suffix.lower():
        ax.set_ylim(0, 20)  # Lower limit for low dimensional
    elif any(task in output_suffix.lower() for task in ['regression', 'classification']):
        ax.set_ylim(0, 100)
    else:
        ax.set_ylim(0, 60)  # Regular limit for others
            
    # Save plot
    plt.tight_layout()
    save_dir = 'logs/plots'
    if 'dimensional' in output_suffix.lower():
        save_dir = 'logs/plots/dimensionality'
    elif any(task in output_suffix.lower() for task in ['regression', 'classification']):
        save_dir = 'logs/plots/tasks'
        
    plt.savefig(f'{save_dir}/real_world_performance{output_suffix}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def create_synthetic_plot(results, METHOD_ORDER, FAMILY_ORDER, METHOD_FAMILIES, FAMILY_COLORS, METHOD_NAMES, title_suffix="", output_suffix=""):
    """
    Creates a scatter plot visualizing the performance of feature selection methods
    on synthetic datasets.
    
    Args:
        results (dict): Dictionary containing performance metrics for each method
        FAMILY_ORDER (list): List defining the order of method families for display
        METHOD_ORDER (dict): Mapping of methods to their respective order numbers
        METHOD_FAMILIES (dict): Mapping of methods to their respective families
        FAMILY_COLORS (dict): Color mapping for each method family
        METHOD_NAMES (dict): Display names for each method
        title_suffix (str): Additional text to append to plot title
        output_suffix (str): Suffix for output filename
    
    The plot shows:
    - X-axis: Average weighted accuracy ranking (lower is better)
    - Y-axis: Average execution time (minutes)
    - Color: Indicates method family
    """

    methods_data = []
    
    # Process all methods in consistent order
    for method_name, order_num in METHOD_ORDER.items():
        if method_name in results:
            metrics = results[method_name]
            if 'mean_weighted_accuracy_ranking' in metrics:
                methods_data.append({
                    'method': method_name,
                    'family': METHOD_FAMILIES[method_name],
                    'ranking': metrics['mean_weighted_accuracy_ranking'],
                    'time': metrics['mean_execution_time_minutes'],
                    'index': order_num
                })
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot points with consistent numbering
    for data in sorted(methods_data, key=lambda x: x['index']):
        color = FAMILY_COLORS[data['family']]
        
        # Plot point
        ax.scatter(data['ranking'], data['time'], s=1000, c=color, alpha=0.6)
        ax.text(data['ranking'], data['time'], str(data['index']),
                color='white', ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Create legend with consistent ordering
    legend_elements = []
    legend_labels = []
    
    # Add methods by family in consistent order
    for family in FAMILY_ORDER:
        legend_elements.append(Line2D([], [], color='none'))
        legend_labels.append(family.upper())
        
        family_methods = [m for m in methods_data if m['family'] == family]
        for data in sorted(family_methods, key=lambda x: x['index']):
            legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                       markerfacecolor=FAMILY_COLORS[family],
                                       markersize=8, alpha=0.6))
            legend_labels.append(f"{data['index']}. {METHOD_NAMES[data['method']]}")
    
    # Customize plot
    ax.legend(legend_elements, legend_labels,
             loc='center left', bbox_to_anchor=(1, 0.5),
             fontsize=10, frameon=True,
             handletextpad=1,
             borderpad=1,
             labelspacing=0.5)
    
    ax.set_xlabel('Average Weighted Accuracy Ranking\n(lower is better)', fontsize=15)  # Increase fontsize
    ax.set_ylabel('Average execution time (minutes)', fontsize=15)  # Increase fontsize
    ax.set_title(f'Average results in synthetic datasets{title_suffix}', pad=20, fontsize=20)  # Increase fontsize
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim(0, 20)
    ax.set_yscale('log')
    
    # Save plot
    plt.tight_layout()
    save_dir = 'logs/plots'
    if 'dimensional' in output_suffix.lower():
        save_dir = 'logs/plots/dimensionality'
    elif any(task in output_suffix.lower() for task in ['regression', 'classification']):
        save_dir = 'logs/plots/tasks'
        
    plt.savefig(f'{save_dir}/synthetic_performance{output_suffix}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()



