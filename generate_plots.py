import json
import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.lines import Line2D
from utils.utils_results_and_plots import create_real_world_plot, create_synthetic_plot

# Consistent method ordering and naming
METHOD_ORDER = {
    # Filter methods (1-7)
    'chi_squared': 1,
    'information_value': 2,
    'correlation': 3,
    'mutual_info': 4,
    'mrmr': 5,
    'fcbf': 6,
    'relief': 7,
    
    # Embedded methods (8-10)
    'rf_fimportances': 8,
    'cb_fimportances': 9,
    'permutation_importance': 10,
    
    # Wrapper methods (11-14)
    'seq_backward': 11,
    'seq_forward': 12,
    'seq_forward_floating': 13,
    'seq_backward_floating': 14,
    
    # Advanced methods (15-16)
    'shap': 15,
    'boruta': 16,
    
    # Hybrid methods (17-20)
    'hybrid_nmi_sfs': 17,
    'hybrid_fcbf_sfs': 18,
    'hybrid_rfe': 19,
    'hybrid_shap_sfs': 20
}

METHOD_NAMES = {
    # Filter methods
    'chi_squared': 'Chi²',
    'information_value': 'Info Value',
    'correlation': 'Correlation',
    'mutual_info': 'Mutual Info',
    'mrmr': 'mRMR',
    'fcbf': 'FCBF',
    'relief': 'Relief',
    
    # Embedded methods
    'rf_fimportances': 'RF-FI',
    'cb_fimportances': 'CB-FI',
    'permutation_importance': 'Perm-FI',
    
    # Wrapper methods
    'seq_backward': 'SBS',
    'seq_forward': 'SFS',
    'seq_forward_floating': 'SFFS',
    'seq_backward_floating': 'SBFS',
    
    # Advanced methods
    'shap': 'SHAP',
    'boruta': 'Boruta',
    
    # Hybrid methods
    'hybrid_nmi_sfs': 'MI - SFS',
    'hybrid_fcbf_sfs': 'FCBF - SFS',
    'hybrid_rfe': 'RFE',
    'hybrid_shap_sfs': 'SHAP - SFS',

}

# Keep existing color schemes and family mappings
FAMILY_COLORS = {
    'filter': '#ff7f0e',    # Orange
    'embedded': '#2ca02c',  # Green
    'wrapper': '#d62728',   # Red
    'advanced': '#1f77b4',  # Blue
    'hybrid': '#9467bd'     # Purple
}

METHOD_FAMILIES = {
    # Filter methods
    'chi_squared': 'filter',
    'information_value': 'filter',
    'correlation': 'filter',
    'mutual_info': 'filter',
    'mrmr': 'filter',
    'fcbf': 'filter',
    'relief': 'filter',
    
    # Embedded methods
    'rf_fimportances': 'embedded',
    'cb_fimportances': 'embedded',
    'permutation_importance': 'embedded',
    
    # Wrapper methods
    'seq_backward': 'wrapper',
    'seq_forward': 'wrapper',
    'seq_forward_floating': 'wrapper',
    'seq_backward_floating': 'wrapper',
    
    # Advanced methods
    'shap': 'advanced',
    'boruta': 'advanced',
    
    # Hybrid methods
    'hybrid_nmi_sfs': 'hybrid',
    'hybrid_fcbf_sfs': 'hybrid',
    'hybrid_rfe': 'hybrid',
    'hybrid_shap_sfs': 'hybrid'
 
}

FAMILY_ORDER = ['filter', 'embedded', 'wrapper', 'advanced', 'hybrid']


# Create output directories
os.makedirs('logs/plots', exist_ok=True)
os.makedirs('logs/plots/dimensionality', exist_ok=True)
os.makedirs('logs/plots/tasks', exist_ok=True)

# Load results
with open('logs/results/avg_metrics_by_method.json', 'r') as f:
    overall_results = json.load(f)
    
with open('logs/results/datasets_categories/avg_dimensionality_results.json', 'r') as f:
    dimensionality_results = json.load(f)
    
with open('logs/results/datasets_categories/avg_task_results.json', 'r') as f:
    task_results = json.load(f)

# Create overall plots
create_real_world_plot(overall_results['real_world'], 
                        METHOD_ORDER,
                        FAMILY_ORDER, 
                        METHOD_FAMILIES, 
                        FAMILY_COLORS, 
                        METHOD_NAMES)
create_synthetic_plot(overall_results['synthetic'],
                        METHOD_ORDER,
                        FAMILY_ORDER, 
                        METHOD_FAMILIES, 
                        FAMILY_COLORS, 
                        METHOD_NAMES)

# Create dimensionality plots
dimensionality_categories = ['High-Dimensional', 'Mid-Dimensional', 'Low-Dimensional']
for category in dimensionality_categories:
    suffix = f" ({category})"
    file_suffix = f"_{category.lower().replace('-', '_')}"
    
    if category in dimensionality_results['real_world']:
        create_real_world_plot(
            dimensionality_results['real_world'][category],
            METHOD_ORDER,
            FAMILY_ORDER, 
            METHOD_FAMILIES, 
            FAMILY_COLORS, 
            METHOD_NAMES,
            title_suffix=suffix,
            output_suffix=file_suffix
        )
    
    if category in dimensionality_results['synthetic']:
        create_synthetic_plot(
            dimensionality_results['synthetic'][category],
            METHOD_ORDER,
            FAMILY_ORDER, 
            METHOD_FAMILIES, 
            FAMILY_COLORS, 
            METHOD_NAMES,
            title_suffix=suffix,
            output_suffix=file_suffix
        )

# Create task plots
task_categories = ['regression', 'binary_classification', 'multiclass_classification']
task_display_names = {
    'regression': 'Regression',
    'binary_classification': 'Binary Classification',
    'multiclass_classification': 'Multiclass Classification'
}

for category in task_categories:
    suffix = f" ({task_display_names[category]})"
    file_suffix = f"_{category.lower()}"
    
    if category in task_results['real_world']:
        create_real_world_plot(
            task_results['real_world'][category],
            METHOD_ORDER,
            FAMILY_ORDER, 
            METHOD_FAMILIES, 
            FAMILY_COLORS, 
            METHOD_NAMES,
            title_suffix=suffix,
            output_suffix=file_suffix
        )
    
    if category in task_results['synthetic']:
        create_synthetic_plot(
            task_results['synthetic'][category],
            METHOD_ORDER,
            FAMILY_ORDER, 
            METHOD_FAMILIES, 
            FAMILY_COLORS, 
            METHOD_NAMES,
            title_suffix=suffix,
            output_suffix=file_suffix
        )
