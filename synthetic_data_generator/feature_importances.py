import numpy as np
import pandas as pd
import re
from typing import Dict, List, Tuple

class FeatureImportanceAnalyzer:
    def __init__(self):
        # Define transformation functions matching those in your generator
        self.transformations = {
            'square': lambda x: x**2,
            'sin': lambda x: np.sin(2*np.pi*x),
            'exp': lambda x: np.exp(x),
            'log': lambda x: np.log(np.abs(x) + 1),
            'sqrt': lambda x: np.sqrt(np.abs(x)),
            'tanh': lambda x: np.tanh(x),
            '': lambda x: x  # Identity function for no transformation
        }
        
        # Define interaction functions
        self.interactions = {
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / (np.abs(y) + 0.1)  # Adding small constant to prevent division by zero
        }

    def _extract_feature_and_transform(self, term: str) -> Tuple[str, str]:
        """
        Extract feature name and transformation from a term.
        
        Args:
            term: String containing feature possibly with transformation
            
        Returns:
            Tuple containing:
            - feature name
            - transformation name (empty string if none)
        """
        transform = ''
        feature = term.strip('()')
        
        # Check for each possible transformation
        for trans in self.transformations.keys():
            if trans and trans in term:
                transform = trans
                feature = term.replace(f"{trans}(", "").rstrip(")")
                break
                
        return feature, transform

    def _parse_term(self, term: str) -> Tuple[float, str, str, str, str, str]:
        """
        Parse a single term from the target formula.
        
        Returns:
            Tuple containing:
            - coefficient
            - first feature name
            - first feature transformation
            - interaction operator (if any)
            - second feature name (if interaction present)
            - second feature transformation (if interaction present)
        """
        # Extract coefficient
        coef_match = re.match(r'^(-?\d+\.?\d*)\*', term)
        coefficient = float(coef_match.group(1)) if coef_match else 1.0
        
        # Remove coefficient from term
        term_without_coef = term[term.find('*')+1:] if coef_match else term
        
        # Check for interactions
        if '*' in term_without_coef or '/' in term_without_coef:
            # Find the operator
            operator = '*' if '*' in term_without_coef else '/'
            
            # Split into the two parts
            if '(' in term_without_coef and ')' in term_without_coef:
                # Handle case where entire interaction is wrapped in parentheses
                interaction_terms = term_without_coef.strip('()').split(f' {operator} ')
            else:
                interaction_terms = term_without_coef.split(f' {operator} ')
            
            # Extract feature and transformation for each part
            feat1, trans1 = self._extract_feature_and_transform(interaction_terms[0])
            feat2, trans2 = self._extract_feature_and_transform(interaction_terms[1])
            
            return coefficient, feat1, trans1, operator, feat2, trans2
            
        else:
            # Handle single feature terms
            feature, transform = self._extract_feature_and_transform(term_without_coef)
            return coefficient, feature, transform, '', '', ''

    def _calculate_term_impact(
            self, 
            X: pd.DataFrame, 
            coefficient: float,
            feature1: str,
            transform1: str,
            operator: str,
            feature2: str = None,
            transform2: str = None
        ) -> np.ndarray:
        """Calculate the impact of a single term in the formula."""
        
        # Get and transform first feature values
        values1 = X[feature1].values
        trans_func1 = self.transformations[transform1]
        transformed1 = trans_func1(values1)
        
        if operator and feature2:
            # Handle interaction terms
            values2 = X[feature2].values
            # Apply transformation to second feature if present
            trans_func2 = self.transformations[transform2]
            transformed2 = trans_func2(values2)
            
            interaction_func = self.interactions[operator]
            result = coefficient * interaction_func(transformed1, transformed2)
        else:
            # Handle single feature terms
            result = coefficient * transformed1
            
        return result

    def analyze_importance(
            self, 
            X: pd.DataFrame, 
            target_formula: str
        ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Analyze feature importance based on target formula and actual data.
        
        Args:
            X: DataFrame containing features
            target_formula: String containing the target formula
            
        Returns:
            Tuple containing:
            - DataFrame with term-wise contributions
            - Dictionary with feature-wise importance scores
        """
        # Split formula into individual terms
        terms = [t.strip() for t in target_formula.split(' + ')]
        
        # Calculate impact of each term
        term_impacts = {}
        feature_contributions = {}
        
        for term in terms:
            coef, feat1, trans1, operator, feat2, trans2 = self._parse_term(term)
            
            # Calculate term's impact
            impact = self._calculate_term_impact(
                X, coef, feat1, trans1, operator, feat2, trans2
            )
            
            # Store term's impact
            term_impacts[term] = impact
            
            # Track contribution per feature
            if feat1 not in feature_contributions:
                feature_contributions[feat1] = []
                
            abs_impact = np.abs(impact)
            
            if operator and feat2:  # If there's an interaction
                # Divide the impact between the two features
                if feat2 not in feature_contributions:
                    feature_contributions[feat2] = []
                    
                # Add half of the impact to each feature
                feature_contributions[feat1].append(abs_impact / 2)
                feature_contributions[feat2].append(abs_impact / 2)
            else:
                # Single feature term - full impact
                feature_contributions[feat1].append(abs_impact)
        
        # Create DataFrame with term-wise statistics
        term_stats = pd.DataFrame({
            'term': list(term_impacts.keys()),
            'mean_absolute_effect': [np.mean(np.abs(imp)) for imp in term_impacts.values()],
            'std_effect': [np.std(imp) for imp in term_impacts.values()],
            'min_effect': [np.min(imp) for imp in term_impacts.values()],
            'max_effect': [np.max(imp) for imp in term_impacts.values()]
        })
        
        # Calculate total absolute effect across all terms
        total_effect = sum(term_stats['mean_absolute_effect'])
        
        # Calculate feature-wise importance
        feature_importance = {}
        for feature, impacts in feature_contributions.items():
            # Sum all impacts since division has been done earlier
            feature_total = np.mean(impacts)
            feature_importance[feature] = (feature_total / total_effect) * 100
            
        return term_stats, feature_importance
    
