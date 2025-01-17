import numpy as np
from typing import List, Tuple, Callable

TransformTuple = Tuple[str, Callable[[float], float]]

# Define possible nonlinear transformations for features
nonlinear_transforms: List[TransformTuple] = [
    ('square', lambda x: x**2),                    # Quadratic relationship
    ('sin', lambda x: np.sin(2*np.pi*x)),          # Sinusoidal relationship
    ('exp', lambda x: np.exp(x)),                  # Exponential relationship
    ('log', lambda x: np.log(np.abs(x) + 1)),      # Logarithmic relationship
    ('sqrt', lambda x: np.sqrt(np.abs(x))),        # Square root relationship
    ('tanh', lambda x: np.tanh(x))                 # Hyperbolic tangent
]
