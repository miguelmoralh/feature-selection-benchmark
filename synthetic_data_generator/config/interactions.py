# config/interactions.py
import numpy as np
from typing import List, Tuple, Callable

InteractionTuple = Tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]

# Define possible interaction types between features (convert to floats to deal with future boolean dummies)
interaction_types: List[InteractionTuple] = [
    ('multiply', lambda x, y: x.astype(float) * y.astype(float)),
    ('divide', lambda x, y: x.astype(float) / (np.abs(y.astype(float)) + 0.1))
]