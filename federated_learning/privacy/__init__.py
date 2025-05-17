"""
Privacy mechanisms for federated learning.
This module provides implementations of different privacy mechanisms:
- Differential Privacy (DP)
- Homomorphic Encryption (Paillier)
"""

from federated_learning.privacy.differential_privacy import apply_differential_privacy
from federated_learning.privacy.homomorphic_encryption import apply_paillier_encryption, initialize_paillier
from federated_learning.privacy.privacy_utils import apply_privacy_mechanism 