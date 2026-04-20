"""Aggregation strategies for OptiGradTrust federated learning."""
from federated_learning.aggregators.byzantine_baselines import (
    krum_aggregate,
    fltrust_aggregate,
    rfa_aggregate,
    signguard_aggregate,
)

__all__ = [
    'krum_aggregate',
    'fltrust_aggregate',
    'rfa_aggregate',
    'signguard_aggregate',
]
