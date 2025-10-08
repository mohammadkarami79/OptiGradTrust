"""
Experiments Package for Reviewer Feedback
==========================================

This package contains all experiments needed to address reviewer feedback
for the OptiGradTrust paper.
"""

__version__ = "1.0.0"
__author__ = "OptiGradTrust Team"

# Import all experiment modules
from . import ablation_study
from . import combined_attacks
from . import computational_overhead
from . import confidence_intervals
from . import extended_metrics
from . import fair_comparison
from . import scalability_tests
from . import feature_correlation
from . import extreme_heterogeneity
from . import statistical_tests
from . import visualization_suite
from . import preprocessing_docs

__all__ = [
    'ablation_study',
    'combined_attacks',
    'computational_overhead',
    'confidence_intervals',
    'extended_metrics',
    'fair_comparison',
    'scalability_tests',
    'feature_correlation',
    'extreme_heterogeneity',
    'statistical_tests',
    'visualization_suite',
    'preprocessing_docs',
]

