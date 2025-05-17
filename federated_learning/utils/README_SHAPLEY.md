# Shapley Value Integration in Federated Learning

## Overview

This module implements Shapley value calculations for federated learning to measure each client's contribution to the global model's performance. Shapley values provide a principled, game-theoretic approach to fairly attribute contributions in collaborative settings.

## Key Features

1. **Monte Carlo Shapley Value Estimation**: Efficiently approximates Shapley values using Monte Carlo sampling to handle the combinatorial complexity.

2. **Integration with Dual Attention Mechanism**: Shapley values are incorporated as a sixth feature in our gradient evaluation system, complementing the existing 5-feature vector.

3. **Contribution-Based Client Weighting**: Clients with higher Shapley values (those contributing more to model performance) receive higher weights during gradient aggregation.

4. **Malicious Client Detection**: Abnormally low Shapley values can indicate potential malicious behavior, enhancing detection capabilities.

## Configuration Parameters

The Shapley value implementation can be configured through the following parameters in `config.py`:

- `ENABLE_SHAPLEY`: Boolean flag to enable/disable Shapley value calculation
- `SHAPLEY_SAMPLES`: Number of Monte Carlo samples for Shapley estimation (higher = more accurate but slower)
- `SHAPLEY_WEIGHT`: Weight of Shapley value in the final trust score (0.0-1.0)
- `VALIDATION_RATIO`: Ratio of test data to use for validation during Shapley calculation
- `SHAPLEY_BATCH_SIZE`: Batch size for validation during Shapley calculation

## Implementation Details

### Shapley Value Calculation

For a given set of clients, Shapley values are calculated by:

1. Measuring baseline model performance with no client updates
2. For each permutation of clients (sampled using Monte Carlo):
   - Apply updates sequentially and measure the marginal contribution of each client
   - The Shapley value is the average marginal contribution across all sampled permutations

### Integration into Gradient Aggregation

The Shapley values are integrated into the client gradient weighting process:

1. Standard gradient features are extracted (reconstruction error, similarity to root gradients, etc.)
2. Shapley values are calculated and added as a sixth feature
3. The dual attention mechanism incorporates Shapley values into its trust score calculation
4. Final client weights for gradient aggregation reflect both trustworthiness and contribution

## Performance Considerations

Shapley value calculation adds computational overhead:

- For each round, an extra validation pass is performed for each client permutation
- The computation scales with the number of clients and SHAPLEY_SAMPLES
- For larger models or many clients, consider:
  - Reducing SHAPLEY_SAMPLES
  - Performing Shapley calculation only every N rounds
  - Using a smaller validation dataset

## Example Usage

```python
# Enable Shapley value calculation in config.py
ENABLE_SHAPLEY = True
SHAPLEY_SAMPLES = 5
SHAPLEY_WEIGHT = 0.3

# The Server will automatically:
# 1. Create a validation dataset
# 2. Calculate Shapley values during gradient aggregation
# 3. Incorporate them into client weighting
```

## References

- Shapley, L. S. (1953). A value for n-person games. Contributions to the Theory of Games, 2(28), 307-317.
- Wang, T., Rausch, J., Zhang, C., Jia, R., & Song, D. (2020). A principled approach to data valuation for federated learning. In Federated Learning (pp. 153-167). Springer, Cham. 