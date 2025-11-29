# Documentation - Random Forest Implementation

## Introduction

This document describes a simple Random Forest classifier implemented from scratch for educational purposes. The implementation is intentionally kept minimal and easy to understand.

## Decision Trees

### What is a Decision Tree?

A decision tree is a tree structure where:
- Each internal node represents a test on a feature
- Each branch represents the outcome of the test
- Each leaf node represents a class label

### CART Algorithm

We use the CART (Classification and Regression Trees) algorithm:

1. Start with all training data at the root
2. Find the best feature and threshold to split the data
3. Split the data into two groups based on this threshold
4. Repeat recursively for each group
5. Stop when a stopping condition is met

### Gini Impurity

To measure the quality of a split, we use Gini impurity:

```
Gini(S) = 1 - Σ(p_i)²
```

where `p_i` is the proportion of class `i` in set `S`.

**Example:**
- Pure node (all same class): Gini = 0
- 50-50 split (two classes): Gini = 0.5
- Random guess (three equal classes): Gini ≈ 0.67

### Finding the Best Split

For each feature and each possible threshold:
1. Split data into left (≤ threshold) and right (> threshold)
2. Calculate weighted Gini impurity
3. Choose the split with the lowest Gini

### Stopping Conditions

The tree stops growing when:
- Maximum depth is reached
- Node has too few samples
- All samples belong to the same class

## Random Forest

### What is Random Forest?

Random Forest is an ensemble method that:
1. Builds multiple decision trees
2. Trains each tree on a different subset of data
3. Makes predictions by majority voting

### Bootstrap Sampling

For each tree, we create a bootstrap sample:
- Sample with replacement from the training data
- Same size as original dataset
- Some samples appear multiple times, others not at all

This introduces diversity between trees.

### Random Feature Selection

At each split, we only consider a random subset of features:
- Typically √(number of features)
- Reduces correlation between trees
- Improves generalization

### Majority Voting

To make a prediction:
1. Pass the input through all trees
2. Collect all predictions
3. Return the class with the most votes

## Implementation Details

### SimpleDecisionTree

**Parameters:**
- `max_depth`: Maximum tree depth (default: 5)
- `min_samples_split`: Minimum samples to split a node (default: 10)

**Methods:**
- `gini_impurity(y)`: Calculate Gini impurity
- `find_best_split(X, y)`: Find best feature and threshold
- `build_tree(X, y, depth)`: Recursively build tree
- `fit(X, y)`: Train the tree
- `predict(X)`: Make predictions

### SimpleRandomForest

**Parameters:**
- `n_estimators`: Number of trees (default: 10)
- `max_depth`: Maximum depth of each tree (default: 5)
- `max_features`: Number of features per split (default: 'sqrt')

**Methods:**
- `bootstrap_sample(X, y)`: Create bootstrap sample
- `fit(X, y)`: Train all trees
- `predict(X)`: Predict using majority voting

## Simplifications

Compared to sklearn, our implementation:

1. **Simpler structure**: All code in one file
2. **Basic splitting**: Only binary splits on thresholds
3. **No optimization**: Uses simple loops (not optimized algorithms)
4. **Limited features**: Only basic parameters
5. **No parallelization**: Trees trained sequentially

## Why Random Forest Works

Random Forest is effective because:

1. **Ensemble effect**: Multiple trees are better than one
2. **Reduced overfitting**: Bootstrap and random features prevent overfitting
3. **Variance reduction**: Averaging reduces variance
4. **Handles non-linearity**: Can model complex relationships

## Example Usage

```python
from main import SimpleRandomForest

# Create and train
rf = SimpleRandomForest(n_estimators=20, max_depth=8)
rf.fit(X_train, y_train)

# Predict
predictions = rf.predict(X_test)
```

## Comparison with sklearn

Our implementation produces similar results to sklearn but:
- Slower (no optimization)
- Simpler (easier to understand)
- Limited (fewer features)

This is expected and acceptable for an educational project.

## References

- Breiman, L. (2001). Random Forests. Machine Learning.
- Scikit-learn documentation: Random Forests
- CART algorithm (Breiman et al., 1984)
