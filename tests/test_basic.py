"""
Basic tests for Random Forest implementation
Tests: Gini impurity, splitting, voting
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from collections import Counter
from airport_random_forest.main import SimpleDecisionTree, SimpleRandomForest


def test_gini_impurity():
    """Test Gini impurity calculation"""
    print("Testing Gini impurity calculation...")
    
    tree = SimpleDecisionTree()
    
    # Pure node (all same class)
    y_pure = np.array([1, 1, 1, 1])
    gini_pure = tree.gini_impurity(y_pure)
    assert gini_pure == 0.0, f"Pure node should have Gini=0, got {gini_pure}"
    
    # 50-50 split
    y_half = np.array([0, 0, 1, 1])
    gini_half = tree.gini_impurity(y_half)
    expected = 0.5
    assert abs(gini_half - expected) < 0.01, f"50-50 split should have Gini≈0.5, got {gini_half}"
    
    # Multi-class
    y_multi = np.array([0, 0, 1, 1, 2, 2])
    gini_multi = tree.gini_impurity(y_multi)
    expected = 1 - 3 * (1/3)**2
    assert abs(gini_multi - expected) < 0.01, f"Expected Gini≈{expected}, got {gini_multi}"
    
    print("✓ Gini impurity tests passed")


def test_tree_splitting():
    """Test tree splitting on simple data"""
    print("\nTesting decision tree splitting...")
    
    # Simple separable data
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [10, 10],
        [11, 11],
        [12, 12]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    
    tree = SimpleDecisionTree(max_depth=5, min_samples_split=2)
    tree.fit(X, y)
    
    predictions = tree.predict(X)
    accuracy = np.mean(predictions == y)
    
    assert accuracy >= 0.5, f"Tree should classify better than random, got accuracy={accuracy}"
    
    print("✓ Tree splitting tests passed")


def test_bootstrap_sampling():
    """Test bootstrap sampling"""
    print("\nTesting bootstrap sampling...")
    
    rf = SimpleRandomForest(n_estimators=5)
    
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    
    X_boot, y_boot = rf.bootstrap_sample(X, y)
    
    # Bootstrap should have same size as original
    assert len(X_boot) == len(X), "Bootstrap sample should have same size"
    assert len(y_boot) == len(y), "Bootstrap labels should have same size"
    
    print("✓ Bootstrap sampling tests passed")


def test_majority_voting():
    """Test majority voting in Random Forest"""
    print("\nTesting majority voting...")
    
    # Create simple dataset
    X = np.array([
        [1, 1],
        [2, 2],
        [10, 10],
        [11, 11]
    ])
    y = np.array([0, 0, 1, 1])
    
    rf = SimpleRandomForest(n_estimators=5, max_depth=3)
    rf.fit(X, y)
    
    predictions = rf.predict(X)
    accuracy = np.mean(predictions == y)
    
    assert accuracy >= 0.5, f"Random Forest should perform reasonably, got accuracy={accuracy}"
    
    print("✓ Majority voting tests passed")


def test_random_features():
    """Test random feature selection"""
    print("\nTesting random feature selection...")
    
    rf = SimpleRandomForest(n_estimators=3)
    
    # Test sqrt
    n_features = 9
    max_features = rf.get_max_features(n_features)
    assert max_features == 3, f"sqrt(9) should be 3, got {max_features}"
    
    # Test log2
    rf.max_features = 'log2'
    n_features = 8
    max_features = rf.get_max_features(n_features)
    assert max_features == 3, f"log2(8) should be 3, got {max_features}"
    
    print("✓ Random feature selection tests passed")

def experiment1_effect_of_number_of_trees(X_data, y_data):
    
    print("\n--- Running Experiment 1: Effect of Number of Trees (5 to 50) ---")
    
    # Use only a small sample for quick runtime
    X = X_data[:50]
    y = y_data[:50]
    
    n_estimators_list = [5, 10, 20, 30, 50]
    
    for n in n_estimators_list:
        # Train and test a minimal Random Forest model
        rf = SimpleRandomForest(n_estimators=n, max_depth=5)
        rf.fit(X, y)
        predictions = rf.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"  Trees (n_estimators)={n}: Accuracy={accuracy:.4f}")
        
    print(f"Expected Result: Accuracy should generally improve and then stabilize.")
    print("✓ Experiment 1 completed")


def experiment2_impact_of_tree_depth(X_data, y_data):
    
    print("\n--- Running Experiment 2: Impact of Tree Depth (3 to 15) ---")
    
    X = X_data[:50]
    y = y_data[:50]
    
    depth_list = [3, 6, 10, 15]
    
    for d in depth_list:
        # Train a minimal Random Forest model with varied max_depth
        rf = SimpleRandomForest(n_estimators=10, max_depth=d)
        rf.fit(X, y)
        # Check performance on the training set (X) to observe potential overfitting
        predictions = rf.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"  Max_Depth={d}: Train Accuracy={accuracy:.4f}")

    print(f"Expected Result: Accuracy should increase with depth; high depth (e.g., 15) may hit 1.0 (overfitting).")
    print("✓ Experiment 2 completed")


def run_all_tests():
    """Run all basic tests and the first two performance experiments"""
    
    # --- Simulated Data Loading (Required for Experiments) ---
    # Create a simulated dataset for performance tests (100 samples, 3 features, 3 classes)
    np.random.seed(42)
    X_simulated = np.random.rand(100, 3) * 100 
    y_simulated = np.random.randint(0, 3, 100) 
    # -------------------------------------------------------------------
    
    print("=" * 60)
    print("Running Basic Tests")
    print("=" * 60)
    
    # 1. Unit Tests
    test_gini_impurity()
    test_tree_splitting()
    test_bootstrap_sampling()
    test_majority_voting()
    test_random_features()
    
    print("\n" + "=" * 60)
    print("Running 2 Performance Experiments")
    print("=" * 60)
    
    # 2. Executing Experiments 
    experiment1_effect_of_number_of_trees(X_simulated, y_simulated)
    experiment2_impact_of_tree_depth(X_simulated, y_simulated)
    
    print("\n" + "=" * 60)
    print("All Tests and Experiments Concluded!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
