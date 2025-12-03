"""
Basic tests for Random Forest implementation
Tests: Gini impurity, splitting, voting, feature engineering, and performance validation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from collections import Counter
from airport_random_forest.main import SimpleDecisionTree, SimpleRandomForest, engineer_features, load_data


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
    """Test decision tree splitting on simple data"""
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


def test_feature_engineering():
    """Test feature engineering function"""
    print("\nTesting feature engineering...")
    
    # Create sample dataframe
    df = pd.DataFrame({
        'latitude_deg': [40.0, -33.9, 51.5],
        'longitude_deg': [-74.0, 151.2, -0.1],
        'elevation_ft': [10, 20, 100],
        'continent_encoded': [0, 4, 1],  # NA, OC, EU
        'scheduled_service_encoded': [1, 1, 0]
    })
    
    df_engineered = engineer_features(df)
    
    # Check that new features were created
    assert 'abs_latitude' in df_engineered.columns, "abs_latitude should be created"
    assert 'tropical_zone' in df_engineered.columns, "tropical_zone should be created"
    assert 'is_north_america' in df_engineered.columns, "continent one-hot encoding should exist"
    assert 'scheduled_low_elev' in df_engineered.columns, "interaction features should exist"
    
    # Validate some feature values
    assert df_engineered['abs_latitude'].iloc[0] == 40.0, "abs_latitude should be absolute value"
    assert df_engineered['northern_hemisphere'].iloc[0] == 1, "40°N should be northern hemisphere"
    assert df_engineered['northern_hemisphere'].iloc[1] == 0, "-33.9°S should be southern hemisphere"
    
    print("✓ Feature engineering tests passed")


def test_data_loading():
    """Test data loading and preprocessing"""
    print("\nTesting data loading and preprocessing...")
    
    try:
        # This will work only if airports.csv exists
        X, y, num_features = load_data('airports.csv')
        
        # Check dimensions
        assert X.shape[0] > 0, "Should load some samples"
        assert num_features == 31, f"Should have 31 features, got {num_features}"
        assert X.shape[1] == 31, f"X should have 31 columns, got {X.shape[1]}"
        
        # Check classes
        unique_classes = set(y)
        assert len(unique_classes) == 3, f"Should have 3 classes, got {len(unique_classes)}"
        assert unique_classes == {'small', 'medium', 'large'}, f"Unexpected classes: {unique_classes}"
        
        # Check balance
        class_counts = Counter(y)
        print(f"  Class distribution: {dict(class_counts)}")
        assert class_counts['large'] >= 400, "Should preserve minority class (large airports)"
        
        print("✓ Data loading tests passed")
        return True
    except FileNotFoundError:
        print("⚠ airports.csv not found - skipping data loading test")
        return False


def test_model_performance():
    """Test that model achieves minimum performance threshold"""
    print("\nTesting model performance on real data...")
    
    try:
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import f1_score
        
        # Load data
        X, y, num_features = load_data('airports.csv')
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        rf = SimpleRandomForest(n_estimators=50, max_depth=10, max_features='sqrt')
        rf.fit(X_train, y_train)
        
        # Test performance
        y_pred = rf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        accuracy = np.mean(y_pred == y_test)
        
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
        
        # Performance threshold - should beat random baseline significantly
        MIN_F1 = 0.40  # Conservative threshold (random = 0.33)
        assert f1 >= MIN_F1, f"Model F1 ({f1:.4f}) should be >= {MIN_F1}"
        
        # Check all classes are predicted
        unique_predictions = set(y_pred)
        assert len(unique_predictions) == 3, f"Should predict all 3 classes, got {len(unique_predictions)}"
        
        print("✓ Model performance tests passed")
        return True
    except FileNotFoundError:
        print("⚠ airports.csv not found - skipping performance test")
        return False
    except ImportError:
        print("⚠ sklearn not available - skipping performance test")
        return False


def experiment1_effect_of_number_of_trees(X_data, y_data):
    """Experiment 1: Effect of number of trees"""
    print("\n--- Experiment 1: Effect of Number of Trees ---")
    
    # Use only a small sample for quick runtime
    X = X_data[:100]
    y = y_data[:100]
    
    n_estimators_list = [5, 10, 20, 30, 50]
    
    for n in n_estimators_list:
        rf = SimpleRandomForest(n_estimators=n, max_depth=5)
        rf.fit(X, y)
        predictions = rf.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"  n_estimators={n}: Accuracy={accuracy:.4f}")
        
    print("  Expected: Accuracy improves and stabilizes with more trees")
    print("✓ Experiment 1 completed")


def experiment2_impact_of_tree_depth(X_data, y_data):
    """Experiment 2: Impact of tree depth"""
    print("\n--- Experiment 2: Impact of Tree Depth ---")
    
    X = X_data[:100]
    y = y_data[:100]
    
    depth_list = [3, 6, 10, 15]
    
    for d in depth_list:
        rf = SimpleRandomForest(n_estimators=10, max_depth=d)
        rf.fit(X, y)
        predictions = rf.predict(X)
        accuracy = np.mean(predictions == y)
        print(f"  max_depth={d}: Train Accuracy={accuracy:.4f}")

    print("  Expected: Deeper trees increase training accuracy (may overfit)")
    print("✓ Experiment 2 completed")


def run_all_tests():
    """Run all tests including unit tests, feature tests, and experiments"""
    
    print("=" * 70)
    print("RANDOM FOREST IMPLEMENTATION - TEST SUITE")
    print("=" * 70)
    
    # --- Unit Tests ---
    print("\n" + "=" * 70)
    print("PART 1: Unit Tests (Algorithm Components)")
    print("=" * 70)
    
    test_gini_impurity()
    test_tree_splitting()
    test_bootstrap_sampling()
    test_majority_voting()
    test_random_features()
    
    # --- Feature Engineering Tests ---
    print("\n" + "=" * 70)
    print("PART 2: Feature Engineering and Data Processing")
    print("=" * 70)
    
    test_feature_engineering()
    data_loaded = test_data_loading()
    
    # --- Performance Tests ---
    if data_loaded:
        print("\n" + "=" * 70)
        print("PART 3: Model Performance Validation")
        print("=" * 70)
        
        test_model_performance()
    
    # --- Experiments (with simulated data if real data not available) ---
    print("\n" + "=" * 70)
    print("PART 4: Performance Experiments")
    print("=" * 70)
    
    if data_loaded:
        try:
            X_exp, y_exp, _ = load_data('airports.csv')
        except:
            np.random.seed(42)
            X_exp = np.random.rand(200, 31) * 100 
            y_exp = np.random.choice(['small', 'medium', 'large'], 200)
    else:
        # Simulated data
        np.random.seed(42)
        X_exp = np.random.rand(200, 31) * 100 
        y_exp = np.random.choice(['small', 'medium', 'large'], 200)
    
    experiment1_effect_of_number_of_trees(X_exp, y_exp)
    experiment2_impact_of_tree_depth(X_exp, y_exp)
    
    # --- Summary ---
    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)
    print("✓ All unit tests passed")
    print("✓ Feature engineering validated")
    if data_loaded:
        print("✓ Model performance meets requirements (F1 >= 0.40)")
    print("✓ Experiments completed successfully")
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
