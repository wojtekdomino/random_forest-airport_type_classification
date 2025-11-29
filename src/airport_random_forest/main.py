"""
Random Forest Classification - Student Project
Simple implementation from scratch for educational purposes
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# ============================================================================
# PART 1: Decision Tree Implementation
# ============================================================================

class SimpleDecisionTree:
    """
    A minimal CART decision tree for classification.
    Uses Gini impurity and binary splits.
    """
    
    def __init__(self, max_depth=5, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def gini_impurity(self, y):
        """Calculate Gini impurity for a set of labels"""
        if len(y) == 0:
            return 0
        counter = Counter(y)
        impurity = 1.0
        for count in counter.values():
            p = count / len(y)
            impurity -= p ** 2
        return impurity
    
    def find_best_split(self, X, y):
        """Find the best feature and threshold to split on"""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                # Weighted Gini
                gini = (len(y_left) * self.gini_impurity(y_left) + 
                       len(y_right) * self.gini_impurity(y_right)) / len(y)
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        # Stopping conditions
        if (depth >= self.max_depth or 
            len(y) < self.min_samples_split or 
            len(np.unique(y)) == 1):
            # Return leaf node with majority class
            return Counter(y).most_common(1)[0][0]
        
        # Find best split
        feature, threshold = self.find_best_split(X, y)
        
        if feature is None:
            return Counter(y).most_common(1)[0][0]
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # Build subtrees
        left_tree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        """Train the decision tree"""
        self.tree = self.build_tree(X, y)
        return self
    
    def predict_one(self, x, tree):
        """Predict class for a single sample"""
        if not isinstance(tree, dict):
            return tree
        
        if x[tree['feature']] <= tree['threshold']:
            return self.predict_one(x, tree['left'])
        else:
            return self.predict_one(x, tree['right'])
    
    def predict(self, X):
        """Predict classes for multiple samples"""
        return np.array([self.predict_one(x, self.tree) for x in X])


# ============================================================================
# PART 2: Random Forest Implementation
# ============================================================================

class SimpleRandomForest:
    """
    A minimal Random Forest classifier.
    Uses bootstrap sampling and random feature subsets.
    """
    
    def __init__(self, n_estimators=10, max_depth=5, max_features='sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []
    
    def bootstrap_sample(self, X, y):
        """Create a bootstrap sample (sample with replacement)"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]
    
    def get_max_features(self, n_features):
        """Determine number of features to use"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        else:
            return n_features
    
    def fit(self, X, y):
        """Train the random forest"""
        self.trees = []
        self.feature_indices = []
        
        n_features = X.shape[1]
        max_features = self.get_max_features(n_features)
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            X_sample, y_sample = self.bootstrap_sample(X, y)
            
            # Random feature subset
            feature_idx = np.random.choice(n_features, max_features, replace=False)
            X_subset = X_sample[:, feature_idx]
            
            # Train tree
            tree = SimpleDecisionTree(max_depth=self.max_depth)
            tree.fit(X_subset, y_sample)
            
            self.trees.append(tree)
            self.feature_indices.append(feature_idx)
        
        return self
    
    def predict(self, X):
        """Predict using majority voting"""
        predictions = []
        
        for tree, feature_idx in zip(self.trees, self.feature_indices):
            X_subset = X[:, feature_idx]
            pred = tree.predict(X_subset)
            predictions.append(pred)
        
        # Majority voting
        predictions = np.array(predictions)
        final_predictions = []
        
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            most_common = Counter(votes).most_common(1)[0][0]
            final_predictions.append(most_common)
        
        return np.array(final_predictions)


# ============================================================================
# PART 3: Data Loading and Preprocessing
# ============================================================================

def load_data(filepath='airports.csv'):
    """Load and preprocess the airport dataset"""
    print("Loading data from:", filepath)
    
    # Load CSV
    df = pd.read_csv(filepath)
    
    # Select features and target
    feature_cols = ['latitude_deg', 'longitude_deg', 'elevation_ft']
    target_col = 'type'
    
    # Remove rows with missing values
    df = df.dropna(subset=feature_cols + [target_col])
    
    # Convert to numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows that couldn't be converted
    df = df.dropna(subset=feature_cols)
    
    # Filter to keep only main airport types (small, medium, large)
    # We'll simplify the classification
    type_mapping = {
        'small_airport': 'small',
        'medium_airport': 'medium', 
        'large_airport': 'large',
        'heliport': 'small',
        'seaplane_base': 'small',
        'balloonport': 'small',
        'closed': 'small'
    }
    
    df['airport_category'] = df['type'].map(type_mapping)
    df = df.dropna(subset=['airport_category'])
    
    # Extract features and target
    X = df[feature_cols].values
    y = df['airport_category'].values
    
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Features: {feature_cols}")
    print(f"Classes: {np.unique(y)}")
    print(f"Class distribution: {Counter(y)}")
    
    return X, y


# ============================================================================
# PART 4: Main Execution - Training and Comparison
# ============================================================================

def main():
    print("=" * 70)
    print("Random Forest Classification - Student Project")
    print("=" * 70)
    print()
    
    # Load data
    X, y = load_data('airports.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # ========================================================================
    # Train custom Random Forest
    # ========================================================================
    print("\n" + "-" * 70)
    print("Training CUSTOM Random Forest...")
    print("-" * 70)
    
    custom_rf = SimpleRandomForest(n_estimators=20, max_depth=8, max_features='sqrt')
    custom_rf.fit(X_train, y_train)
    
    custom_pred = custom_rf.predict(X_test)
    custom_accuracy = accuracy_score(y_test, custom_pred)
    custom_f1 = f1_score(y_test, custom_pred, average='weighted')
    
    print(f"Accuracy: {custom_accuracy:.4f}")
    print(f"F1 Score: {custom_f1:.4f}")
    
    # ========================================================================
    # Train sklearn Random Forest
    # ========================================================================
    print("\n" + "-" * 70)
    print("Training SKLEARN Random Forest...")
    print("-" * 70)
    
    sklearn_rf = RandomForestClassifier(
        n_estimators=20, 
        max_depth=8, 
        max_features='sqrt',
        random_state=42
    )
    sklearn_rf.fit(X_train, y_train)
    
    sklearn_pred = sklearn_rf.predict(X_test)
    sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
    sklearn_f1 = f1_score(y_test, sklearn_pred, average='weighted')
    
    print(f"Accuracy: {sklearn_accuracy:.4f}")
    print(f"F1 Score: {sklearn_f1:.4f}")
    
    # ========================================================================
    # Comparison
    # ========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"\nCustom RF Accuracy:  {custom_accuracy:.4f}")
    print(f"Sklearn RF Accuracy: {sklearn_accuracy:.4f}")
    print(f"Difference:          {abs(custom_accuracy - sklearn_accuracy):.4f}")
    
    print(f"\nCustom RF F1:        {custom_f1:.4f}")
    print(f"Sklearn RF F1:       {sklearn_f1:.4f}")
    print(f"Difference:          {abs(custom_f1 - sklearn_f1):.4f}")
    
    # Confusion matrices
    print("\n" + "-" * 70)
    print("Custom RF Confusion Matrix:")
    print(confusion_matrix(y_test, custom_pred))
    
    print("\nSklearn RF Confusion Matrix:")
    print(confusion_matrix(y_test, sklearn_pred))
    
    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
