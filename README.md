# Random Forest Classification of Airport Types
### (Student Project – Python Programming, 2025)

## Overview

This is a simple, academic project created for the **Python Programming course**.  
The goal is to implement a **basic Random Forest classifier from scratch**, using a small airport dataset (`airports.csv`) and to compare the results with the official implementation from **scikit-learn**.

The project follows the official course requirements:
- VirtualEnv  
- PyScaffold project structure  
- own implementation vs scikit-learn implementation  
- complete documentation  
- basic unit tests  

The project is intentionally kept **short, clear, and easy to understand**.

## Dataset

The project uses a single dataset file: **`airports.csv`**, downloaded from Kaggle.  
The dataset contains selected airport-related attributes such as latitude, longitude, elevation, region, and airport type.

The target variable in the project is the **airport type** (e.g., small, medium, large).

## Project Goals

1. Implement a simple **CART Decision Tree**:
   - Gini impurity  
   - threshold splits  
   - minimal stopping rules  

2. Implement a minimal **Random Forest classifier**:
   - bootstrap sampling  
   - random feature subsets  
   - majority voting  

3. Compare the results with `sklearn.ensemble.RandomForestClassifier`.

4. Document the algorithm in clear, student-friendly language.

5. Write simple tests verifying key components.

## Project Structure

```
project_root/
├── README.md
├── airports.csv             # Kaggle airport dataset
├── docs/
│   └── documentation.md     # description of algorithm + how it works
├── experiments/
│   └── experiments.md       # results + comparison vs scikit-learn
├── src/airport_random_forest/
│   └── main.py              # main file: data loading, training, comparison
├── tests/
│   └── test_basic.py        # simple tests (impurity, split, voting)
├── requirements.txt
└── setup.cfg                # PyScaffold configuration
```

All project files are minimal so the project stays easy to read and evaluate.

## How to Run

### 1. Create and activate VirtualEnv

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the main file

```bash
python src/airport_random_forest/main.py
```

This will:
- Load the airport dataset
- Train the custom Random Forest
- Train sklearn Random Forest
- Compare the results

### 4. Run the tests

```bash
# Install pytest if not already installed
pip install pytest pytest-cov

# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run tests with coverage report
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_basic.py
```

## Experiments

The experiments include:
- Influence of the number of trees (`n_estimators`)
- Comparison of impurity functions (Gini vs Entropy)
- Effect of bootstrap sampling
- Feature importance (basic)
- Accuracy comparison: **custom Random Forest vs scikit-learn**

All results are described in `/experiments/experiments.md`.

## Documentation

A short, academic, easy-to-understand description of:
- Decision trees  
- The Random Forest algorithm  
- The logic behind the implementation  
- Assumptions and simplifications  

is included in `/docs/documentation.md`.

## Implementation Details

### SimpleDecisionTree
- Uses CART algorithm
- Gini impurity for split quality
- Binary splits on feature thresholds
- Recursive tree building

### SimpleRandomForest
- Bootstrap sampling for each tree
- Random feature selection (√n features)
- Majority voting for predictions
- Multiple trees trained independently

## Results

The custom implementation achieves **identical accuracy** to scikit-learn (~50% on balanced 3-class problem, baseline: 33%), demonstrating correct understanding and implementation of the Random Forest algorithm.

**Key metrics:**
- Accuracy: 0.5034 (both custom and sklearn)
- F1 Score: 0.4880 (custom) vs 0.4878 (sklearn)
- Difference: <0.001 (negligible)

The implementation successfully:
- Handles class imbalance through balanced sampling
- Predicts all three classes (not just majority)
- Matches sklearn's performance exactly
- Demonstrates proper ensemble learning

See `/experiments/experiments.md` for detailed results and analysis.

## Dependencies

- Python 3.8+
- NumPy
- pandas
- scikit-learn (for comparison)

See `requirements.txt` for specific versions.

## Testing

Basic tests verify:
- Correctness of Gini impurity calculation  
- Correctness of choosing a split  
- Tree prediction on simple data  
- Bootstrap sampling  
- Majority voting  

Tests are implemented in: `/tests/test_basic.py`

## Final Notes

This project is intentionally simple and structured in the most readable way possible.  
The aim is to clearly show:
- understanding of the algorithm  
- correct usage of PyScaffold and VirtualEnv  
- correct comparison with scikit-learn  
- clean documentation and tests

## Author

Student project made by Wojciech Domino & Mateusz Maj for Python Programming course, 2025.