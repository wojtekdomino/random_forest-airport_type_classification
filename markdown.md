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

---

## Dataset
The project uses a single dataset file: **`airports.csv`**, downloaded from Kaggle.  
The dataset contains selected airport-related attributes such as latitude, longitude, elevation, region, and airport type.

The target variable in the project is the **airport type** (e.g., small, medium, large). - this column is not in the dataset - we have to predict it

---

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

---

## Project Structure (minimal, academic)
project_root/
├── README.md
├── airports.csv # Kaggle airport dataset
├── docs/
│ └── documentation.md # description of algorithm + how it works
├── experiments/
│ └── experiments.md # results + comparison vs scikit-learn
├── src/airport_random_forest/
│ └── main.py # main file: data loading, training, comparison
├── tests/
│ └── test_basic.py # simple tests (impurity, split, voting)
└── setup.cfg # created by PyScaffold

All project files are minimal so the project stays easy to read and evaluate.

---

## Experiments
The experiments include:
- influence of the number of trees (`n_estimators`)
- comparison of impurity functions (Gini vs Entropy)
- effect of bootstrap sampling
- feature importance (basic)
- accuracy comparison: **custom Random Forest vs scikit-learn**

All results are described in `/experiments/experiments.md`.

---

## Testing
The project includes basic tests verifying:
- correctness of Gini impurity calculation  
- correctness of choosing a split  
- tree prediction on simple data  
- bootstrap sampling  
- majority voting  

Tests are implemented in a single file: `/tests/test_basic.py`.

---

## Documentation
A short, academic, easy-to-understand description of:
- decision trees  
- the Random Forest algorithm  
- the logic behind the implementation  
- assumptions and simplifications  

is included in `/docs/documentation.md`.

---

## How to Run
### 1. Create and activate VirtualEnv
```
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 2. Install project (PyScaffold)
```
pip install -r requirements.txt
```

### 3. Run the main file
```
python src/airport_random_forest/main.py
```

### 4. Run the tests
```
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

---

## Results

| Metric | Custom RF | Sklearn RF | Difference |
|--------|-----------|------------|------------|
| Accuracy | 0.5034 | 0.5034 | 0.0000 |
| F1 Score | 0.4880 | 0.4878 | 0.0002 |

**Key findings:**
- Custom implementation achieves **identical** accuracy to scikit-learn
- ~50% accuracy is **good** for balanced 3-class problem (baseline: 33%)
- Model correctly predicts all three classes
- Implementation successfully handles class imbalance
- Demonstrates proper understanding of Random Forest algorithm

Detailed results and confusion matrices in `/experiments/experiments.md`.