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

## Problem Definition

### Classification Task
Classify airports into **3 categories** based on geographic and engineered features:
- **Small airports** (small_airport, heliport, seaplane_base, etc.)
- **Medium airports** (medium_airport)
- **Large airports** (large_airport)

### Input Features
**Basic features (3):**
- `latitude_deg` - Geographic latitude
- `longitude_deg` - Geographic longitude
- `elevation_ft` - Elevation in feet

**Categorical features (2):**
- `continent` - Continent code (NA, EU, AS, SA, OC, AF, AN) → encoded
- `scheduled_service` - Whether airport has scheduled flights (yes/no) → binary

**Engineered features (26):**
- `abs_latitude` - Distance from equator
- `northern_hemisphere`, `eastern_hemisphere` - Hemisphere indicators
- `tropical_zone`, `temperate_zone`, `polar_zone` - Climate zone approximations
- `elevation_normalized`, `high_elevation`, `low_elevation` - Elevation patterns
- `lat_lon_interaction`, `lat_elev_interaction`, `lon_elev_interaction` - Geographic interactions
- `lat_squared`, `lon_squared`, `elev_squared` - Non-linear relationships
- `likely_americas`, `likely_europe_africa`, `likely_asia_oceania` - Continental regions (from lon)
- `is_north_america`, `is_europe`, `is_asia`, `is_south_america`, `is_oceania`, `is_africa` - Continent one-hot encoding
- `scheduled_low_elev`, `scheduled_high_lat` - Scheduled service interaction features

**Total: 31 features** used for classification

### Target Metric: **F1-Score (Weighted)**

**Why F1-Score?**
- The dataset is heavily **imbalanced** (42,475 small vs 488 large airports)
- **Accuracy** alone can be misleading on imbalanced data (predicting all "small" gives ~90% accuracy but is useless)
- **F1-score** balances precision and recall, giving a better measure of real classification quality
- **Weighted average** accounts for class imbalance

### Success Criteria

✓ **F1-Score >= 0.50** on test set (both implementations) → **Achieved: 0.60-0.61**  
✓ **All 3 classes predicted** (not just majority class) → **Achieved**  
✓ **Custom implementation within 5%** of sklearn performance → **Achieved: 1.5% difference**  
✓ **At least 1.5x improvement** over random baseline (0.33) → **Achieved: 1.82x**

**Performance achieved:** The project successfully exceeds all targets with F1-score of ~0.61, representing 82% improvement over random baseline. This is achieved through careful feature engineering combining geographic coordinates, categorical variables (continent, scheduled service), and their interactions.

## Dataset

The project uses a single dataset file: **`airports.csv`**, downloaded from Kaggle.  
The dataset contains selected airport-related attributes such as latitude, longitude, elevation, region, and airport type.

**Original distribution:**
- small_airport: 42,475
- heliport: 22,405
- medium_airport: 4,688
- large_airport: 488

**Balanced dataset (after intelligent sampling):**
- small: 500 samples
- medium: 500 samples  
- large: 488 samples (all available)

**Strategy:** Keep ALL minority class samples (large), balance majority classes to 500 max.

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

The custom implementation **successfully exceeds all project goals**:

**Key metrics:**
- **Custom RF F1-Score: ~0.60** ✓✓ (target: >= 0.50, **+20% over target!**)
- **Sklearn RF F1-Score: ~0.61** ✓✓ (target: >= 0.50, **+22% over target!**)
- **Difference: ~0.01** ✓ (target: < 0.05)
- **All 3 classes predicted** ✓
- **Improvement: 1.82x over random baseline** ✓✓ (target: >= 1.5x)

**Comparison to baseline:**
- Random guessing: F1 ~ 0.33 (3 classes)
- Current performance: F1 ~ 0.61
- **Improvement: 82% better than random (1.82x)**

**Impact of features:**
- Geographic only (21 features): F1 ~ 0.51
- + Categorical (continent, scheduled_service): F1 ~ 0.61
- **Improvement from categorical features: +20%**

**Key success factors:**
1. **Categorical features**: Continent and scheduled service provide strong signals
   - Scheduled service airports are typically larger (commercial vs. private)
   - Continental patterns reflect economic development (NA/EU vs. AF/OC)
2. **Feature engineering**: 26 derived features from 5 base features
   - Geographic patterns (climate zones, hemispheres)
   - Non-linear relationships (squared terms, interactions)
   - Domain knowledge (elevation patterns, regional indicators)
3. **Balanced sampling**: Preserves minority classes while preventing overfitting
4. **Ensemble learning**: 100 trees with random feature selection reduce variance

**Why not higher accuracy?**
Airport size is still primarily determined by economic factors not present in this dataset (city population, GDP, tourism statistics). The model successfully extracts maximum information from available geographic and categorical features.

The implementation successfully:
- Handles severe class imbalance (64K small vs 486 large airports)
- Predicts all three classes with good precision and recall
- Matches sklearn's performance closely (within 1.5%)
- Demonstrates proper Random Forest implementation from scratch

See `/experiments/experiments.md` for detailed results and analysis.

## Dependencies

- Python 3.8+
- NumPy
- pandas
- scikit-learn (for comparison)

See `requirements.txt` for specific versions.

## Testing

Comprehensive test suite validates implementation correctness:

### Unit Tests
- **Gini impurity calculation** - Validates split quality metric
- **Tree splitting logic** - Ensures correct binary splits
- **Bootstrap sampling** - Tests ensemble diversity mechanism
- **Majority voting** - Validates prediction aggregation
- **Random feature selection** - Tests sqrt/log2 feature sampling

### Feature Engineering Tests
- **Feature creation** - Validates all 31 features are generated correctly
- **Categorical encoding** - Tests continent and scheduled_service encoding
- **Data preprocessing** - Ensures proper handling of missing values and type mapping

### Performance Tests
- **Model accuracy** - Validates F1-score >= 0.40 (significantly above random baseline)
- **Class coverage** - Ensures all 3 classes are predicted
- **Real data validation** - Tests on actual airport dataset (1,486 samples)

### Performance Experiments
- **Effect of tree count** - Shows accuracy improvement with more trees (5→50)
- **Impact of tree depth** - Demonstrates depth vs. overfitting trade-off (3→15)

### Running Tests

```bash
# Run all tests
python tests/test_basic.py

# Expected output:
# ✓ All unit tests passed
# ✓ Feature engineering validated  
# ✓ Model performance meets requirements (F1 >= 0.40)
# ✓ Experiments completed successfully
```

**Test Results:**
- All unit tests: ✓ PASS
- Feature engineering: ✓ PASS
- Model performance: ✓ PASS (F1 = 0.60, well above 0.40 threshold)
- Experiments: ✓ PASS

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