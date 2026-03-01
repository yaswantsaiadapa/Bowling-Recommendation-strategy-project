# Cricket Strategy Recommendation System

## Project Overview

A comprehensive **Data Science and Machine Learning system** for cricket performance analysis and strategic recommendations. This project analyzes ball-by-ball cricket data to extract meaningful patterns about batsman and bowler performance, predict match outcomes, and recommend optimal strategies for team composition and field placement.

The system combines statistical analysis, machine learning models, and interactive visualizations to provide data-driven insights into cricket gameplay and player performance metrics.

---

## Problem Statement

Cricket performance depends on numerous interconnected factors:
- **Batsman Performance**: Strike rate and boundary percentage vary with bowler type and match phase
- **Bowler Effectiveness**: Success depends on pitch conditions, opposition strengths, and bowling strategy
- **Match Phases**: Powerplay (1-6), Middle (7-15), and Death (16-20) overs require different approaches
- **Field Placement**: Optimal fielder positions depend on batsman style, bowler type, pitch conditions, and phase

This system addresses these by:
1. Analyzing historical cricket data to extract performance patterns
2. Building predictive models for boundaries, good balls, and wickets
3. Generating strategic recommendations based on real-world metrics
4. Providing interactive dashboards for data exploration and visualization

---

## Key Features

### 1. **Batsman Analysis**
- Comprehensive batting metrics: runs, strike rate, boundary percentage, dismissal rate
- Adaptability analysis across different bowler types
- Phase-wise performance breakdown (Powerplay, Middle, Death)
- Batsman similarity matrices for matchup analysis

### 2. **Bowler Analysis**
- Complete bowler statistics: economy rate, strike rate, dot ball percentage, wicket percentage
- Bowler strength categorization (Elite, Strong, Average, Weak)
- Success index combining multiple performance metrics
- Bowling style effectiveness analysis

### 3. **Predictive Models**
- **Boundary Prediction**: Identifies deliveries likely to yield 4 or 6 runs
- **Good Ball Prediction**: Detects restrictive deliveries (≤1 run, no boundaries)
- **Pitch Analysis**: Classifies pitch line and length characteristics
- **Ball Type Classification**: Identifies delivery type (fast, spin, etc.)

### 4. **Strategic Recommendations**
- Field placement suggestions based on:
  - Pitch conditions (line and length)
  - Bowler type and style
  - Batsman style and preferences
  - Current match phase
- Batting strategy recommendations against specific opposition
- Matchup success rate analysis between batsman-bowler pairs

### 5. **Phase-wise Analysis**
- Strike rate trends across match phases
- Boundary percentage analysis
- Adaptability metrics per phase
- Playing role identification

### 6. **Interactive Dashboard**
- Plotly-based web interface for data exploration
- Player performance comparisons
- Real-time analytics and visualizations
- Match simulation and strategy testing

---

## Project Structure

```
cricket-strategy/
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
│
├── data/                                  # Data directory
│   ├── csv/                              # Raw cricket data
│   │   ├── ballbyball.csv                # Ball-by-ball match data
│   │   ├── country.csv                   # Country information
│   │   ├── ground.csv                    # Cricket ground details
│   │   ├── matches.csv                   # Match metadata
│   │   ├── players.csv                   # Player information
│   │   ├── season.csv                    # Season information
│   │   ├── team.csv                      # Team data
│   │   └── town.csv                      # Town/location data
│   │
│   ├── final_processed_data.csv          # Main processed dataset
│   ├── Preprocessed_Data.csv             # Alternative preprocessed data
│   ├── batsman_profiles.csv              # Batsman statistics
│   ├── batsman_stats.csv                 # Batsman performance metrics
│   ├── batsman_similarity.csv            # Batsman similarity matrix
│   ├── bowler_stats.csv                  # Bowler performance metrics
│   ├── phase_sr.csv                      # Phase-wise strike rate analysis
│   └── notebook_extraction_report.json   # Data extraction metadata
│
├── models/                               # Trained ML models (Joblib)
│   ├── predict_boundary_model.joblib
│   ├── predict_good_ball_model_random_forest.joblib
│   ├── pitch_line_model.joblib
│   ├── pitch_length_model.joblib
│   └── model_balltype.joblib
│
├── model_feature_engineered_datasets/   # Feature datasets for models
│   ├── df_model_boundary.csv
│   ├── df_model_good_ball.csv
│   ├── df_model_line.csv
│   └── df_model_ball_type.csv
│
├── notebooks/                            # Jupyter analysis notebooks
│   ├── data_exploration_and_processing.ipynb
│   ├── data_processing_1.ipynb
│   ├── core_batsman_analysis.ipynb
│   ├── batsman_analysis.ipynb
│   ├── core_bowling_analysis.ipynb
│   ├── phase_wise_analysis.ipynb
│   ├── batsman_analysis_engine.ipynb
│   ├── predict_boundary_and_batsman_analysis_engine.ipynb
│   └── plotly_dashboard.ipynb
│
├── static/                               # Web assets
│   └── cricket.jpg                       # Cricket theme image
│
└── Cricket strategy Recommendation System.pdf
```

---

## Data Overview

### Raw Source Data (data/csv/)
Comprehensive cricket dataset covering:
- **ballbyball.csv**: Individual delivery information including bowler, batsman, runs, wickets, and pitch details
- **matches.csv**: Match-level metadata (venues, teams, dates)
- **players.csv**: Player information and profiles
- **team.csv, season.csv, country.csv, ground.csv, town.csv**: Reference and contextual data

### Processed Datasets (data/)
- **final_processed_data.csv**: Integrated dataset combining all raw data with engineered features
- **batsman_profiles.csv**: Aggregated batsman-level statistics
- **bowler_stats.csv**: Aggregated bowler-level statistics
- **phase_sr.csv**: Phase-specific performance metrics
- **Preprocessed_Data.csv**: Alternative preprocessed format

---

## Machine Learning Models

Five trained classification models using Random Forest:

| Model | Target | Key Features | Output |
|-------|--------|--------------|--------|
| `predict_boundary_model.joblib` | Boundary prediction | Innings, over, pitch, shot type, player styles | Boundary (Yes/No) |
| `predict_good_ball_model_random_forest.joblib` | Restrictive delivery | Same as above | Good Ball (Yes/No) |
| `pitch_line_model.joblib` | Pitch line classification | Delivery parameters | Line category |
| `pitch_length_model.joblib` | Pitch length classification | Delivery parameters | Length category |
| `model_balltype.joblib` | Ball type classification | Bowler attributes, conditions | Ball type category |

**Architecture**: Random Forest with balanced class weights for handling imbalanced cricket data

---

## Notebooks & Analysis

### Data Preparation
- **data_exploration_and_processing.ipynb**: Initial data exploration, quality checks, summary statistics
- **data_processing_1.ipynb**: Data integration, feature creation, dataset validation

### Batsman Analytics
- **batsman_analysis.ipynb**: Core batting metrics, similarity matrices, performance trends
- **core_batsman_analysis.ipynb**: Adaptability analysis, strategy recommendations, field placement logic
- **batsman_analysis_engine.ipynb**: Comprehensive recommendation engine, player comparisons

### Bowler Analytics
- **core_bowling_analysis.ipynb**: Bowler statistics, strength metrics, success index calculation

### Strategic Analysis
- **phase_wise_analysis.ipynb**: Strike rate trends, phase adaptability, role identification
- **predict_boundary_and_batsman_analysis_engine.ipynb**: Model training, feature engineering, predictions

### Dashboard
- **plotly_dashboard.ipynb**: Interactive Plotly dashboard for visualization and exploration

---

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (for full dataset processing)
- **Disk Space**: 2GB for data + models
- **Processor**: Any modern CPU (training takes 5-30 minutes)

### 1. Install Dependencies

```bash
cd "C:\Users\yaswa\OneDrive\Desktop\projects\artificial intelligence project"
pip install -r src/requirements.txt
```

**Required packages**:
- pandas, numpy - Data manipulation
- matplotlib, seaborn - Visualization
- scikit-learn - Machine learning
- plotly - Interactive charts
- joblib - Model serialization

### 2. Data Verification
Confirm all CSV files exist in `data/` and `data/csv/` directories

### 3. Running Notebooks

```bash
jupyter notebook
# Open any .ipynb file and run cells sequentially
```

**Note**: Update hardcoded file paths in notebooks to match your system if needed.

---

## Usage Guide

### Analyze Batsman Performance
```python
# Notebook: notebooks/batsman_analysis.ipynb
# Generates: Batsman profiles, metrics, similarity matrices
```

### Analyze Bowler Effectiveness
```python
# Notebook: notebooks/core_bowling_analysis.ipynb
# Generates: bowler_stats.csv, strength categorization, success index
```

### Train Predictive Models
```python
# Notebook: notebooks/predict_boundary_and_batsman_analysis_engine.ipynb
# Creates: Trained Random Forest models, feature datasets
```

### Phase-wise Performance
```python
# Notebook: notebooks/phase_wise_analysis.ipynb
# Analyzes: Powerplay, Middle, Death phase performance
```

### Use Trained Models
```python
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model
model = joblib.load('models/predict_boundary_model.joblib')

# Prepare features
features = pd.DataFrame({
    'inningNumber': [1],
    'oversActual': [5.2],
    'pitchLine': [1],
    'pitchLength': [2],
    'shotType': [1],
    'time_of_day': [1],
    'Batsman_Batting_Style': [1],
    'Bowler_Bowling_Style': [1],
    'isDeathOver': [0],
    'line_x_length': [2]
})

# Predict
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)
prediction = model.predict(X_scaled)
probability = model.predict_proba(X_scaled)

print(f"Boundary Probability: {probability[0][1]:.2%}")
```

---

## Core Metrics

### Batsman Metrics
- **Strike Rate**: Runs per 100 balls
- **Boundary %**: Percentage of deliveries with 4/6
- **Dismissal Rate**: Balls faced per wicket
- **Adaptability Index**: Consistency across opponents (0-1 scale)
- **Phase Performance**: Metrics per match phase

### Bowler Metrics
- **Economy Rate**: Runs conceded per over
- **Strike Rate**: Balls per wicket
- **Wicket %**: Wickets per delivery
- **Dot Ball %**: Deliveries with 0 runs
- **Boundary %**: Deliveries with 4/6
- **Success Index**: Composite metric (30% economy + 30% dot balls + 40% wickets)
- **Strength Zone**: Elite/Strong/Average/Weak categorization

### Model Performance
- **Accuracy**: Prediction correctness percentage
- **ROC-AUC**: Model discrimination ability (0-1 scale)
- **Precision/Recall**: Per-class performance metrics
- **Confusion Matrix**: Prediction breakdown

---

## Data Flow

```
Raw Data (data/csv/)
    ↓
Data Exploration & Processing
    ↓
final_processed_data.csv (Main dataset)
    ↓
    ├─→ Batsman Analysis → batsman_profiles.csv
    ├─→ Bowling Analysis → bowler_stats.csv
    ├─→ Phase Analysis → phase_sr.csv
    ├─→ Model Training → models/ & feature datasets/
    └─→ Dashboard & Visualizations
```

---

## Key Analyses

### 1. Batsman Adaptability
Measures how effectively a batsman adjusts to different bowling styles and match situations.

**Calculation**:
- Analyzes performance against different bowler types
- Computes variance in strike rate
- Generates adaptability index (1 / (1 + std_deviation))

### 2. Phase-wise Performance
Evaluates performance across three match phases:
- **Powerplay (Overs 1-6)**: Aggressive boundary-hitting focus
- **Middle (Overs 7-15)**: Building partnerships, maintaining momentum
- **Death (Overs 16-20)**: Maximum run rate with wicket preservation

**Metrics**: Strike rate, boundary %, dot ball %, adaptability score per phase

### 3. Bowler Success Index
Weighted composite metric combining:
```
Success Index = (100 - Economy) × 0.3 + Dot Ball % × 0.3 + Wicket % × 0.4
```
This balances economy rate, restrictive bowling, and wicket-taking ability.

### 4. Field Placement Strategy
Recommends fielder positions based on:
- **Match Phase**: Powerplay formations vs Death overs
- **Pitch Conditions**: Short vs Full deliveries
- **Bowler Type**: Fast bowlers vs Spinners
- **Batsman Profile**: Left-handed vs Right-handed preferences

---

## Model Training Pipeline

### Training Process
1. Load preprocessed data
2. Feature engineering (9-10 features per model)
3. Train-test split (80-20 with stratification)
4. StandardScaler normalization
5. RandomForestClassifier with balanced class weights
6. Evaluate: Accuracy, ROC-AUC, Classification Report
7. Export model to joblib format

### Features Used
- `inningNumber`, `oversActual`: Temporal context
- `pitchLine`, `pitchLength`: Pitch characteristics
- `shotType`: Batting approach
- `time_of_day`: Dusk/Night/Day conditions
- `Batsman_Batting_Style`, `Bowler_Bowling_Style`: Player attributes
- `isDeathOver`: Phase indicator (overs 16+)
- `line_x_length`: Interaction feature

### Hyperparameters
- Estimators: 200
- Random State: 42 (reproducibility)
- Class Weight: Balanced (handles imbalance)
- Test Size: 20%

---

## File Organization

### To Understand Project Flow:
1. Start: `README.md` (this file)
2. Reference: `Cricket strategy Recommendation System.pdf`
3. Explore: `notebooks/data_exploration_and_processing.ipynb`

### To Analyze Batsmen:
1. `notebooks/core_batsman_analysis.ipynb`
2. `notebooks/phase_wise_analysis.ipynb`
3. Output: `data/batsman_profiles.csv`

### To Analyze Bowlers:
1. `notebooks/core_bowling_analysis.ipynb`
2. Output: `data/bowler_stats.csv`

### To Build Predictions:
1. `notebooks/predict_boundary_and_batsman_analysis_engine.ipynb`
2. Models: `models/*.joblib`
3. Features: `model_feature_engineered_datasets/*.csv`

### To Explore Data:
1. Raw: `data/csv/*.csv`
2. Processed: `data/final_processed_data.csv`
3. Aggregated: `data/batsman_*.csv`, `data/bowler_*.csv`

---

## Troubleshooting

### Import Errors
```
Solution: pip install -r src/requirements.txt
```

### File Path Issues
```
Solution: Update paths in notebooks to match your system:
Change: path = r'C:\Users\...\data\...'
To: Your actual path
```

### Model Loading Errors
```
Solution: Verify all files in models/ exist
Option: Retrain models using predict_boundary_and_batsman_analysis_engine.ipynb
```

### Feature Mismatch
```
Solution: Ensure feature names match model expectations
Run feature engineering cells before predictions
```

## Performance Considerations

- Dataset size: Thousands of ball-level records
- Processing time: Full pipeline runs in 10-30 minutes
- Model training: 5-15 minutes for 200 estimators
- Prediction latency: <100ms per delivery on modern hardware

---

## Project Statistics

- **Total Notebooks**: 9 analysis and processing workflows
- **Trained Models**: 5 (all Random Forest)
- **Features Engineered**: 10+ per model
- **Data Files**: 20+ CSV datasets
- **Metrics Calculated**: 30+ performance indicators

---

## Dependencies

```
pandas==1.5.3          # Data manipulation
numpy==1.24.3          # Numerical computing
matplotlib==3.7.1      # Visualization
seaborn==0.12.2        # Statistical plots
scikit-learn==1.2.2    # Machine learning
plotly==5.15.0         # Interactive charts
streamlit==1.28.0      # Web apps (optional)
joblib==1.3.0          # Model persistence
```

---

## Implementation Notes

- All notebooks use `final_processed_data.csv` as the primary data source
- Models save to `models/` in joblib format for cross-platform compatibility
- Feature datasets export to `model_feature_engineered_datasets/` for reference
- Statistical calculations use scikit-learn's preprocessing and metrics modules
- Visualizations leverage Plotly for interactive exploration

---

## Citation & Attribution

Cricket data sourced from comprehensive match records covering multiple seasons and tournaments. Player statistics aggregated from ball-by-ball delivery data with comprehensive validation.

---

## Contact & Support

For questions or improvements:
1. Review relevant notebook comments
2. Check data path configurations
3. Verify dependency installation
4. Test individual components before full pipeline execution

---

**Last Updated**: March 2026

**Project Status**:
- ✅ Data processing: Complete
- ✅ Batsman analysis engine: Complete
- ✅ Bowler analysis engine: Complete
- ✅ Predictive models: Trained and available
- ✅ Strategic recommendations: Implemented
- ✅ Dashboard: Available

**Scope**: Data Science & Machine Learning system for cricket performance analytics and strategic recommendations.
