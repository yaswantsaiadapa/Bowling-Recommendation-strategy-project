# Cricket Strategy Recommendation System

## Project Overview

A **Data Science & Machine Learning** system for cricket performance analysis and strategic recommendations. The system ingests ball-by-ball ICC T20 World Cup data, builds rich player profiles, trains predictive models, and generates data-driven bowling strategy and field placement recommendations.

---

## Project Structure

```
cricket-strategy/
├── README.md
├── src/
│   └── requirements.txt
│
├── data/
│   ├── csv/                          ← Raw source data (do not edit)
│   │   ├── ballbyball.csv
│   │   ├── matches.csv
│   │   ├── players.csv
│   │   ├── ground.csv
│   │   ├── team.csv
│   │   ├── season.csv
│   │   ├── country.csv
│   │   └── town.csv
│   │
│   ├── final_processed_data.csv      ← Main cleaned dataset (produced by 01)
│   ├── batsman_profiles.csv          ← Batsman stats (produced by 03)
│   ├── batsman_similarity.csv        ← Cosine similarity matrix (produced by 03)
│   ├── bowler_stats.csv              ← Bowler stats (produced by 05)
│   ├── bowling_success_model.csv     ← Matchup stats (produced by 05)
│   └── phase_sr.csv                  ← Phase-wise stats (produced by 04)
│
├── models/
│   ├── label_encoders.joblib         ← All categorical encoders (produced by 06)
│   ├── predict_boundary_model.joblib ← (produced by 06)
│   └── predict_good_ball_model.joblib← (produced by 06)
│
└── notebooks/                        ← Run in order
    ├── 01_data_processing.ipynb
    ├── 02_data_exploration.ipynb
    ├── 03_batsman_analysis.ipynb
    ├── 04_phase_wise_analysis.ipynb
    ├── 05_bowling_analysis.ipynb
    ├── 06_predictive_models.ipynb
    └── 07_strategy_recommendation.ipynb
```

---

## Run Order

Run notebooks **strictly in numerical order**. Each notebook reads outputs produced by the previous one.

| # | Notebook | Reads | Writes |
|---|----------|-------|--------|
| 01 | `01_data_processing.ipynb` | `data/csv/*.csv` | `data/final_processed_data.csv` |
| 02 | `02_data_exploration.ipynb` | `final_processed_data.csv` | *(visualisations only)* |
| 03 | `03_batsman_analysis.ipynb` | `final_processed_data.csv` | `batsman_profiles.csv`, `batsman_similarity.csv` |
| 04 | `04_phase_wise_analysis.ipynb` | `final_processed_data.csv` | `phase_sr.csv` |
| 05 | `05_bowling_analysis.ipynb` | `final_processed_data.csv` | `bowler_stats.csv`, `bowling_success_model.csv` |
| 06 | `06_predictive_models.ipynb` | `final_processed_data.csv` | `models/*.joblib` |
| 07 | `07_strategy_recommendation.ipynb` | all `data/*.csv` | *(recommendations + visualisations)* |

---

## Setup

```bash
# 1. Install dependencies
pip install -r src/requirements.txt

# 2. Launch Jupyter
jupyter notebook

# 3. Open notebooks/ and run in order 01 → 07
```

**One path to change**: In every notebook, `PROJECT_ROOT` is set to the parent of the notebooks folder:
```python
PROJECT_ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))
```
This means **no hardcoded paths**. As long as you open notebooks from the `notebooks/` directory, paths resolve automatically.

---

## Notebooks in Detail

### 01 — Data Processing
Merges `ballbyball.csv` + `matches.csv` + `players.csv` + `ground.csv` into a single clean ball-by-ball dataset. Corrects pre-delivery running totals. Drops null rows in analytical columns.

### 02 — Data Exploration
15 visualisations covering: delivery outcomes, pitch distributions, line×length heatmaps, shot type analysis, bowler style effectiveness, phase patterns, over-by-over run curves, innings comparison, time-of-day effects.

### 03 — Batsman Analysis
Computes: runs, SR, average, boundary%, dot ball%, dismissal rate, performance vs each bowler style, phase averages, adaptability index, consistency index, cosine similarity matrix.

### 04 — Phase-wise Analysis
Breaks down every batsman's performance into Powerplay / Middle / Death. Computes phase adaptability index. Includes radar charts and heatmaps.

### 05 — Bowling Analysis
Computes: economy, SR, wicket%, dot ball%, boundary%. Derives a success index (30% economy + 30% dot balls + 40% wickets) normalised to 0–1. Strength zones: Elite / Strong / Average / Weak. Phase-wise economy breakdown. Bowler-vs-batsman matchup model.

### 06 — Predictive Models
Trains two Random Forest classifiers:
- **Boundary model** — predicts whether a delivery will yield 4 or 6
- **Good ball model** — predicts whether a delivery will be restrictive (≤1 run, no boundary)

Features: pitch zone, shot type, player styles, match state, pressure index, phase indicators. Evaluation: ROC-AUC, 5-fold CV, confusion matrix, feature importance.

### 07 — Strategy Recommendation Engine
Functions:
- `get_batsman_threat(name, phase)` — threat level and aggressiveness score
- `recommend_bowling_strategy(batsman, phase, bowler_style)` — pitch line, length, tactical plan
- `suggest_field_placement(...)` — 9 fielder positions with LHB adjustment
- `full_matchup_report(...)` — combines all three + historical matchup data
- `find_best_bowler_vs(batsman, squad)` — ranks squad bowlers by success index

---

## Key Metrics

### Batsman
| Metric | Description |
|--------|-------------|
| Strike Rate | Runs per 100 balls |
| Average | Runs per dismissal |
| Boundary % | Boundaries per 100 balls |
| Dot Ball % | Scoreless deliveries per 100 balls |
| Adaptability Index | Consistency across bowling styles (0=rigid, 1=adaptable) |
| Consistency Index | Average / std_dev of run output |

### Bowler
| Metric | Description |
|--------|-------------|
| Economy Rate | Runs per over |
| Bowling SR | Balls per wicket |
| Dot Ball % | Scoreless deliveries per 100 balls |
| Boundary % | Boundaries conceded per 100 balls |
| Wicket % | Wickets per 100 balls |
| Success Index | 30% economy + 30% dot balls + 40% wickets (0–1 normalised) |

### Models
| Model | Target | ROC-AUC Target |
|-------|--------|---------------|
| Boundary Prediction | isBoundary | ~0.70+ |
| Good Ball Prediction | isGoodBall | ~0.75+ |

---

## Data Flow

```
data/csv/*.csv
    ↓  01_data_processing
data/final_processed_data.csv
    ├──→ 03_batsman_analysis  →  batsman_profiles.csv, batsman_similarity.csv
    ├──→ 04_phase_wise        →  phase_sr.csv
    ├──→ 05_bowling           →  bowler_stats.csv, bowling_success_model.csv
    ├──→ 06_predictive_models →  models/*.joblib
    └──→ 07_strategy          →  recommendations + visualisations
         (reads all above outputs)
```

---

## Dependencies

```
pandas>=1.5.3        # Data manipulation
numpy>=1.24.3        # Numerical computing
matplotlib>=3.7.1    # Static plots
seaborn>=0.12.2      # Statistical plots
scikit-learn>=1.2.2  # ML models
plotly>=5.15.0       # Interactive charts
joblib>=1.3.0        # Model persistence
```

---

**Last Updated**: March 2026  
**Status**: Complete end-to-end pipeline
