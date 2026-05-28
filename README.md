# SplitBrainDetector

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/django-4.2-green.svg" alt="Django 4.2"/>
  <img src="https://img.shields.io/badge/catboost-1.2-yellow.svg" alt="CatBoost"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.3-orange.svg" alt="scikit-learn"/>
  <img src="https://img.shields.io/badge/license-MIT-lightgrey.svg" alt="MIT License"/>
  <img src="https://img.shields.io/badge/status-research-blueviolet.svg" alt="Research"/>
</p>

> **Binary classification system for assessing microservice cluster configurations by split brain probability using machine learning methods.**

SplitBrainDetector is a research-grade machine learning system that evaluates microservice cluster topologies for their susceptibility to the split brain problem (network partition that results in two or more isolated subclusters operating independently). It combines six trained ML models — three base classifiers and three ensemble architectures — with isotonic probability calibration and an interactive web interface for cluster topology design.

---

## Table of Contents

- [Motivation](#motivation)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [System Architecture](#system-architecture)
- [Models](#models)
- [Experimental Results](#experimental-results)
- [Web Interface](#web-interface)
- [Reproducing Experiments](#reproducing-experiments)
- [API Usage](#api-usage)
- [Limitations](#limitations)
- [Citation](#citation)
- [License](#license)
- [Authors](#authors)

---

## Motivation

Modern distributed systems are typically built around microservice architectures, which provide horizontal scalability, fault tolerance, and technology independence. However, microservice clusters are vulnerable to the **split brain problem** — a critical state in which network partitions cause the cluster to fragment into isolated subclusters, each independently continuing transaction processing. The consequences include data inconsistency, conflict resolution failures upon recovery, and, in severe cases, complete system unavailability.

Existing algorithmic approaches (Raft consensus, CRDTs, quorum systems, leader-based replication) operate at the protocol level and react to partitions **after** they occur. None of them allow quantitative assessment of a specific topology's structural fragility **before** deployment.

**SplitBrainDetector fills this gap** by providing a tool for distributed systems architects to compare alternative topologies by their split brain susceptibility during the design phase.

In the Ukrainian context — where military threats and regular damage to energy infrastructure cause unpredictable network disruptions between geographically distributed data centers — early structural reliability assessment has particular practical value.

---

## Key Features

-  **Six trained ML models** — three base classifiers (CatBoost, Gradient Boosting, Random Forest) and three ensemble architectures (E1_Mixed, E2_CB_Bias, E3_CB_Hyper) with different diversification strategies.
-  **Isotonic probability calibration** — converts raw model outputs into well-calibrated probabilities; Expected Calibration Error reduced by a factor of 213× for CatBoost (0.6401 → 0.0030).
-  **Three threshold selection methods** — Youden's J (balanced), MaxF1 (precision-recall balance), High-Recall (≥80% recall for safety-critical scenarios).
-  **Interactive web interface** — Django-based graphical editor on HTML5 Canvas for building arbitrary cluster topologies and getting real-time predictions from all six models.
-  **Fully reproducible experiments** — fixed random seeds, synthetic data generation, and a complete pipeline from data generation through model evaluation.
-  **Production-ready serialization** — all models persist as `.pkl` files with paired JSON threshold metadata.

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip
- ~5 GB of free disk space (for trained model artifacts)
- Optional: GPU with CUDA support for faster CatBoost training

### Installation

```bash
# Clone the repository
git clone https://github.com/PSnik-Kostiantyn/SplitBrainDetector.git
cd SplitBrainDetector

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Web Interface

```bash
python manage.py runserver
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.

### Train Models from Scratch (Optional)

Pre-trained models are included in the repository. If you wish to retrain:

```bash
# Rename or delete existing .pkl files in models/ directory
# Then trigger training via the web interface or run:
python train.py
```

> **Note:** Full training of all six models takes approximately 4–5 hours on a 16-core CPU. Pre-trained models are recommended for evaluation purposes.

---

## Project Structure

```
SplitBrainDetector/
├── manage.py                      # Django entry point
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── SplitBrainDetector/            # Django project settings
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── core/                          # Main application
│   ├── views.py                   # HTTP request handlers
│   ├── urls.py                    # URL routing
│   ├── ml/                        # ML pipeline
│   │   ├── data_generator.py      # Synthetic data generation (Erdős–Rényi)
│   │   ├── preprocess.py          # Matrix → 240-dim vector conversion
│   │   ├── isClusterDead.py       # DFS-based label functions
│   │   ├── ensemble_model.py      # EnsembleModel class (see Appendix A of thesis)
│   │   ├── calibration.py         # Isotonic calibration
│   │   ├── thresholds.py          # Youden / MaxF1 / High-Recall
│   │   └── train.py               # Full training pipeline
│   ├── templates/                 # HTML templates
│   │   ├── home.html
│   │   ├── graphic.html           # Main interactive editor
│   │   ├── train.html
│   │   └── info.html
│   └── static/                    # CSS, JS, images
├── models/                        # Trained model artifacts
│   ├── catboost.pkl
│   ├── gradient_boosting.pkl
│   ├── random_forest.pkl
│   ├── e1_mixed.pkl
│   ├── e2_cb_bias.pkl
│   ├── e3_cb_hyper.pkl
│   └── thresholds/                # JSON files with optimal thresholds
│       ├── catboost_thresholds.json
│       └── ...
└── docs/                          # Additional documentation
    └── methodology.md
```

---

## Models

### Base Models

| Model | Library | Trees | Depth | Training Size |
|-------|---------|-------|-------|---------------|
| **CatBoost** | catboost 1.2 | up to 3000 (early stop) | 7 | 1.1 M samples |
| **Gradient Boosting** | scikit-learn | up to 500 (early stop) | 9 | 1.1 M samples |
| **Random Forest** | scikit-learn | 600 | 15 | 500 K samples |

All base models use a unified `NATURAL_SPW ≈ 10.1` constant for class balancing, computed from the model class distribution (~9% positive rate).

### Ensemble Architectures

| Ensemble | Components | Diversification Strategy |
|----------|------------|-------------------------|
| **E1_Mixed** | Random Forest + Gradient Boosting + CatBoost | Different algorithms (different inductive biases) |
| **E2_CB_Bias** | CatBoost @10%, CatBoost @50%, CatBoost @90% | Different training set positive class enrichment |
| **E3_CB_Hyper** | CatBoost-shallow + CatBoost-deep + CatBoost-balanced | Different hyperparameters (depth, learning rate, regularization) |

All ensembles use **soft voting** with weighted averaging of calibrated probabilities. Implementation details: see `core/ml/ensemble_model.py`.

---

## Experimental Results

All metrics are computed on a held-out test set of 20,000 samples (model class distribution: 9.02% positive).

### Before vs. After Calibration

| Model | ECE (before) | ECE (after) | Improvement |
|-------|--------------|-------------|-------------|
| CatBoost | 0.6401 | **0.0030** | **213×** |
| Gradient Boosting | 0.6482 | 0.0052 | 125× |
| Random Forest | 0.3484 | 0.0035 | 99× |
| E1_Mixed | 0.6346 | 0.0136 | 47× |
| E2_CB_Bias | 0.3334 | 0.0084 | 40× |
| E3_CB_Hyper | 0.7619 | 0.0057 | 134× |

### Performance Comparison (after calibration, Youden threshold)

| Model | ROC-AUC | PR-AUC | F1 | Youden's J | Threshold τ |
|-------|---------|--------|-----|------------|-------------|
| **CatBoost**  | **0.7961** | **0.3588** | 0.2925 | **0.4287** | 0.0957 |
| Gradient Boosting | 0.7750 | 0.3341 | 0.2900 | 0.3860 | 0.0955 |
| Random Forest | 0.6611 | 0.2191 | 0.2163 | 0.2385 | 0.0899 |
| E1_Mixed | 0.7662 | 0.3345 | 0.2833 | 0.3863 | 0.1034 |
| E2_CB_Bias | 0.7670 | 0.3527 | 0.2797 | 0.3843 | 0.1031 |
| **E3_CB_Hyper**  | 0.7849 | 0.3503 | **0.3012** | 0.4100 | 0.1137 |

>  **Key methodological finding:** In this configuration, no ensemble outperforms the strongest base model (CatBoost). This is consistent with the fact that ensemble averaging only provides gains when component models have comparable quality and different error sources. The dominance of CatBoost reflects the specific structure of this task rather than a failure of ensemble methods in general.

---

## Web Interface

The web interface includes four pages:

- **Home** — project overview and getting-started instructions.
- **Graphic Editor** (`/graphic/`) — interactive cluster topology builder with real-time predictions.
- **Train** (`/train/`) — UI for retraining models with custom calibration data.
- **Info** — detailed background on the split brain problem and methodology.

### How to Use the Graphic Editor

1. **Add nodes**: enter a node name in the format `XY` (e.g., `A1`, `B2`, `C3`) where `X ∈ {A, B, C}` is the microservice type and `Y` is the index.
2. **Move nodes**: drag with the left mouse button to position them on the canvas.
3. **Create connections**: select either "Bidirectional" or "One-way" mode and click on two nodes in sequence.
4. **Submit**: click "Submit" to send the topology to the server and receive predictions from all six models.

A live adjacency matrix preview updates automatically as you build the topology.

---

## Reproducing Experiments

All experiments use `random_seed = 42` to ensure full reproducibility.

### Training Pipeline

```bash
# Generate synthetic training data
python -m core.ml.data_generator --seed 42 --output data/

# Train base models
python -m core.ml.train --model catboost
python -m core.ml.train --model gradient_boosting
python -m core.ml.train --model random_forest

# Train ensembles
python -m core.ml.train --model e1_mixed
python -m core.ml.train --model e2_cb_bias
python -m core.ml.train --model e3_cb_hyper

# Calibrate all models
python -m core.ml.calibration --calibrate-all

# Search optimal thresholds (Youden, MaxF1, High-Recall)
python -m core.ml.thresholds --search-all
```

### Data Splits

| Split | Size | Class Distribution | Purpose |
|-------|------|--------------------|---------|
| Training | 1.1 M | enriched (35–40% positive) | Model fitting |
| Validation | 10–20 K | natural (~9% positive) | Early stopping |
| Calibration | 8 K | natural (~9% positive) | Isotonic calibration |
| Test | 20 K | natural (~9% positive) | Final evaluation |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- Igor Sikorsky Kyiv Polytechnic Institute (KPI), Department of Software Engineering of Intelligent Cyber-Physical Systems in Energy.
- G.E. Pukhov Institute for Modelling in Energy Engineering, National Academy of Sciences of Ukraine.
- Research presented at the *Cyber Security of Energy* conference (28 May 2025, Kyiv).

