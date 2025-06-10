# Machine Learning for Time Series (ML4TS)

A comprehensive introduction to time series analysis and forecasting using Python.

## 📊 Overview

This repository provides hands-on tutorials and implementations for time series analysis and forecasting, covering both traditional statistical methods and modern machine learning approaches. Each notebook builds upon previous concepts, creating a complete learning path from basics to advanced topics.

## 🚀 Getting Started

### Prerequisites

- Python >= 3.9, < 3.14
- pip or [uv](https://astral.sh/blog/uv) package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/vortico/time-series.git
cd time-series
```

2. Install dependencies using `uv` (recommended):
```bash
uv sync
```

Or using `pip`:
```bash
pip install -r requirements.txt
```

3. Launch Jupyter notebook:
```bash
jupyter notebook
```

## 📚 Table of Contents

The notebooks are designed to be followed in sequence, each building upon concepts from previous chapters:

| Notebook | Topic | Description |
|----------|-------|-------------|
| [01-introduction.ipynb](notebooks/01-introduction.ipynb) | **Introduction to Time Series** | Fundamental concepts, data structures, and visualization techniques |
| [02-naive-forecasting-models.ipynb](notebooks/02-naive-forecasting-models.ipynb) | **Naive Forecasting Models** | Simple baseline models including naive, seasonal naive, and drift methods |
| [03-basic-transformations.ipynb](notebooks/03-basic-transformations.ipynb) | **Basic Transformations** | Data preprocessing, differencing, and stabilizing variance |
| [04-decomposition-methods.ipynb](notebooks/04-decomposition-methods.ipynb) | **Decomposition Methods** | Trend and seasonal decomposition techniques |
| [05-exponential-smoothing.ipynb](notebooks/05-exponential-smoothing.ipynb) | **Exponential Smoothing** | Simple, double, and triple exponential smoothing methods |
| [06-evaluating-forecasts.ipynb](notebooks/06-evaluating-forecasts.ipynb) | **Evaluating Forecasts** | Metrics, validation techniques, and model comparison |
| [07-arima-family.ipynb](notebooks/07-arima-family.ipynb) | **ARIMA Family Models** | AutoRegressive Integrated Moving Average models and variations |
| [08-forecasting-advanced-topics.ipynb](notebooks/08-forecasting-advanced-topics.ipynb) | **Advanced Forecasting** | Advanced techniques and modern approaches |
| [09-time-series-classification.ipynb](notebooks/09-time-series-classification.ipynb) | **Time Series Classification** | Classification problems with time series data |

## 📁 Repository Structure

```
ml4ts/
├── notebooks/              # Jupyter notebooks with tutorials
│   ├── 01-introduction.ipynb
│   ├── 02-naive-forecasting-models.ipynb
│   ├── ...
│   └── utils.py           # Helper functions and utilities
├── data/                  # Sample datasets
│   ├── google.parquet     # Google stock price data
│   ├── electricity_au.parquet  # Australian electricity demand
│   ├── energy_demand.parquet   # Spanish energy demand data
│   └── chemical_process.parquet # Chemical process measurements
├── img/                   # Images and diagrams
└── pyproject.toml        # Project configuration and dependencies
```

## 📊 Datasets

The repository includes several real-world datasets for hands-on practice:

- **Google Stock Data** (`google.parquet`, `GOOGL.csv`): Historical stock prices for Google/Alphabet
- **Australian Electricity** (`electricity_au.parquet`, `electricity_au_month.parquet`): Electricity demand data from Australia
- **Energy Demand** (`energy_demand.parquet`): Energy demand patterns from Spain
- **Chemical Process** (`chemical_process.parquet`): Industrial process measurements

## 🛠️ Key Dependencies

- **Core Libraries**: `numpy`, `matplotlib`
- **Time Series Specific**: `statsmodels`, `sktime`, `pmdarima`, `skforecast`
- **Data Handling**: `pyarrow` for efficient data loading
- **Environment**: `jupyter notebook` for interactive learning

## 🎯 Learning Path

1. **Start with the basics** - Begin with notebook 01 to understand time series fundamentals
2. **Follow the sequence** - Each notebook builds upon previous concepts
3. **Practice with data** - Use the provided datasets to experiment with different techniques
4. **Experiment** - Modify the code and try different parameters to deepen understanding

## 🤝 Contributing

This repository is part of Vortico's educational materials. For questions or suggestions, please refer to the [GitHub repository](https://github.com/vortico/time-series).

## 📄 License

This project is maintained by [Vortico](https://vortico.tech) for educational purposes.
