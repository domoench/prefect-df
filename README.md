An automated ML pipeline to forecast electricity demand for the PJM balancing authority.

# Goal

The goal of this project is to demonstrate:
- ML Ops
    - ML workflow orchestration.
    - Versioning
        - Dataset versioning.
        - Model versioning and experiment tracking.
    - Isolation between development and deployed environment infrastructure.
    - Model performance comparisons to baseline
- ML
    - Timeseries feature engineering
    - Timeseries forecasting with XGBoost

# Data

[EIA Open Data](https://www.eia.gov/developer)'s hourly electricity
demand data.

# Stack

Stack
- [X] Prefect (ML Workflow Orchestration)
- [ ] MLFlow
- [X] DVC
- [X] XGBoost

# References

- [Rob Mulla's Kaggle tutorial](https://www.kaggle.com/code/robikscube/pt2-time-series-forecasting-with-xgboost/notebook) on timeseries forecasting with XGBoost
