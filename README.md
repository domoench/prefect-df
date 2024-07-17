A small ML pipeline to produce a model that forecasts electricity demand for the
PJM balancing authority.

# Goal

The goal of this project is to demonstrate:
- ML Ops
    - ML workflow orchestration.
    - Dataset versioning.
    - Model versioning and experiment tracking.
    - Isolation between deveopment and deployed environment infrastructure.
- ML
    - Timeseries forecasting with XGBoost

# Data

[EIA Open Data](https://www.eia.gov/developer)'s hourly electricity
demand data.

# Stack

Stack
- [X] Prefect (ML Workflow Orchestration)
- [ ] MLFlow
- [ ] DVC
- [X] XGBoost

# References

- [Rob Mulla's Kaggle tutorial](https://www.kaggle.com/code/robikscube/pt2-time-series-forecasting-with-xgboost/notebook) on timeseries forecasting with XGBoost
