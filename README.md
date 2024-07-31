An automated ML pipeline to forecast electricity demand for the PJM balancing authority.

<img 
    src="https://github.com/user-attachments/assets/efd0d3b8-3d4c-4d50-a93b-0009093ca0e9"
    alt="Predicted vs. true electricity demand timeseries"
    width="400">
# Goal

The goal of this project is to demonstrate:

- ML Ops
    - ML workflow orchestration.
    - Versioning:
        - Dataset versioning.
        - Model versioning and experiment tracking.
    - ETL Pipeline.
    - Reliability: Pipeline performance visibility and alerting.
    - Isolation between development and deployed environment infrastructure.
    - Model performance comparisons to baseline.
    - Hyperparameter tuning.
- ML
    - Timeseries feature engineering.
    - Timeseries forecasting with XGBoost.
    - Timeseries cross validation.

# Data

[EIA Open Data](https://www.eia.gov/developer)'s hourly electricity
demand data.

# Stack

- [X] Prefect (ML Workflow Orchestration)
- [ ] MLFlow
- [X] DVC
- [X] XGBoost

# References

- Géron, Aurélien. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow. "O'Reilly Media, Inc.", 2022.
- [Rob Mulla's Kaggle tutorial](https://www.kaggle.com/code/robikscube/pt2-time-series-forecasting-with-xgboost/notebook) on timeseries forecasting with XGBoost
