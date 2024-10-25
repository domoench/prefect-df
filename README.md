[WIP] An automated ML pipeline to forecast hourly electricity demand for the PJM balancing authority.

<img
    src="https://github.com/user-attachments/assets/efd0d3b8-3d4c-4d50-a93b-0009093ca0e9"
    alt="Predicted vs. true electricity demand timeseries"
    width="400">
# Goal

The goal of this project is to demonstrate:

- ML Ops
    - [X] ML workflow orchestration.
    - [X] Versioning:
        - [X] Dataset versioning.
        - [X] Model versioning and experiment tracking.
    - [X] ETL Pipeline.
    - [ ] Reliability:
        - [ ] Pipeline performance visibility and alerting. (WIP)
        - [ ] Automated unit tests (WIP)
    - [X] Isolation between development and deployed/production environment infrastructure.
    - [X] Performance comparisons between (model, version)s and a non-ML baseline.
    - [X] Hyperparameter tuning.
    - [ ] Online prediction service
- ML
    - [X] Timeseries feature engineering.
    - [X] Timeseries forecasting with XGBoost.
    - [X] Timeseries cross validation.

# Data

- Electricity demand timeseries: [EIA Open Data](https://www.eia.gov/developer)'s hourly electricity
demand data.
- Weather data: [Open-meteo](https://open-Meteo.com/)
- Holiday calendar: [Calendarific](https://calendarific.com/api-documentation)

# Stack

- Dev Env: [Docker Compose](https://docs.docker.com/compose/)
- ML Workflow Orchestration: [Prefect](https://www.prefect.io/)
- Experiment tracking: [mlflow](https://mlflow.org/)
- Model registry: [mlflow](https://mlflow.org/)
- Dataset version tracking and 'Data Warehouse': [DVC](https://dvc.org/) (+ git repo)
- Runtime data validation: [Great Expectations](https://greatexpectations.io/)
- Model: [XGBoost](https://xgboost.readthedocs.io/en/stable/python/python_intro.html)
- Prod Env: [AWS Copilot](https://aws.amazon.com/containers/copilot/)-managed containers on Fargate.

# Development

Prefect Flow deployment:

```
 prefect deploy --name DEPLOYMENT_NAME --prefect-file flows/deployments/FLOW_DEPLOYMENT_CONFIG.yaml
```

# References

- Géron, Aurélien. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow. "O'Reilly Media, Inc.", 2022.
- [Rob Mulla's Kaggle tutorial](https://www.kaggle.com/code/robikscube/pt2-time-series-forecasting-with-xgboost/notebook) on timeseries forecasting with XGBoost
- [Prefect ECS Workers](https://docs.prefect.io/integrations/prefect-aws/ecs_guide)
