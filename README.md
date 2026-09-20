# Wellness Tourism Package: Purchase Prediction with MLOps

An automated pipeline that predicts whether a customer will buy the **Wellness Tourism Package** *before* the sales team contacts them, for the travel company "Visit with Us".
Every push to `main` re-registers the data, prepares it, tunes an XGBoost model, tracks every run in MLflow, commits the best model back to the repo, and the Streamlit app serves it.

**Live app:** https://wellness-tourism-purchase-prediction.streamlit.app/

## Business problem

Choosing which customers to call by hand is inconsistent, slow and error-prone, so sales time goes to people who will not buy.
This project ranks customers by their likelihood to purchase, so outreach can start with the most promising ones, and keeps the model up to date automatically through CI/CD.

## Results (held-out test set, 826 customers)

| Metric | Value |
|---|---|
| ROC-AUC | 0.92 |
| Recall (buyers) | 0.79 |
| Precision (buyers) | 0.59 |
| F1 (buyers) | 0.67 |
| Accuracy | 0.85 |

At a decision threshold of 0.45, calling the top 26% of customers reaches about 79% of actual buyers, and roughly 59% of those calls convert versus 19% when calling everyone.
The model is tuned for recall, because a missed buyer costs more than an extra call. Train recall is higher than test recall (0.96 vs 0.79), so some overfitting is present and the test figures are the honest estimate.

## Dataset

`tourism_project/data/tourism.csv`: 4,128 customers, 18 usable features (customer details and sales-interaction data) and the target `ProdTaken` (1 = bought the package, 19.3% of customers).

Cleaning applied in `prep.py`:
- Drop `CustomerID` (identifier) and `Unnamed: 0` (leftover row index).
- Fix the label `Fe Male` to `Female` (155 rows).
- Stratified 80/20 train-test split on `ProdTaken` (`random_state=42`): 3,302 train and 826 test rows.

## Repository structure

```
.github/workflows/pipeline.yml      CI/CD workflow (3 jobs)
tourism_project/
  data/tourism.csv                  raw dataset
  model_building/
    data_register.py                validates and registers the dataset
    prep.py                         cleaning and train/test split
    train.py                        XGBoost grid search, MLflow tracking, saves the model
  deployment/
    app.py                          Streamlit app
    requirements.txt                app dependencies
    best_tourism_package_model_v1.joblib   trained model (committed by the pipeline)
  requirements.txt                  pipeline dependencies
```

## The pipeline

Defined in `.github/workflows/pipeline.yml`. It runs on every push to `main` and can also be started manually (`workflow_dispatch`).

| Job | What it does | Output |
|---|---|---|
| `register-dataset` | Installs dependencies, runs `data_register.py` (checks the expected columns and reports the class balance) | `registered-data` artifact |
| `data-prep` | Runs `prep.py` | `data-splits` artifact (`Xtrain`, `Xtest`, `ytrain`, `ytest`) |
| `model-training` | Downloads the splits, starts an MLflow server, runs `train.py`, then commits the model to `main` with `[skip ci]` so it does not trigger another run | `best_tourism_package_model_v1.joblib` in `tourism_project/deployment/` |

## Model and experiment tracking

- Pipeline: `StandardScaler` on 12 numeric features and `OneHotEncoder` on 6 categorical features, followed by `XGBClassifier` with `scale_pos_weight` set to the non-buyer/buyer ratio (about 4.2).
- `GridSearchCV`: 144 parameter combinations, 5-fold cross-validation, scored on recall.
- Every combination is logged to MLflow as a nested run (parameters, mean and std CV recall). The parent run logs the best parameters plus train and test accuracy, precision, recall and F1.
- Best parameters: `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`, `colsample_bytree=0.5`, `colsample_bylevel=0.5`, `reg_lambda=1` (mean CV recall 0.771).

## Deployment

The Streamlit app (`tourism_project/deployment/app.py`) collects the 18 customer and pitch inputs, builds a one-row DataFrame, and applies the 0.45 threshold to the model's purchase probability.
It is hosted on Streamlit Community Cloud (Python 3.11) straight from this repository, reading the model the pipeline committed.

## Run it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r tourism_project/requirements.txt

python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py

mlflow ui --port 5000              # in a second terminal
python tourism_project/model_building/train.py
```

Open http://localhost:5000 to browse the experiment `tourism-package-prediction`. To run the app:

```bash
pip install -r tourism_project/deployment/requirements.txt
streamlit run tourism_project/deployment/app.py
```

## Limitations

- One 80/20 split on about 4,000 rows, so results are directional and should be confirmed on live campaigns.
- Relationships in the data (for example more follow-ups going with higher conversion) are associations, not proven causes; test them with A/B experiments.
- 117 rows are identical apart from `CustomerID`; they were kept and may flatter test scores slightly.
- `ProductPitched` and `Designation` map one-to-one, so their feature importance is shared.
