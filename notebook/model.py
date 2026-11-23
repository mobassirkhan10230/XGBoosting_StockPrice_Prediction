import numpy as np
import joblib
import pandas as pd
import os

from xgboost import XGBClassifier, XGBRegressor

from sklearn.model_selection import (
    train_test_split,
    TimeSeriesSplit,
    RandomizedSearchCV
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

def train_direction_model(X, y, horizon=1):

    # 1. Initialize XGBoost Classifier
    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=1
    )

    # 2. Hyperparameter search space
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    # 3. Chronological train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        shuffle=False
    )

    # 4. Time series cross-validation
    tscv = TimeSeriesSplit(
        n_splits=5,
        gap=horizon
    )

    # 5. Randomized Search
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        cv=tscv,
        scoring="accuracy",
        random_state=42,
        verbose=1,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    print("\nBest Hyperparameters:")
    print(search.best_params_)

    # 6. Test prediction
    y_pred = best_model.predict(X_test)

    # 7. Classification evaluation
    print("\nXGBoost Classification Evaluation")
    print("---------------------------------")

    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred, zero_division=0):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 8. Retrain using all available training data
    best_model.fit(X, y)

    return best_model


def train_price_model(X, y, horizon=1):

    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=42,
        n_jobs=1
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0]
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        shuffle=False
    )

    tscv = TimeSeriesSplit(
        n_splits=5,
        gap=horizon
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=10,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        random_state=42,
        verbose=1,
        n_jobs=-1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    print("\nBest Hyperparameters:")
    print(search.best_params_)
    print("\nBest CV RMSE:")
    print(-search.best_score_)

    print("\nXGBoost Regression Evaluation")
    print("-----------------------------")
    print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"MSE  : {mean_squared_error(y_test, y_pred):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"R2   : {r2_score(y_test, y_pred):.4f}")

    best_model.fit(X, y)
    return best_model


def model_assignment (X,Y):

    # 1. Copy independent and dependent variables
    x = X.copy()
    y = Y.copy()

    # 2. Clean column names
    x.columns = x.columns.astype(str).str.strip()

    # 3. Remove Date column if present
    date_columns = [
        col for col in x.columns
        if col.lower() in ["date", "datetime", "timestamp"]
    ]

    x = x.drop(columns=date_columns)

    # 4. Convert features into numeric values
    x = x.apply(pd.to_numeric, errors="coerce")

    # 5. Convert target into numeric values
    y = pd.to_numeric(y, errors="coerce")

    # 6. Replace infinity with NaN
    x = x.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    # 7. Remove invalid rows
    valid_rows = x.notna().all(axis=1) & y.notna()

    print("Invalid rows:", (~valid_rows).sum())
    print("Valid rows:", valid_rows.sum())

    x = x.loc[valid_rows].copy()
    y = y.loc[valid_rows].copy()

    if x.empty:
        raise ValueError("No valid training data remaining.")

    print("\nFeature columns:")
    print(x.columns.tolist())

    print("\nFeature data types:")
    print(x.dtypes)

    print("\nTarget data type:")
    print(y.dtype)


    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        shuffle=False
    )

     # 2. Initialize XGBoost Regressor model
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=42,
        n_jobs=1
    )

    # 4. Hyperparameter search space
    param_grid = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "min_child_weight": [1, 3, 5, 7],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "gamma": [0, 0.1, 0.3, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [1, 2, 5, 10]
    }

    # 5. Time Series Cross Validation
    tscv = TimeSeriesSplit(
        n_splits=5,
        gap=5
    )

    # 6. Randomized Search CV
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_grid,
        n_iter=30,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    
    
    
    
    # 7. Best model
    best_model = random_search.best_estimator_

    print("\nBest Hyperparameters:")
    print(random_search.best_params_)

    print("\nBest CV RMSE:")
    print(-random_search.best_score_)

    # 8. Prediction
    y_pred = best_model.predict(X_test)

    # 9. Model evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)


    print("\nXGBoost Model Evaluation")
    print("------------------------")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R2   : {r2:.4f}")

    # 10. Save the model
    os.makedirs("models", exist_ok=True)

    joblib.dump(
        best_model,
        "models/best_xgboost_model.pkl"
    )

    print("\nBest XGBoost model saved successfully!")

    return best_model, y_pred

    
