# ============================================================
# RETAIL DEMAND FORECASTING SYSTEM
# FORECASTING SERVICE
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# PREPARE FORECASTING DATA
# ============================================================

def prepare_forecasting_data(
    df,
    date_column,
    target_column
):

    data = df.copy()

    # --------------------------------------------------------
    # CHECK COLUMNS
    # --------------------------------------------------------

    if date_column not in data.columns:

        raise ValueError(
            f"Date column '{date_column}' was not found."
        )

    if target_column not in data.columns:

        raise ValueError(
            f"Target column '{target_column}' was not found."
        )

    # --------------------------------------------------------
    # CONVERT DATE
    # --------------------------------------------------------

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # CONVERT TARGET TO NUMERIC
    # --------------------------------------------------------

    data[target_column] = pd.to_numeric(
        data[target_column],
        errors="coerce"
    )

    # --------------------------------------------------------
    # REMOVE INVALID ROWS
    # --------------------------------------------------------

    data = data.dropna(
        subset=[
            date_column,
            target_column
        ]
    )

    # --------------------------------------------------------
    # SORT BY DATE
    # --------------------------------------------------------

    data = data.sort_values(
        date_column
    )

    # --------------------------------------------------------
    # CREATE DATE FEATURES
    # --------------------------------------------------------

    data["year"] = (
        data[date_column].dt.year
    )

    data["month"] = (
        data[date_column].dt.month
    )

    data["day"] = (
        data[date_column].dt.day
    )

    data["day_of_week"] = (
        data[date_column].dt.dayofweek
    )

    data["day_of_year"] = (
        data[date_column].dt.dayofyear
    )

    # --------------------------------------------------------
    # FEATURE COLUMNS
    # --------------------------------------------------------

    feature_columns = [
        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year"
    ]

    X = data[
        feature_columns
    ]

    y = data[
        target_column
    ]

    return (
        data,
        X,
        y,
        feature_columns
    )


# ============================================================
# RANDOM FOREST
# ============================================================

def train_random_forest(
    X,
    y
):

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X,
        y
    )

    return model


# ============================================================
# GRADIENT BOOSTING
# ============================================================

def train_gradient_boosting(
    X,
    y
):

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(
        X,
        y
    )

    return model


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return {

        "mae": round(
            float(mae),
            2
        ),

        "rmse": round(
            float(rmse),
            2
        ),

        "r2": round(
            float(r2),
            4
        )

    }


# ============================================================
# TRAIN AND COMPARE MODELS
# ============================================================

def train_forecasting_models(
    df,
    date_column,
    target_column
):

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    (
        data,
        X,
        y,
        feature_columns
    ) = prepare_forecasting_data(
        df,
        date_column,
        target_column
    )

    # --------------------------------------------------------
    # CHECK DATA SIZE
    # --------------------------------------------------------

    if len(data) < 10:

        raise ValueError(
            "At least 10 valid rows are required "
            "for model training."
        )

    # --------------------------------------------------------
    # TRAIN / TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    random_forest = train_random_forest(
        X_train,
        y_train
    )

    rf_metrics = evaluate_model(
        random_forest,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # GRADIENT BOOSTING
    # --------------------------------------------------------

    gradient_boosting = train_gradient_boosting(
        X_train,
        y_train
    )

    gb_metrics = evaluate_model(
        gradient_boosting,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # COMPARE MODELS
    # --------------------------------------------------------

    models = {

        "Random Forest": {
            "model": random_forest,
            "metrics": rf_metrics
        },

        "Gradient Boosting": {
            "model": gradient_boosting,
            "metrics": gb_metrics
        }

    }

    # --------------------------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------------------------
    #
    # Higher R² is better.
    #

    best_model_name = max(
        models,
        key=lambda name:
            models[name]["metrics"]["r2"]
    )

    best_model = models[
        best_model_name
    ]["model"]

    best_metrics = models[
        best_model_name
    ]["metrics"]

    return {

        "models": models,

        "best_model_name":
            best_model_name,

        "best_model":
            best_model,

        "best_metrics":
            best_metrics,

        "feature_columns":
            feature_columns,

        "data":
            data

    }


# ============================================================
# GENERATE FUTURE FORECAST
# ============================================================

def generate_forecast(
    model,
    last_date,
    periods=30
):

    future_dates = pd.date_range(

        start=(
            last_date
            + pd.Timedelta(days=1)
        ),

        periods=periods,

        freq="D"

    )

    future = pd.DataFrame({

        "date":
            future_dates

    })

    # --------------------------------------------------------
    # CREATE SAME FEATURES USED DURING TRAINING
    # --------------------------------------------------------

    future["year"] = (
        future["date"].dt.year
    )

    future["month"] = (
        future["date"].dt.month
    )

    future["day"] = (
        future["date"].dt.day
    )

    future["day_of_week"] = (
        future["date"].dt.dayofweek
    )

    future["day_of_year"] = (
        future["date"].dt.dayofyear
    )

    feature_columns = [

        "year",
        "month",
        "day",
        "day_of_week",
        "day_of_year"

    ]

    predictions = model.predict(
        future[
            feature_columns
        ]
    )

    future["forecast"] = (
        predictions
    )

    future["forecast"] = (
        future["forecast"]
        .clip(lower=0)
        .round(2)
    )

    return future