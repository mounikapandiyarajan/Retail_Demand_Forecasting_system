# ============================================================
# RETAIL DEMAND FORECASTING SYSTEM
# FLASK BACKEND
# ============================================================

import os
import json
import math
import warnings

import numpy as np
import pandas as pd

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash
)

from flask_login import (
    LoginManager,
    login_user,
    login_required,
    logout_user,
    current_user
)

from werkzeug.security import (
    generate_password_hash,
    check_password_hash
)

from werkzeug.utils import secure_filename

from functools import wraps

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from config import Config
from database.models import db, User

from services.data_cleaner import (
    clean_dataset,
    normalize_column_name,
    automatic_column_mapping
)

from services.forecasting_service import (
    train_forecasting_models,
    generate_forecast as generate_service_forecast
)

warnings.filterwarnings("ignore")

app = Flask(__name__)
app.config.from_object(Config)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "data", "uploads")
RAW_FOLDER = os.path.join(BASE_DIR, "data", "raw")
CLEANED_FOLDER = os.path.join(BASE_DIR, "data", "cleaned")
REPORT_FOLDER = os.path.join(BASE_DIR, "data", "reports")
FORECAST_FOLDER = os.path.join(BASE_DIR, "data", "forecasts")
MODULE_DATA_FOLDER = os.path.join(CLEANED_FOLDER, "module_sources")
MODULE_SOURCE_METADATA_PATH = os.path.join(
    MODULE_DATA_FOLDER,
    "sources.json"
)

SHARED_DATASET_PATH = os.path.join(
    CLEANED_FOLDER, "cleaned_shared_dataset.csv"
)

SHARED_REPORT_PATH = os.path.join(
    REPORT_FOLDER, "cleaning_report.json"
)

FORECAST_RESULT_PATH = os.path.join(
    FORECAST_FOLDER, "forecast_results.csv"
)

FORECAST_REPORT_PATH = os.path.join(
    FORECAST_FOLDER, "forecast_report.json"
)

for folder in [
    UPLOAD_FOLDER,
    RAW_FOLDER,
    CLEANED_FOLDER,
    REPORT_FOLDER,
    FORECAST_FOLDER,
    MODULE_DATA_FOLDER
]:
    os.makedirs(folder, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

ALLOWED_EXTENSIONS = {"csv"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"
login_manager.login_message = "Please login to access this page."


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def role_required(*allowed_roles):
    def decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                flash("Please login to continue.", "warning")
                return redirect(url_for("login"))

            if current_user.role not in allowed_roles:
                flash(
                    "You do not have permission to access this page.",
                    "danger"
                )
                return redirect(url_for("dashboard"))

            return function(*args, **kwargs)

        return wrapper

    return decorator


def read_csv_safely(file_path):
    try:
        return pd.read_csv(file_path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(
            file_path,
            encoding="latin1",
            low_memory=False
        )


def save_cleaning_report(report):
    try:
        with open(
            SHARED_REPORT_PATH,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                report,
                file,
                indent=4,
                default=str
            )
        return True
    except Exception:
        return False


def load_shared_dataset():
    if not os.path.exists(SHARED_DATASET_PATH):
        return None

    try:
        return read_csv_safely(SHARED_DATASET_PATH)
    except Exception:
        return None


def shared_dataset_available():
    return os.path.exists(SHARED_DATASET_PATH)


def get_shared_dataset_info():
    result = {
        "available": False,
        "filename": None,
        "rows": 0,
        "columns": 0,
        "missing_values": 0,
        "duplicate_rows": 0,
        "columns_list": [],
        "mapping": {}
    }

    if not shared_dataset_available():
        return result


    try:
        df = load_shared_dataset()

        if df is None or df.empty:
            return result

        result["available"] = True
        result["filename"] = os.path.basename(SHARED_DATASET_PATH)
        result["rows"] = int(len(df))
        result["columns"] = int(len(df.columns))
        result["missing_values"] = int(df.isnull().sum().sum())
        result["duplicate_rows"] = int(df.duplicated().sum())
        result["columns_list"] = list(df.columns)
        result["mapping"] = automatic_column_mapping(df)

        return result

    except Exception:
        return result


MODULE_NAMES = {
    "forecasting",
    "inventory",
    "analytics",
    "seasonal",
    "alerts",
    "reports"
}


def module_source_path(module_name):
    return os.path.join(
        MODULE_DATA_FOLDER,
        f"{module_name}_independent.csv"
    )


def load_module_source_metadata():
    try:
        with open(
            MODULE_SOURCE_METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_module_source_metadata(metadata):
    with open(
        MODULE_SOURCE_METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=4, default=str)


def get_module_dataset(module_name, source="shared"):
    if source == "independent":
        path = module_source_path(module_name)
        if os.path.exists(path):
            try:
                return read_csv_safely(path), "independent", path
            except Exception:
                pass

    if shared_dataset_available():
        dataset = load_shared_dataset()
        if dataset is not None:
            return dataset, "shared", SHARED_DATASET_PATH

    return None, None, None


def get_module_source_context(module_name):
    shared_info = get_shared_dataset_info()
    independent_path = module_source_path(module_name)
    metadata = load_module_source_metadata().get(module_name, {})
    independent_available = os.path.exists(independent_path)
    selected_source = request.args.get("source", "shared")

    if selected_source == "independent" and not independent_available:
        selected_source = "shared"

    dataset, source, path = get_module_dataset(
        module_name,
        selected_source
    )

    return {
        "shared_dataset_available": shared_info["available"],
        "shared_dataset_filename": shared_info["filename"],
        "shared_dataset_rows": shared_info["rows"],
        "shared_dataset_columns": shared_info["columns"],
        "independent_dataset_available": independent_available,
        "independent_dataset_filename": metadata.get("filename"),
        "selected_dataset_source": source or "shared",
        "selected_dataset_filename": (
            os.path.basename(path) if path else None
        ),
        "selected_dataset": dataset
    }


def find_column(df, candidates):
    normalized = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for candidate in candidates:
        candidate_normalized = normalize_column_name(candidate)

        if candidate_normalized in normalized:
            return normalized[candidate_normalized]

    for normalized_name, original_name in normalized.items():
        for candidate in candidates:
            candidate_normalized = normalize_column_name(candidate)

            if (
                candidate_normalized in normalized_name
                or normalized_name in candidate_normalized
            ):
                return original_name

    return None


def calculate_inventory_statistics(dataset=None):
    result = {
        "inventory_available": False,
        "total_products": None,
        "low_stock": None,
        "overstock": None,
        "reorder_required": None,
        "average_daily_demand": None,
        "lead_time": None,
        "safety_stock": None,
        "reorder_point": None,
        "inventory_message": "No shared dataset has been uploaded yet."
    }

    df = dataset if dataset is not None else load_shared_dataset()

    if df is None:
        return result

    if df.empty:
        result["inventory_message"] = (
            "The shared dataset contains no data."
        )
        return result

    product_column = find_column(
        df,
        [
            "product",
            "product_name",
            "product_id",
            "sku",
            "item",
            "item_id"
        ]
    )

    inventory_column = find_column(
        df,
        [
            "inventory",
            "stock",
            "current_stock",
            "stock_level",
            "stock_quantity",
            "available_stock",
            "quantity_in_stock",
            "on_hand"
        ]
    )

    demand_column = find_column(
        df,
        [
            "demand",
            "quantity",
            "qty",
            "units",
            "units_sold",
            "sales_quantity",
            "quantity_sold",
            "sold_quantity"
        ]
    )

    sales_column = find_column(
        df,
        [
            "sales",
            "sales_amount",
            "revenue",
            "total_sales"
        ]
    )

    lead_time_column = find_column(
        df,
        [
            "lead_time",
            "leadtime",
            "delivery_time",
            "supplier_lead_time",
            "days_to_delivery"
        ]
    )

    reorder_column = find_column(
        df,
        [
            "reorder_level",
            "reorder_point",
            "reorder_threshold",
            "minimum_stock",
            "min_stock",
            "safety_level"
        ]
    )

    safety_stock_column = find_column(
        df,
        ["safety_stock"]
    )

    maximum_stock_column = find_column(
        df,
        [
            "maximum_stock",
            "max_stock",
            "max_stock_level",
            "overstock_threshold"
        ]
    )

    if product_column:
        result["total_products"] = int(
            df[product_column].nunique()
        )
    else:
        result["total_products"] = int(len(df))

    if inventory_column:
        stock = pd.to_numeric(
            df[inventory_column],
            errors="coerce"
        )

        valid_stock = stock.dropna()

        if not valid_stock.empty:
            if reorder_column:
                threshold = pd.to_numeric(
                    df[reorder_column],
                    errors="coerce"
                )

                result["low_stock"] = int(
                    (stock < threshold)
                    .fillna(False)
                    .sum()
                )

                result["reorder_required"] = int(
                    (stock <= threshold)
                    .fillna(False)
                    .sum()
                )

            if maximum_stock_column:
                maximum_stock = pd.to_numeric(
                    df[maximum_stock_column],
                    errors="coerce"
                )

                result["overstock"] = int(
                    (stock > maximum_stock)
                    .fillna(False)
                    .sum()
                )

    actual_demand_column = demand_column or sales_column

    if actual_demand_column:
        demand_values = pd.to_numeric(
            df[actual_demand_column],
            errors="coerce"
        ).dropna()

        if not demand_values.empty:
            result["average_daily_demand"] = round(
                float(demand_values.mean()),
                2
            )

    if lead_time_column:
        lead_values = pd.to_numeric(
            df[lead_time_column],
            errors="coerce"
        ).dropna()

        if not lead_values.empty:
            result["lead_time"] = round(
                float(lead_values.mean()),
                2
            )

    if safety_stock_column:
        safety_values = pd.to_numeric(
            df[safety_stock_column],
            errors="coerce"
        ).dropna()

        if not safety_values.empty:
            result["safety_stock"] = round(
                float(safety_values.mean()),
                2
            )

    average_demand = result["average_daily_demand"]
    lead_time = result["lead_time"]

    if average_demand is not None and lead_time is not None:
        result["reorder_point"] = round(
            average_demand * lead_time,
            2
        )

    result["inventory_available"] = True
    result["inventory_message"] = (
        "Inventory information is calculated "
        "from the shared cleaned dataset."
    )

    return result


# ============================================================
# FORECASTING ENGINE
# ============================================================

FORECAST_HORIZON = 30


def get_forecasting_columns(df):
    mapping = automatic_column_mapping(df)

    date_column = mapping.get("date")
    demand_column = mapping.get("demand")
    sales_column = mapping.get("sales")

    target_column = demand_column or sales_column

    return date_column, target_column, mapping


def prepare_forecasting_data(df):
    date_column, target_column, mapping = (
        get_forecasting_columns(df)
    )

    result = {
        "success": False,
        "message": "",
        "date_column": date_column,
        "target_column": target_column,
        "mapping": mapping,
        "data": None
    }

    if not date_column:
        result["message"] = (
            "No date column was detected for forecasting."
        )
        return result

    if not target_column:
        result["message"] = (
            "No demand or sales column was detected for forecasting."
        )
        return result

    if (
        date_column not in df.columns
        or target_column not in df.columns
    ):
        result["message"] = (
            "Required forecasting columns are not available."
        )
        return result

    data = df[[date_column, target_column]].copy()

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )

    data[target_column] = pd.to_numeric(
        data[target_column],
        errors="coerce"
    )

    data = data.dropna(
        subset=[date_column, target_column]
    )

    if data.empty:
        result["message"] = (
            "No valid date and demand records are available."
        )
        return result

    data = data[data[target_column] >= 0]

    if data.empty:
        result["message"] = (
            "No valid non-negative demand records are available."
        )
        return result

    data = (
        data
        .groupby(
            date_column,
            as_index=False
        )[target_column]
        .sum()
    )

    data = data.sort_values(date_column)
    data = data.reset_index(drop=True)

    if len(data) < 10:
        result["message"] = (
            "At least 10 valid time-series records "
            "are recommended for forecasting."
        )
        return result

    result["success"] = True
    result["message"] = (
        "Forecasting data prepared successfully."
    )
    result["data"] = data

    return result


def create_time_features(dates):
    features = pd.DataFrame(
        index=range(len(dates))
    )

    dates = pd.to_datetime(dates)

    features["year"] = dates.dt.year
    features["month"] = dates.dt.month
    features["day"] = dates.dt.day
    features["day_of_week"] = dates.dt.dayofweek
    features["day_of_year"] = dates.dt.dayofyear

    features["week_of_year"] = (
        dates.dt.isocalendar()
        .week
        .astype(int)
        .values
    )

    features["quarter"] = dates.dt.quarter

    return features


def run_random_forest_forecast(
    time_series,
    horizon=30
):
    date_column = time_series.columns[0]
    target_column = time_series.columns[1]

    dates = time_series[date_column]
    values = time_series[target_column]

    features = create_time_features(dates)

    total_records = len(features)

    test_size = max(
        3,
        int(total_records * 0.2)
    )

    if total_records - test_size < 5:
        test_size = 3

    X_train = features.iloc[:-test_size]
    X_test = features.iloc[-test_size:]

    y_train = values.iloc[:-test_size]
    y_test = values.iloc[-test_size:]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    rmse = math.sqrt(
        mean_squared_error(
            y_test,
            test_predictions
        )
    )

    # --------------------------------------------------------
    # HISTORICAL ACTUAL VS PREDICTED
    # --------------------------------------------------------
    #
    # Refit the model on all historical data so that the chart
    # shows predictions for the complete historical dataset.
    #
    # --------------------------------------------------------

    historical_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )

    historical_model.fit(
        features,
        values
    )

    historical_predictions = historical_model.predict(
        features
    )

    historical_comparison = pd.DataFrame({
        "date": dates,
        "actual": values,
        "predicted": np.maximum(
            historical_predictions,
            0
        )
    })

    historical_comparison["actual"] = (
        historical_comparison["actual"]
        .round(2)
    )

    historical_comparison["predicted"] = (
        historical_comparison["predicted"]
        .round(2)
    )

    last_date = pd.to_datetime(
        dates.max()
    )

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    future_features = create_time_features(
        future_dates
    )

    future_predictions = historical_model.predict(
        future_features
    )

    future_predictions = np.maximum(
        future_predictions,
        0
    )

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "predicted_demand": np.round(
            future_predictions,
            2
        )
    })

    return {
        "model": "Random Forest",
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "forecast": forecast_df,
        "historical_comparison": historical_comparison
    }


def run_gradient_boosting_forecast(
    time_series,
    horizon=30
):
    date_column = time_series.columns[0]
    target_column = time_series.columns[1]

    dates = time_series[date_column]
    values = time_series[target_column]

    features = create_time_features(dates)

    total_records = len(features)

    test_size = max(
        3,
        int(total_records * 0.2)
    )

    if total_records - test_size < 5:
        test_size = 3

    X_train = features.iloc[:-test_size]
    X_test = features.iloc[-test_size:]

    y_train = values.iloc[:-test_size]
    y_test = values.iloc[-test_size:]

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        test_predictions
    )

    rmse = math.sqrt(
        mean_squared_error(
            y_test,
            test_predictions
        )
    )

    # --------------------------------------------------------
    # HISTORICAL ACTUAL VS PREDICTED
    # --------------------------------------------------------

    historical_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    historical_model.fit(
        features,
        values
    )

    historical_predictions = historical_model.predict(
        features
    )

    historical_comparison = pd.DataFrame({
        "date": dates,
        "actual": values,
        "predicted": np.maximum(
            historical_predictions,
            0
        )
    })

    historical_comparison["actual"] = (
        historical_comparison["actual"]
        .round(2)
    )

    historical_comparison["predicted"] = (
        historical_comparison["predicted"]
        .round(2)
    )

    last_date = pd.to_datetime(
        dates.max()
    )

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="D"
    )

    future_features = create_time_features(
        future_dates
    )

    future_predictions = historical_model.predict(
        future_features
    )

    future_predictions = np.maximum(
        future_predictions,
        0
    )

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "predicted_demand": np.round(
            future_predictions,
            2
        )
    })

    return {
        "model": "Gradient Boosting",
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "forecast": forecast_df,
        "historical_comparison": historical_comparison
    }


def run_prophet_forecast(
    time_series,
    horizon=30
):
    try:
        from prophet import Prophet
    except ImportError:
        return {
            "model": "Prophet",
            "available": False,
            "error": "Prophet is not installed.",
            "mae": None,
            "rmse": None,
            "forecast": None,
            "historical_comparison": None
        }

    date_column = time_series.columns[0]
    target_column = time_series.columns[1]

    prophet_df = time_series[
        [date_column, target_column]
    ].copy()

    prophet_df.columns = ["ds", "y"]

    prophet_df["ds"] = pd.to_datetime(
        prophet_df["ds"]
    )

    prophet_df["y"] = pd.to_numeric(
        prophet_df["y"],
        errors="coerce"
    )

    prophet_df = prophet_df.dropna()

    if len(prophet_df) < 10:
        return {
            "model": "Prophet",
            "available": True,
            "error": "Not enough records for Prophet.",
            "mae": None,
            "rmse": None,
            "forecast": None,
            "historical_comparison": None
        }

    test_size = max(
        3,
        int(len(prophet_df) * 0.2)
    )

    train_df = prophet_df.iloc[:-test_size]
    test_df = prophet_df.iloc[-test_size:]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(train_df)

    test_dates = test_df[["ds"]]

    test_prediction = model.predict(
        test_dates
    )

    predicted_test = test_prediction["yhat"].values
    actual_test = test_df["y"].values

    mae = mean_absolute_error(
        actual_test,
        predicted_test
    )

    rmse = math.sqrt(
        mean_squared_error(
            actual_test,
            predicted_test
        )
    )

    final_model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    final_model.fit(prophet_df)

    # --------------------------------------------------------
    # HISTORICAL ACTUAL VS PREDICTED
    # --------------------------------------------------------

    historical_forecast = final_model.predict(
        prophet_df[["ds"]]
    )

    historical_comparison = pd.DataFrame({
        "date": historical_forecast["ds"],
        "actual": prophet_df["y"].values,
        "predicted": np.maximum(
            historical_forecast["yhat"].values,
            0
        )
    })

    historical_comparison["actual"] = (
        historical_comparison["actual"]
        .round(2)
    )

    historical_comparison["predicted"] = (
        historical_comparison["predicted"]
        .round(2)
    )

    future = final_model.make_future_dataframe(
        periods=horizon,
        freq="D"
    )

    prediction = final_model.predict(
        future
    )

    future_prediction = (
        prediction.tail(horizon)[
            ["ds", "yhat"]
        ]
        .copy()
    )

    future_prediction.columns = [
        "date",
        "predicted_demand"
    ]

    future_prediction["predicted_demand"] = np.maximum(
        future_prediction["predicted_demand"],
        0
    )

    future_prediction["predicted_demand"] = (
        future_prediction["predicted_demand"]
        .round(2)
    )

    return {
        "model": "Prophet",
        "available": True,
        "error": None,
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "forecast": future_prediction,
        "historical_comparison": historical_comparison
    }


def save_forecast_report(report):
    try:
        with open(
            FORECAST_REPORT_PATH,
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                report,
                file,
                indent=4,
                default=str
            )

        return True

    except Exception:
        return False


def generate_service_forecast_result(dataset):
    result = {
        "success": False,
        "message": "",
        "date_column": None,
        "target_column": None,
        "models": [],
        "best_model": None,
        "best_mae": None,
        "best_rmse": None,
        "forecast": None,
        "forecast_rows": 0,
        "historical_comparison": None
    }

    if dataset is None:
        result["message"] = "No dataset is available for forecasting."
        return result

    if dataset.empty:
        result["message"] = "The selected dataset is empty."
        return result

    date_column, target_column, mapping = get_forecasting_columns(dataset)
    result["date_column"] = date_column
    result["target_column"] = target_column

    if not date_column or not target_column:
        result["message"] = (
            "A valid date and demand or sales column are required."
        )
        return result

    try:
        trained = train_forecasting_models(
            dataset,
            date_column,
            target_column
        )
        prepared_data = trained["data"]
        best_model = trained["best_model"]
        best_name = trained["best_model_name"]
        metrics = trained["best_metrics"]

        future_forecasts = {}
        for name, model_data in trained["models"].items():
            model_forecast = generate_service_forecast(
                model_data["model"],
                prepared_data[date_column].max(),
                FORECAST_HORIZON
            )
            future_forecasts[name] = model_forecast.rename(
                columns={"forecast": "predicted_demand"}
            )

        future = future_forecasts[best_name]

        features = prepared_data[trained["feature_columns"]]
        historical_predictions = best_model.predict(features)
        historical = pd.DataFrame({
            "date": prepared_data[date_column],
            "actual": prepared_data[target_column].round(2),
            "predicted": np.maximum(historical_predictions, 0).round(2)
        })

        result["models"] = []
        for name, model_data in trained["models"].items():
            model_metrics = model_data["metrics"]
            result["models"].append({
                "model": name,
                "mae": model_metrics["mae"],
                "rmse": model_metrics["rmse"],
                "r2": model_metrics["r2"],
                "forecast": future_forecasts[name],
                "available": True,
                "error": None
            })

        result["best_model"] = best_name
        result["best_mae"] = metrics["mae"]
        result["best_rmse"] = metrics["rmse"]
        result["forecast"] = future
        result["forecast_rows"] = int(len(future))
        result["historical_comparison"] = historical

        save_df = future.copy()
        save_df["model"] = best_name
        save_df.to_csv(FORECAST_RESULT_PATH, index=False)
        save_forecast_report({
            "date_column": date_column,
            "target_column": target_column,
            "forecast_horizon": FORECAST_HORIZON,
            "best_model": best_name,
            "best_mae": metrics["mae"],
            "best_rmse": metrics["rmse"],
            "models": [
                {
                    "model": item["model"],
                    "mae": item["mae"],
                    "rmse": item["rmse"],
                    "r2": item["r2"]
                }
                for item in result["models"]
            ],
            "forecast_file": FORECAST_RESULT_PATH
        })

        result["success"] = True
        result["message"] = "Forecast generated using the forecasting service."
        return result

    except Exception as error:
        result["message"] = f"Forecasting service failed: {error}"
        return result


def generate_forecast(dataset=None):
    return generate_service_forecast_result(
        dataset if dataset is not None else load_shared_dataset()
    )

    # Legacy inline implementation retained below for compatibility.
    result = {
        "success": False,
        "message": "",
        "date_column": None,
        "target_column": None,
        "models": [],
        "best_model": None,
        "best_mae": None,
        "best_rmse": None,
        "forecast": None,
        "forecast_rows": 0,
        "historical_comparison": None
    }

    df = dataset if dataset is not None else load_shared_dataset()

    if df is None:
        result["message"] = (
            "No shared dataset is available."
        )
        return result

    if df.empty:
        result["message"] = (
            "The shared dataset is empty."
        )
        return result

    preparation = prepare_forecasting_data(df)

    if not preparation["success"]:
        result["message"] = preparation["message"]
        result["date_column"] = preparation["date_column"]
        result["target_column"] = preparation["target_column"]
        return result

    time_series = preparation["data"]

    result["date_column"] = preparation["date_column"]
    result["target_column"] = preparation["target_column"]

    try:
        rf_result = run_random_forest_forecast(
            time_series,
            FORECAST_HORIZON
        )
        result["models"].append(rf_result)

    except Exception as e:
        result["models"].append({
            "model": "Random Forest",
            "mae": None,
            "rmse": None,
            "forecast": None,
            "historical_comparison": None,
            "error": str(e)
        })

    try:
        gb_result = run_gradient_boosting_forecast(
            time_series,
            FORECAST_HORIZON
        )
        result["models"].append(gb_result)

    except Exception as e:
        result["models"].append({
            "model": "Gradient Boosting",
            "mae": None,
            "rmse": None,
            "forecast": None,
            "historical_comparison": None,
            "error": str(e)
        })

    try:
        prophet_result = run_prophet_forecast(
            time_series,
            FORECAST_HORIZON
        )
        result["models"].append(prophet_result)

    except Exception as e:
        result["models"].append({
            "model": "Prophet",
            "mae": None,
            "rmse": None,
            "forecast": None,
            "historical_comparison": None,
            "error": str(e)
        })

    valid_models = [
        model
        for model in result["models"]
        if (
            model.get("forecast") is not None
            and model.get("rmse") is not None
        )
    ]

    if not valid_models:
        result["message"] = (
            "No forecasting model could be executed successfully."
        )
        return result

    best_model = min(
        valid_models,
        key=lambda x: x["rmse"]
    )

    result["best_model"] = best_model["model"]
    result["best_mae"] = best_model["mae"]
    result["best_rmse"] = best_model["rmse"]
    result["forecast"] = best_model["forecast"]
    result["forecast_rows"] = int(
        len(best_model["forecast"])
    )

    # ========================================================
    # HISTORICAL CHART DATA
    # ========================================================

    historical_comparison = best_model.get(
        "historical_comparison"
    )

    if historical_comparison is not None:
        result["historical_comparison"] = (
            historical_comparison.copy()
        )

    try:
        save_df = best_model["forecast"].copy()

        save_df["model"] = best_model["model"]

        save_df.to_csv(
            FORECAST_RESULT_PATH,
            index=False
        )

    except Exception as e:
        result["message"] = (
            f"Forecast generated, but result "
            f"could not be saved: {str(e)}"
        )
        return result

    model_report = []

    for model in result["models"]:
        model_report.append({
            "model": model.get("model"),
            "mae": model.get("mae"),
            "rmse": model.get("rmse"),
            "available": model.get("available", True),
            "error": model.get("error")
        })

    report = {
        "date_column": result["date_column"],
        "target_column": result["target_column"],
        "forecast_horizon": FORECAST_HORIZON,
        "best_model": result["best_model"],
        "best_mae": result["best_mae"],
        "best_rmse": result["best_rmse"],
        "models": model_report,
        "forecast_file": FORECAST_RESULT_PATH
    }

    save_forecast_report(report)

    result["success"] = True
    result["message"] = (
        "Forecast generated successfully."
    )

    return result


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    return redirect(url_for("login"))


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get(
            "confirm_password",
            ""
        )

        role = request.form.get(
            "role",
            "Business Analyst"
        )

        if not name:
            flash("Please enter your name.", "danger")
            return redirect(url_for("register"))

        if not email:
            flash("Please enter your email.", "danger")
            return redirect(url_for("register"))

        if not password:
            flash("Please enter a password.", "danger")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match.", "danger")
            return redirect(url_for("register"))

        if len(password) < 6:
            flash(
                "Password must contain at least 6 characters.",
                "danger"
            )
            return redirect(url_for("register"))

        allowed_roles = {
            "Admin",
            "Inventory Manager",
            "Business Analyst"
        }

        if role not in allowed_roles:
            role = "Business Analyst"

        existing_user = User.query.filter_by(
            email=email
        ).first()

        if existing_user:
            flash(
                "An account with this email already exists.",
                "warning"
            )
            return redirect(url_for("register"))

        hashed_password = generate_password_hash(password)

        new_user = User(
            name=name,
            email=email,
            password=hashed_password,
            role=role
        )

        db.session.add(new_user)
        db.session.commit()

        flash(
            "Registration successful. Please login.",
            "success"
        )

        return redirect(url_for("login"))

    return render_template("register.html")


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        email = request.form.get(
            "email",
            ""
        ).strip().lower()

        password = request.form.get(
            "password",
            ""
        )

        user = User.query.filter_by(
            email=email
        ).first()

        if (
            user
            and
            check_password_hash(
                user.password,
                password
            )
        ):
            login_user(user)

            flash(
                "Welcome back!",
                "success"
            )

            return redirect(url_for("dashboard"))

        flash(
            "Invalid email or password.",
            "danger"
        )

    return render_template("login.html")


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():
    dataset_info = get_shared_dataset_info()

    return render_template(
        "dashboard.html",
        user=current_user,
        shared_dataset_available=dataset_info["available"],
        shared_dataset_rows=dataset_info["rows"],
        shared_dataset_columns=dataset_info["columns"]
    )


# ============================================================
# DATASET MANAGEMENT
# ADMIN ONLY
# ============================================================

@app.route("/datasets")
@login_required
@role_required("Admin")
def datasets():
    dataset_info = get_shared_dataset_info()

    return render_template(
        "dataset_upload.html",
        shared_dataset_available=dataset_info["available"],
        shared_dataset_rows=dataset_info["rows"],
        shared_dataset_columns=dataset_info["columns"]
    )


# ============================================================
# DATASET UPLOAD + AUTOMATIC CLEANING
# ADMIN ONLY
# ============================================================

@app.route(
    "/upload-dataset",
    methods=["POST"]
)
@login_required
@role_required("Admin")
def upload_dataset():

    dataset_type = request.form.get(
        "dataset_type",
        ""
    ).strip().lower()

    allowed_dataset_types = {
        "sales",
        "inventory",
        "product",
        "supplier",
        "weather",
        "holiday",
        "other"
    }

    if dataset_type not in allowed_dataset_types:
        flash(
            "Please select a valid dataset type.",
            "danger"
        )
        return redirect(url_for("datasets"))

    if "dataset" not in request.files:
        flash(
            "No dataset file was selected.",
            "danger"
        )
        return redirect(url_for("datasets"))

    file = request.files["dataset"]

    if file.filename == "":
        flash(
            "Please select a CSV file.",
            "danger"
        )
        return redirect(url_for("datasets"))

    filename = secure_filename(file.filename)

    if not allowed_file(filename):
        flash(
            "Only CSV files are currently supported.",
            "danger"
        )
        return redirect(url_for("datasets"))

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    try:
        file.save(upload_path)

    except Exception as e:
        flash(
            f"Could not save the uploaded file: {str(e)}",
            "danger"
        )
        return redirect(url_for("datasets"))

    try:
        original_df = read_csv_safely(upload_path)

    except pd.errors.EmptyDataError:
        flash(
            "The uploaded CSV file is empty.",
            "danger"
        )
        return redirect(url_for("datasets"))

    except pd.errors.ParserError:
        flash(
            "The CSV file has an invalid structure and could not be read.",
            "danger"
        )
        return redirect(url_for("datasets"))

    except Exception as e:
        flash(
            f"Could not process the CSV file: {str(e)}",
            "danger"
        )
        return redirect(url_for("datasets"))

    if original_df.empty:
        flash(
            "The uploaded CSV contains no data.",
            "danger"
        )
        return redirect(url_for("datasets"))

    original_rows = len(original_df)
    original_columns = len(original_df.columns)

    original_missing_values = int(
        original_df.isnull().sum().sum()
    )

    original_duplicate_rows = int(
        original_df.duplicated().sum()
    )

    raw_filename = f"{dataset_type}_{filename}"
    raw_path = os.path.join(
        RAW_FOLDER,
        raw_filename
    )

    try:
        original_df.to_csv(
            raw_path,
            index=False
        )
    except Exception:
        flash(
            "Dataset was uploaded, but the raw copy could not be saved.",
            "warning"
        )

    try:
        cleaned_df, cleaning_report = clean_dataset(
            original_df,
            remove_outliers=False
        )

    except Exception as e:
        flash(
            f"Dataset cleaning failed: {str(e)}",
            "danger"
        )
        return redirect(url_for("datasets"))

    try:
        cleaned_df.to_csv(
            SHARED_DATASET_PATH,
            index=False
        )

    except Exception as e:
        flash(
            f"Cleaned dataset could not be saved: {str(e)}",
            "danger"
        )
        return redirect(url_for("datasets"))

    report_saved = save_cleaning_report(
        cleaning_report
    )

    column_mapping = cleaning_report.get(
        "automatic_column_mapping",
        {}
    )

    mapped_count = sum(
        1
        for value in column_mapping.values()
        if value is not None
    )

    total_mapping_fields = len(column_mapping)

    has_date = (
        column_mapping.get("date") is not None
    )

    has_demand = (
        column_mapping.get("demand") is not None
    )

    has_sales = (
        column_mapping.get("sales") is not None
    )

    forecasting_ready = (
        has_date
        and
        (has_demand or has_sales)
    )

    cleaned_rows = len(cleaned_df)
    cleaned_columns = len(cleaned_df.columns)

    cleaned_missing_values = int(
        cleaned_df.isnull().sum().sum()
    )

    cleaned_duplicate_rows = int(
        cleaned_df.duplicated().sum()
    )

    preview_df = cleaned_df.head(10)

    preview_html = preview_df.to_html(
        classes=(
            "table "
            "table-hover "
            "table-bordered"
        ),
        index=False
    )

    flash(
        "Dataset uploaded and cleaned successfully. "
        "The cleaned dataset is now the shared dataset "
        "for Forecasting, Inventory, Analytics and Reports.",
        "success"
    )

    return render_template(
        "dataset_upload.html",
        upload_success=True,
        filename=filename,
        dataset_type=dataset_type,
        total_rows=original_rows,
        total_columns=original_columns,
        missing_values=original_missing_values,
        duplicate_rows=original_duplicate_rows,
        cleaned_rows=cleaned_rows,
        cleaned_columns=cleaned_columns,
        cleaned_missing_values=cleaned_missing_values,
        cleaned_duplicate_rows=cleaned_duplicate_rows,
        columns=list(cleaned_df.columns),
        preview_html=preview_html,
        column_mapping=column_mapping,
        mapped_count=mapped_count,
        total_mapping_fields=total_mapping_fields,
        cleaned_success=True,
        cleaning_report=cleaning_report,
        report_saved=report_saved,
        forecasting_ready=forecasting_ready,
        shared_dataset_available=True,
        shared_dataset_rows=cleaned_rows,
        shared_dataset_columns=cleaned_columns
    )


# ============================================================
# INDEPENDENT MODULE DATASET UPLOAD
# ============================================================

@app.route(
    "/module-upload/<module_name>",
    methods=["POST"]
)
@login_required
def upload_module_dataset(module_name):

    if module_name not in MODULE_NAMES:
        flash("That module does not support independent datasets.", "danger")
        return redirect(url_for("dashboard"))

    file = request.files.get("dataset")

    if file is None or file.filename == "":
        flash("Please select a CSV file.", "danger")
        return redirect(url_for(module_name))

    filename = secure_filename(file.filename)

    if not allowed_file(filename):
        flash("Only CSV files are currently supported.", "danger")
        return redirect(url_for(module_name))

    temporary_path = os.path.join(
        UPLOAD_FOLDER,
        f"{module_name}_{filename}"
    )
    independent_path = module_source_path(module_name)

    try:
        file.save(temporary_path)
        original_df = read_csv_safely(temporary_path)

        if original_df.empty:
            raise ValueError("The uploaded CSV contains no data.")

        cleaned_df, cleaning_report = clean_dataset(
            original_df,
            remove_outliers=False
        )
        cleaned_df.to_csv(independent_path, index=False)

        metadata = load_module_source_metadata()
        metadata[module_name] = {
            "filename": filename,
            "rows": int(len(cleaned_df)),
            "columns": int(len(cleaned_df.columns)),
            "cleaning_report": cleaning_report
        }
        save_module_source_metadata(metadata)

    except Exception as error:
        flash(f"Independent dataset could not be processed: {error}", "danger")
        return redirect(url_for(module_name))

    flash(
        f"{filename} is ready for independent {module_name} analysis. ",
        "success"
    )
    return redirect(
        url_for(module_name, source="independent")
    )


# ============================================================
# FORECASTING PAGE
# ============================================================

@app.route("/forecasting")
@login_required
def forecasting():

    source_context = get_module_source_context("forecasting")
    dataset_info = get_shared_dataset_info()

    forecast_result = None
    forecast_models = []
    best_model = None
    best_mae = None
    best_rmse = None
    forecast_rows = []
    forecasting_message = None

    # --------------------------------------------------------
    # EMPTY CHART VARIABLES
    # --------------------------------------------------------

    chart_dates = []
    chart_actual = []
    chart_predicted = []

    if source_context["selected_dataset"] is not None:

        df = source_context["selected_dataset"]

        if df is not None:

            preparation = prepare_forecasting_data(df)

            if preparation["success"]:

                forecast_result = generate_forecast(df)

                if forecast_result["success"]:

                    forecast_models = (
                        forecast_result["models"]
                    )

                    best_model = (
                        forecast_result["best_model"]
                    )

                    best_mae = (
                        forecast_result["best_mae"]
                    )

                    best_rmse = (
                        forecast_result["best_rmse"]
                    )

                    if forecast_result["forecast"] is not None:

                        forecast_rows = (
                            forecast_result["forecast"]
                            .copy()
                            .to_dict(
                                orient="records"
                            )
                        )

                    # ------------------------------------------------
                    # HISTORICAL CHART DATA
                    # ------------------------------------------------

                    historical_df = (
                        forecast_result
                        .get("historical_comparison")
                    )

                    if historical_df is not None:
                        historical_df = (
                            historical_df.copy()
                        )

                        historical_df["date"] = (
                            pd.to_datetime(
                                historical_df["date"]
                            )
                            .dt.strftime("%Y-%m-%d")
                        )

                        chart_dates = (
                            historical_df["date"]
                            .tolist()
                        )

                        chart_actual = (
                            historical_df["actual"]
                            .tolist()
                        )

                        chart_predicted = (
                            historical_df["predicted"]
                            .tolist()
                        )

                    forecasting_message = (
                        forecast_result["message"]
                    )

                else:

                    forecasting_message = (
                        forecast_result["message"]
                    )

            else:

                forecasting_message = (
                    preparation["message"]
                )

    return render_template(
        "forecasting.html",

        module_name="forecasting",

        shared_dataset_available=
            dataset_info["available"],

        shared_dataset_rows=
            dataset_info["rows"],

        shared_dataset_columns=
            dataset_info["columns"],

        shared_dataset_columns_list=
            dataset_info["columns_list"],

        column_mapping=
            dataset_info["mapping"],

        forecast_result=
            forecast_result,

        forecast_models=
            forecast_models,

        best_model=
            best_model,

        best_mae=
            best_mae,

        best_rmse=
            best_rmse,

        forecast_rows=
            forecast_rows,

        forecasting_message=
            forecasting_message,

        # --------------------------------------------------------
        # CHART DATA
        # --------------------------------------------------------

        chart_dates=
            chart_dates,

        chart_actual=
            chart_actual,

        chart_predicted=
            chart_predicted,

        **{
            key: value
            for key, value in source_context.items()
            if key not in {
                "selected_dataset",
                "shared_dataset_available",
                "shared_dataset_rows",
                "shared_dataset_columns",
                "shared_dataset_filename"
            }
        }
    )


# ============================================================
# RUN FORECASTING MANUALLY
# ============================================================

@app.route(
    "/run-forecasting",
    methods=["POST"]
)
@login_required
def run_forecasting():

    source_context = get_module_source_context("forecasting")

    if source_context["selected_dataset"] is None:

        flash(
            "No shared dataset is available. "
            "Please upload a dataset from Data Management first.",
            "warning"
        )

        return redirect(
            url_for("forecasting")
        )

    result = generate_forecast(source_context["selected_dataset"])

    if result["success"]:

        flash(
            f"Forecast generated successfully using "
            f"{result['best_model']}. "
            f"Forecast horizon: {FORECAST_HORIZON} days.",
            "success"
        )

    else:

        flash(
            result["message"],
            "danger"
        )

    return redirect(
        url_for("forecasting")
    )


# ============================================================
# INVENTORY PAGE
# ============================================================

@app.route("/inventory")
@login_required
def inventory():

    source_context = get_module_source_context("inventory")
    inventory_data = calculate_inventory_statistics(
        source_context["selected_dataset"]
    )

    return render_template(
        "inventory.html",

        module_name="inventory",

        inventory_available=
            inventory_data["inventory_available"],

        total_products=
            inventory_data["total_products"],

        low_stock=
            inventory_data["low_stock"],

        overstock=
            inventory_data["overstock"],

        reorder_required=
            inventory_data["reorder_required"],

        average_daily_demand=
            inventory_data["average_daily_demand"],

        lead_time=
            inventory_data["lead_time"],

        safety_stock=
            inventory_data["safety_stock"],

        reorder_point=
            inventory_data["reorder_point"],

        inventory_message=
            inventory_data["inventory_message"],

        **{
            key: value
            for key, value in source_context.items()
            if key != "selected_dataset"
        }
    )


# ============================================================
# ANALYTICS PAGE
# ============================================================

@app.route("/analytics")
@login_required
def analytics():

    source_context = get_module_source_context("analytics")
    dataset = source_context["selected_dataset"]
    dataset_info = get_shared_dataset_info()
    analytics_available = dataset is not None
    total_sales = None
    total_demand = None
    products_analyzed = None
    data_period = None

    if analytics_available:
        mapping = automatic_column_mapping(dataset)
        sales_column = mapping.get("sales")
        demand_column = mapping.get("demand") or mapping.get("quantity")
        product_column = mapping.get("product")
        date_column = mapping.get("date")
        if sales_column:
            sales_values = pd.to_numeric(
                dataset[sales_column],
                errors="coerce"
            ).dropna()
            if not sales_values.empty:
                total_sales = round(float(sales_values.sum()), 2)
        if demand_column:
            demand_values = pd.to_numeric(
                dataset[demand_column],
                errors="coerce"
            ).dropna()
            if not demand_values.empty:
                total_demand = round(float(demand_values.sum()), 2)
        if product_column:
            products_analyzed = int(dataset[product_column].nunique())
        if date_column:
            dates = pd.to_datetime(dataset[date_column], errors="coerce").dropna()
            if not dates.empty:
                data_period = f"{dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}"

    return render_template(
        "analytics.html",

        module_name="analytics",

        shared_dataset_available=
            dataset_info["available"],

        shared_dataset_rows=
            dataset_info["rows"],

        shared_dataset_columns=
            dataset_info["columns"],

        shared_dataset_columns_list=
            dataset_info["columns_list"],
        analytics_available=analytics_available,
        total_sales=total_sales,
        total_demand=total_demand,
        products_analyzed=products_analyzed,
        data_period=data_period,
        **{
            key: value
            for key, value in source_context.items()
            if key not in {
                "selected_dataset",
                "shared_dataset_available",
                "shared_dataset_rows",
                "shared_dataset_columns",
                "shared_dataset_filename"
            }
        }
    )


# ============================================================
# SEASONAL ANALYSIS PAGE
# ============================================================

@app.route("/seasonal")
@login_required
def seasonal():

    source_context = get_module_source_context("seasonal")

    return render_template(
        "seasonal.html",
        module_name="seasonal",
        **{
            key: value
            for key, value in source_context.items()
            if key != "selected_dataset"
        }
    )


# ============================================================
# SMART ALERTS PAGE
# ============================================================

@app.route("/alerts")
@login_required
def alerts():

    source_context = get_module_source_context("alerts")
    dataset_info = get_shared_dataset_info()

    return render_template(
        "alerts.html",

        module_name="alerts",

        shared_dataset_available=
            dataset_info["available"],

        shared_dataset_rows=
            dataset_info["rows"],

        shared_dataset_columns=
            dataset_info["columns"],
        **{
            key: value
            for key, value in source_context.items()
            if key not in {
                "selected_dataset",
                "shared_dataset_available",
                "shared_dataset_rows",
                "shared_dataset_columns",
                "shared_dataset_filename"
            }
        }
    )


# ============================================================
# REPORTS PAGE
# ============================================================

@app.route("/reports")
@login_required
def reports():

    source_context = get_module_source_context("reports")
    dataset_info = get_shared_dataset_info()

    return render_template(
        "reports.html",

        module_name="reports",

        shared_dataset_available=
            dataset_info["available"],

        shared_dataset_rows=
            dataset_info["rows"],

        shared_dataset_columns=
            dataset_info["columns"],
        **{
            key: value
            for key, value in source_context.items()
            if key not in {
                "selected_dataset",
                "shared_dataset_available",
                "shared_dataset_rows",
                "shared_dataset_columns",
                "shared_dataset_filename"
            }
        }
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
@login_required
def logout():

    logout_user()

    flash(
        "You have been logged out.",
        "info"
    )

    return redirect(
        url_for("login")
    )


# ============================================================
# CREATE DATABASE TABLES
# ============================================================

with app.app_context():
    db.create_all()


# ============================================================
# START APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )
