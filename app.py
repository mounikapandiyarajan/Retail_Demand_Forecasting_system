# ============================================================
# RETAIL DEMAND FORECASTING SYSTEM
# Flask Backend
# ============================================================

import os
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

from config import Config
from database.models import db, User


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)
app.config.from_object(Config)


# ============================================================
# DATASET FOLDERS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "uploads"
)

RAW_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

CLEANED_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "cleaned"
)

REPORT_FOLDER = os.path.join(
    BASE_DIR,
    "data",
    "reports"
)


# ============================================================
# CREATE FOLDERS
# ============================================================

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    RAW_FOLDER,
    exist_ok=True
)

os.makedirs(
    CLEANED_FOLDER,
    exist_ok=True
)

os.makedirs(
    REPORT_FOLDER,
    exist_ok=True
)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

app.config["MAX_CONTENT_LENGTH"] = (
    100 * 1024 * 1024
)


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_EXTENSIONS = {
    "csv"
}


def allowed_file(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# DATABASE
# ============================================================

db.init_app(app)


# ============================================================
# LOGIN MANAGER
# ============================================================

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"

login_manager.login_message = (
    "Please login to access this page."
)


@login_manager.user_loader
def load_user(user_id):

    return User.query.get(
        int(user_id)
    )


# ============================================================
# ROLE BASED ACCESS CONTROL
# ============================================================

def role_required(*allowed_roles):

    def decorator(function):

        @wraps(function)
        def wrapper(*args, **kwargs):

            if not current_user.is_authenticated:

                flash(
                    "Please login to continue.",
                    "warning"
                )

                return redirect(
                    url_for("login")
                )


            user_role = current_user.role


            if user_role not in allowed_roles:

                flash(
                    "You do not have permission to access this page.",
                    "danger"
                )

                return redirect(
                    url_for("dashboard")
                )


            return function(
                *args,
                **kwargs
            )

        return wrapper

    return decorator


# ============================================================
# COLUMN DETECTION HELPERS
# ============================================================

COLUMN_KEYWORDS = {

    "date": [
        "date",
        "order_date",
        "sales_date",
        "sale_date",
        "transaction_date",
        "timestamp",
        "datetime"
    ],

    "product": [
        "product",
        "product_id",
        "product_name",
        "sku",
        "sku_id",
        "item",
        "item_id",
        "item_name"
    ],

    "demand": [
        "demand",
        "predicted_demand",
        "forecast_demand",
        "units_demand",
        "daily_demand"
    ],

    "sales": [
        "sales",
        "sale",
        "sales_quantity",
        "sales_qty",
        "units_sold",
        "quantity_sold",
        "sold_quantity",
        "revenue"
    ],

    "store": [
        "store",
        "store_id",
        "store_name",
        "shop",
        "shop_id",
        "branch",
        "branch_id"
    ],

    "price": [
        "price",
        "unit_price",
        "selling_price",
        "sale_price",
        "product_price",
        "cost_price"
    ],

    "discount": [
        "discount",
        "discount_percent",
        "discount_percentage",
        "discount_rate"
    ],

    "promotion": [
        "promotion",
        "promo",
        "promotional",
        "is_promotion",
        "promotion_flag"
    ],

    "inventory": [
        "inventory",
        "stock",
        "current_stock",
        "stock_level",
        "stock_quantity",
        "inventory_quantity",
        "available_stock",
        "quantity_in_stock",
        "on_hand",
        "on_hand_quantity"
    ],

    "category": [
        "category",
        "product_category",
        "category_name",
        "department"
    ],

    "supplier": [
        "supplier",
        "supplier_id",
        "supplier_name",
        "vendor",
        "vendor_id",
        "vendor_name"
    ],

    "lead_time": [
        "lead_time",
        "leadtime",
        "supplier_lead_time",
        "delivery_time",
        "shipping_time"
    ]
}


def normalize_column_name(column):

    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def detect_column_mapping(df):

    mapping = {}

    normalized_columns = {}

    for column in df.columns:

        normalized_columns[column] = (
            normalize_column_name(column)
        )


    for field, keywords in COLUMN_KEYWORDS.items():

        mapping[field] = None

        # Exact match first

        for original, normalized in normalized_columns.items():

            if normalized in keywords:

                mapping[field] = original

                break


        # Partial match second

        if mapping[field] is None:

            for original, normalized in normalized_columns.items():

                for keyword in keywords:

                    if (
                        keyword in normalized
                        or
                        normalized in keyword
                    ):

                        mapping[field] = original

                        break

                if mapping[field] is not None:

                    break


    return mapping


def count_mapped_fields(mapping):

    return sum(
        1
        for value in mapping.values()
        if value is not None
    )


# ============================================================
# DATASET READING HELPER
# ============================================================

def read_csv_safely(file_path):

    try:

        return pd.read_csv(
            file_path,
            low_memory=False
        )

    except UnicodeDecodeError:

        return pd.read_csv(
            file_path,
            encoding="latin1",
            low_memory=False
        )


# ============================================================
# FIND LATEST INVENTORY DATASET
# ============================================================

def find_latest_inventory_dataset():

    if not os.path.exists(RAW_FOLDER):

        return None


    inventory_files = []


    for filename in os.listdir(RAW_FOLDER):

        if not filename.lower().endswith(".csv"):

            continue


        if filename.lower().startswith("inventory_"):

            path = os.path.join(
                RAW_FOLDER,
                filename
            )

            inventory_files.append(path)


    if not inventory_files:

        return None


    inventory_files.sort(
        key=os.path.getmtime,
        reverse=True
    )


    return inventory_files[0]


# ============================================================
# INVENTORY ANALYSIS
# ============================================================

def calculate_inventory_statistics():

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

        "inventory_message":
            "No inventory dataset has been uploaded yet."

    }


    inventory_path = (
        find_latest_inventory_dataset()
    )


    if not inventory_path:

        return result


    try:

        df = read_csv_safely(
            inventory_path
        )

    except Exception:

        return result


    if df.empty:

        result["inventory_message"] = (
            "The uploaded inventory dataset contains no data."
        )

        return result


    df.columns = [

        normalize_column_name(column)

        for column in df.columns

    ]


    # --------------------------------------------------------
    # IDENTIFY IMPORTANT INVENTORY COLUMNS
    # --------------------------------------------------------

    mapping = detect_column_mapping(df)


    product_column = mapping.get(
        "product"
    )

    inventory_column = mapping.get(
        "inventory"
    )

    demand_column = mapping.get(
        "demand"
    )

    sales_column = mapping.get(
        "sales"
    )

    lead_time_column = mapping.get(
        "lead_time"
    )


    # --------------------------------------------------------
    # PRODUCT COUNT
    # --------------------------------------------------------

    if product_column and product_column in df.columns:

        total_products = int(
            df[product_column]
            .nunique()
        )

    else:

        total_products = len(df)


    result["total_products"] = (
        total_products
    )


    # --------------------------------------------------------
    # INVENTORY / STOCK ANALYSIS
    # --------------------------------------------------------

    if inventory_column and inventory_column in df.columns:

        stock = pd.to_numeric(
            df[inventory_column],
            errors="coerce"
        )

        valid_stock = stock.dropna()


        if not valid_stock.empty:

            # ------------------------------------------------
            # LOW STOCK
            # ------------------------------------------------
            #
            # We only calculate this when a reorder
            # threshold is actually present.
            # Otherwise we do NOT invent a threshold.
            # ------------------------------------------------

            threshold_column = None

            threshold_candidates = [
                "reorder_threshold",
                "reorder_point",
                "minimum_stock",
                "min_stock",
                "minimum_stock_level",
                "safety_stock"
            ]


            for candidate in threshold_candidates:

                if candidate in df.columns:

                    threshold_column = candidate

                    break


            if threshold_column:

                threshold = pd.to_numeric(
                    df[threshold_column],
                    errors="coerce"
                )

                result["low_stock"] = int(
                    (
                        stock < threshold
                    )
                    .fillna(False)
                    .sum()
                )


            # ------------------------------------------------
            # OVERSTOCK
            # ------------------------------------------------

            max_stock_column = None

            max_stock_candidates = [
                "maximum_stock",
                "max_stock",
                "max_stock_level",
                "overstock_threshold"
            ]


            for candidate in max_stock_candidates:

                if candidate in df.columns:

                    max_stock_column = candidate

                    break


            if max_stock_column:

                max_stock = pd.to_numeric(
                    df[max_stock_column],
                    errors="coerce"
                )

                result["overstock"] = int(
                    (
                        stock > max_stock
                    )
                    .fillna(False)
                    .sum()
                )


            # ------------------------------------------------
            # REORDER REQUIRED
            # ------------------------------------------------

            if threshold_column:

                threshold = pd.to_numeric(
                    df[threshold_column],
                    errors="coerce"
                )

                result["reorder_required"] = int(
                    (
                        stock <= threshold
                    )
                    .fillna(False)
                    .sum()
                )


    # --------------------------------------------------------
    # DEMAND
    # --------------------------------------------------------

    actual_demand_column = (
        demand_column
        or
        sales_column
    )


    if (
        actual_demand_column
        and
        actual_demand_column in df.columns
    ):

        demand_values = pd.to_numeric(
            df[actual_demand_column],
            errors="coerce"
        ).dropna()


        if not demand_values.empty:

            result["average_daily_demand"] = round(
                float(
                    demand_values.mean()
                ),
                2
            )


    # --------------------------------------------------------
    # LEAD TIME
    # --------------------------------------------------------

    if (
        lead_time_column
        and
        lead_time_column in df.columns
    ):

        lead_values = pd.to_numeric(
            df[lead_time_column],
            errors="coerce"
        ).dropna()


        if not lead_values.empty:

            result["lead_time"] = round(
                float(
                    lead_values.mean()
                ),
                2
            )


    # --------------------------------------------------------
    # REORDER POINT
    # --------------------------------------------------------

    average_demand = (
        result["average_daily_demand"]
    )

    lead_time = (
        result["lead_time"]
    )


    if (
        average_demand is not None
        and
        lead_time is not None
    ):

        result["reorder_point"] = round(
            average_demand * lead_time,
            2
        )


    # --------------------------------------------------------
    # SAFETY STOCK
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # Do not invent safety stock.
    #
    # If an actual safety-stock column exists,
    # use its real average.
    # --------------------------------------------------------

    if "safety_stock" in df.columns:

        safety_values = pd.to_numeric(
            df["safety_stock"],
            errors="coerce"
        ).dropna()


        if not safety_values.empty:

            result["safety_stock"] = round(
                float(
                    safety_values.mean()
                ),
                2
            )


    result["inventory_available"] = True

    result["inventory_message"] = (
        "Inventory dataset loaded successfully. "
        "Values shown below are calculated from the "
        "uploaded inventory data."
    )


    return result


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    if current_user.is_authenticated:

        return redirect(
            url_for("dashboard")
        )


    return redirect(
        url_for("login")
    )


# ============================================================
# REGISTER
# ============================================================

@app.route(
    "/register",
    methods=["GET", "POST"]
)
def register():

    if current_user.is_authenticated:

        return redirect(
            url_for("dashboard")
        )


    if request.method == "POST":

        # ----------------------------------------------------
        # GET FORM DATA
        # ----------------------------------------------------

        name = request.form.get(
            "name",
            ""
        ).strip()


        email = request.form.get(
            "email",
            ""
        ).strip().lower()


        password = request.form.get(
            "password",
            ""
        )


        confirm_password = request.form.get(
            "confirm_password",
            ""
        )


        role = request.form.get(
            "role",
            "Business Analyst"
        )


        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if not name:

            flash(
                "Please enter your name.",
                "danger"
            )

            return redirect(
                url_for("register")
            )


        if not email:

            flash(
                "Please enter your email.",
                "danger"
            )

            return redirect(
                url_for("register")
            )


        if not password:

            flash(
                "Please enter a password.",
                "danger"
            )

            return redirect(
                url_for("register")
            )


        if password != confirm_password:

            flash(
                "Passwords do not match.",
                "danger"
            )

            return redirect(
                url_for("register")
            )


        if len(password) < 6:

            flash(
                "Password must contain at least 6 characters.",
                "danger"
            )

            return redirect(
                url_for("register")
            )


        # ----------------------------------------------------
        # VALIDATE ROLE
        # ----------------------------------------------------

        allowed_roles = {
            "Admin",
            "Inventory Manager",
            "Business Analyst"
        }


        if role not in allowed_roles:

            role = "Business Analyst"


        # ----------------------------------------------------
        # CHECK EXISTING USER
        # ----------------------------------------------------

        existing_user = User.query.filter_by(
            email=email
        ).first()


        if existing_user:

            flash(
                "An account with this email already exists.",
                "warning"
            )

            return redirect(
                url_for("register")
            )


        # ----------------------------------------------------
        # HASH PASSWORD
        # ----------------------------------------------------

        hashed_password = generate_password_hash(
            password
        )


        # ----------------------------------------------------
        # CREATE USER
        # ----------------------------------------------------

        new_user = User(

            name=name,

            email=email,

            password=hashed_password,

            role=role

        )


        db.session.add(
            new_user
        )

        db.session.commit()


        flash(
            "Registration successful. Please login.",
            "success"
        )


        return redirect(
            url_for("login")
        )


    return render_template(
        "register.html"
    )


# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if current_user.is_authenticated:

        return redirect(
            url_for("dashboard")
        )


    if request.method == "POST":

        # ----------------------------------------------------
        # GET LOGIN DATA
        # ----------------------------------------------------

        email = request.form.get(
            "email",
            ""
        ).strip().lower()


        password = request.form.get(
            "password",
            ""
        )


        # ----------------------------------------------------
        # FIND USER
        # ----------------------------------------------------

        user = User.query.filter_by(
            email=email
        ).first()


        # ----------------------------------------------------
        # VERIFY PASSWORD
        # ----------------------------------------------------

        if (
            user
            and
            check_password_hash(
                user.password,
                password
            )
        ):

            login_user(
                user
            )


            flash(
                "Welcome back!",
                "success"
            )


            return redirect(
                url_for("dashboard")
            )


        flash(
            "Invalid email or password.",
            "danger"
        )


    return render_template(
        "login.html"
    )


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/dashboard")
@login_required
def dashboard():

    return render_template(
        "dashboard.html",
        user=current_user
    )


# ============================================================
# DATASET MANAGEMENT PAGE
# ADMIN ONLY
# ============================================================

@app.route("/datasets")
@login_required
@role_required("Admin")
def datasets():

    return render_template(
        "dataset_upload.html"
    )


# ============================================================
# DATASET UPLOAD + ANALYSIS
# ADMIN ONLY
# ============================================================

@app.route(
    "/upload-dataset",
    methods=["POST"]
)
@login_required
@role_required("Admin")
def upload_dataset():

    # ========================================================
    # 1. GET DATASET TYPE
    # ========================================================

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

        return redirect(
            url_for("datasets")
        )


    # ========================================================
    # 2. CHECK FILE
    # ========================================================

    if "dataset" not in request.files:

        flash(
            "No dataset file was selected.",
            "danger"
        )

        return redirect(
            url_for("datasets")
        )


    file = request.files[
        "dataset"
    ]


    if file.filename == "":

        flash(
            "Please select a CSV file.",
            "danger"
        )

        return redirect(
            url_for("datasets")
        )


    # ========================================================
    # 3. CHECK FILE FORMAT
    # ========================================================

    filename = secure_filename(
        file.filename
    )


    if not allowed_file(filename):

        flash(
            "Only CSV files are currently supported.",
            "danger"
        )

        return redirect(
            url_for("datasets")
        )


    # ========================================================
    # 4. CREATE FILE PATH
    # ========================================================

    upload_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    # ========================================================
    # 5. SAVE FILE
    # ========================================================

    try:

        file.save(
            upload_path
        )

    except Exception as e:

        flash(
            f"Could not save the uploaded file: {str(e)}",
            "danger"
        )

        return redirect(
            url_for("datasets")
        )


    # ========================================================
    # 6. READ CSV
    # ========================================================

    try:

        df = read_csv_safely(
            upload_path
        )


    except pd.errors.EmptyDataError:

        flash(
            "The uploaded CSV file is empty.",
            "danger"
        )

        return redirect(
            url_for("datasets")
        )


    except pd.errors.ParserError:

        flash(
            "The CSV file has an invalid structure and could not be read.",
            "danger"
        )

        return redirect(
            url_for("datasets")
        )


    except Exception as e:

        flash(
            f"Could not process the CSV file: {str(e)}",
            "danger"
        )

        return redirect(
            url_for("datasets")
        )


    # ========================================================
    # 7. CHECK EMPTY DATAFRAME
    # ========================================================

    if df.empty:

        flash(
            "The uploaded CSV contains no data.",
            "danger"
        )

        return redirect(
            url_for("datasets")
        )


    # ========================================================
    # 8. CLEAN COLUMN NAMES
    # ========================================================

    df.columns = [

        normalize_column_name(column)

        for column in df.columns

    ]


    # ========================================================
    # 9. BASIC DATASET STATISTICS
    # ========================================================

    total_rows = len(df)


    total_columns = len(
        df.columns
    )


    missing_values = int(
        df.isnull()
        .sum()
        .sum()
    )


    duplicate_rows = int(
        df.duplicated()
        .sum()
    )


    # ========================================================
    # 10. AUTOMATIC COLUMN MAPPING
    # ========================================================

    column_mapping = (
        detect_column_mapping(df)
    )


    mapped_count = (
        count_mapped_fields(
            column_mapping
        )
    )


    total_mapping_fields = (
        len(COLUMN_KEYWORDS)
    )


    # ========================================================
    # 11. SAVE RAW COPY
    # ========================================================

    raw_filename = (
        f"{dataset_type}_{filename}"
    )


    raw_path = os.path.join(
        RAW_FOLDER,
        raw_filename
    )


    try:

        df.to_csv(
            raw_path,
            index=False
        )

    except Exception as e:

        flash(
            "Dataset analyzed but raw copy could not be saved.",
            "warning"
        )


    # ========================================================
    # 12. DATA CLEANING STATUS
    # ========================================================
    #
    # We are NOT silently changing the uploaded dataset.
    #
    # The raw dataset is preserved.
    # Cleaning will be performed as the next processing stage.
    # ========================================================

    cleaned_success = False

    cleaned_missing_values = 0

    cleaned_duplicate_rows = 0


    # ========================================================
    # 13. FORECASTING READINESS
    # ========================================================

    has_date = (
        column_mapping.get("date")
        is not None
    )


    has_demand = (
        column_mapping.get("demand")
        is not None
    )


    has_sales = (
        column_mapping.get("sales")
        is not None
    )


    has_product = (
        column_mapping.get("product")
        is not None
    )


    forecasting_ready = (
        has_date
        and
        (has_demand or has_sales)
    )


    # ========================================================
    # 14. GET COLUMN NAMES
    # ========================================================

    columns = list(
        df.columns
    )


    # ========================================================
    # 15. CREATE PREVIEW
    # ========================================================

    preview_df = df.head(
        10
    )


    preview_html = preview_df.to_html(

        classes=(
            "table "
            "table-hover "
            "table-bordered"
        ),

        index=False

    )


    # ========================================================
    # 16. SHOW ANALYSIS
    # ========================================================

    return render_template(

        "dataset_upload.html",

        upload_success=True,

        filename=filename,

        dataset_type=dataset_type,

        total_rows=total_rows,

        total_columns=total_columns,

        missing_values=missing_values,

        duplicate_rows=duplicate_rows,

        columns=columns,

        preview_html=preview_html,

        column_mapping=column_mapping,

        mapped_count=mapped_count,

        total_mapping_fields=total_mapping_fields,

        cleaned_success=cleaned_success,

        cleaned_missing_values=cleaned_missing_values,

        cleaned_duplicate_rows=cleaned_duplicate_rows,

        forecasting_ready=forecasting_ready

    )


# ============================================================
# SALES FORECASTING PAGE
# ============================================================

@app.route("/forecasting")
@login_required
def forecasting():

    return render_template(
        "forecasting.html"
    )


# ============================================================
# INVENTORY PAGE
# ============================================================

@app.route("/inventory")
@login_required
def inventory():

    # --------------------------------------------------------
    # IMPORTANT
    #
    # This function does NOT create fake values.
    #
    # It checks whether an actual inventory CSV has been
    # uploaded.
    # --------------------------------------------------------

    inventory_data = (
        calculate_inventory_statistics()
    )


    return render_template(

        "inventory.html",

        inventory_available=
            inventory_data[
                "inventory_available"
            ],

        total_products=
            inventory_data[
                "total_products"
            ],

        low_stock=
            inventory_data[
                "low_stock"
            ],

        overstock=
            inventory_data[
                "overstock"
            ],

        reorder_required=
            inventory_data[
                "reorder_required"
            ],

        average_daily_demand=
            inventory_data[
                "average_daily_demand"
            ],

        lead_time=
            inventory_data[
                "lead_time"
            ],

        safety_stock=
            inventory_data[
                "safety_stock"
            ],

        reorder_point=
            inventory_data[
                "reorder_point"
            ],

        inventory_message=
            inventory_data[
                "inventory_message"
            ]

    )


# ============================================================
# ANALYTICS PAGE
# ============================================================

@app.route("/analytics")
@login_required
def analytics():

    return render_template(
        "analytics.html"
    )


# ============================================================
# SMART ALERTS PAGE
# ============================================================

@app.route("/alerts")
@login_required
def alerts():

    return render_template(
        "alerts.html"
    )


# ============================================================
# REPORTS PAGE
# ============================================================

@app.route("/reports")
@login_required
def reports():

    return render_template(
        "reports.html"
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