# ============================================================
# RETAIL DEMAND FORECASTING SYSTEM
# Flask Backend
# ============================================================

import os
import re
import pandas as pd
import numpy as np

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
# OPTIONAL DATA CLEANING SERVICE
# ============================================================

try:

    from services.data_cleaner import clean_dataset

except ImportError:

    clean_dataset = None


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
        "."
        in filename
        and
        filename.rsplit(
            ".",
            1
        )[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# AUTOMATIC COLUMN MAPPING ENGINE
# ============================================================

def automatic_column_mapping(columns):

    """
    Automatically identifies important retail dataset
    columns based on common column-name variations.

    Detects:

    - Date
    - Order ID
    - Order Name
    - Transaction ID
    - Invoice ID
    - Product
    - Sales
    - Demand
    - Store
    - Price
    - Discount
    - Promotion
    - Category
    - Inventory
    - Supplier
    - Lead Time
    - Temperature
    - Holiday
    - Festival

    The system is designed to handle datasets where
    column names may be written in different formats.
    """

    # ========================================================
    # NORMALIZE COLUMN NAMES
    # ========================================================

    normalized_columns = {}


    for column in columns:

        original = str(
            column
        ).strip()


        normalized = re.sub(
            r"[^a-zA-Z0-9]",
            "_",
            original.lower()
        )


        normalized = re.sub(
            r"_+",
            "_",
            normalized
        ).strip("_")


        normalized_columns[
            original
        ] = normalized


    # ========================================================
    # POSSIBLE COLUMN NAMES
    # ========================================================

    column_patterns = {


        # ====================================================
        # DATE
        # ====================================================

        "date": [

            "date",
            "order_date",
            "sales_date",
            "sale_date",
            "transaction_date",
            "invoice_date",
            "purchase_date",
            "delivery_date",
            "shipping_date",
            "created_date",
            "datetime",
            "timestamp"

        ],


        # ====================================================
        # ORDER ID
        # ====================================================

        "order_id": [

            "order_id",
            "orderid",
            "order_no",
            "order_number",
            "order_num",
            "sales_order",
            "sales_order_id",
            "sales_order_number",
            "order_code"

        ],


        # ====================================================
        # ORDER NAME
        # ====================================================

        "order_name": [

            "order_name",
            "ordername",
            "order_title",
            "order_description"

        ],


        # ====================================================
        # TRANSACTION ID
        # ====================================================

        "transaction_id": [

            "transaction_id",
            "transactionid",
            "transaction_no",
            "transaction_number",
            "transaction_num",
            "transaction_code",
            "txn_id",
            "txn_number"

        ],


        # ====================================================
        # INVOICE ID
        # ====================================================

        "invoice_id": [

            "invoice_id",
            "invoiceid",
            "invoice_no",
            "invoice_number",
            "invoice_num",
            "invoice_code"

        ],


        # ====================================================
        # PRODUCT
        # ====================================================

        "product": [

            "product",
            "product_name",
            "product_id",
            "item",
            "item_name",
            "item_id",
            "sku",
            "sku_id",
            "product_code",
            "item_code"

        ],


        # ====================================================
        # SALES
        # ====================================================

        "sales": [

            "sales",
            "sale",
            "sales_amount",
            "sale_amount",
            "revenue",
            "total_sales",
            "sales_value",
            "sales_revenue",
            "turnover",
            "total_revenue"

        ],


        # ====================================================
        # DEMAND / QUANTITY
        # ====================================================

        "demand": [

            "demand",
            "units_sold",
            "unit_sold",
            "quantity",
            "qty",
            "sales_quantity",
            "sales_qty",
            "units",
            "units_demanded",
            "demand_quantity",
            "demand_qty"

        ],


        # ====================================================
        # STORE
        # ====================================================

        "store": [

            "store",
            "store_id",
            "store_name",
            "store_code",
            "shop",
            "shop_id",
            "shop_name",
            "branch",
            "branch_id",
            "branch_name",
            "location"

        ],


        # ====================================================
        # PRICE
        # ====================================================

        "price": [

            "price",
            "unit_price",
            "selling_price",
            "sale_price",
            "product_price",
            "item_price",
            "cost",
            "unit_cost",
            "selling_cost"

        ],


        # ====================================================
        # DISCOUNT
        # ====================================================

        "discount": [

            "discount",
            "discount_rate",
            "discount_percentage",
            "discount_percent",
            "discount_amount",
            "markdown"

        ],


        # ====================================================
        # PROMOTION
        # ====================================================

        "promotion": [

            "promotion",
            "promotion_status",
            "promo",
            "promotional",
            "campaign",
            "offer",
            "promotion_flag",
            "promo_flag"

        ],


        # ====================================================
        # CATEGORY
        # ====================================================

        "category": [

            "category",
            "product_category",
            "item_category",
            "product_type",
            "department",
            "segment"

        ],


        # ====================================================
        # INVENTORY
        # ====================================================

        "inventory": [

            "inventory",
            "stock",
            "current_stock",
            "stock_level",
            "stock_quantity",
            "inventory_level",
            "available_stock",
            "available_inventory"

        ],


        # ====================================================
        # SUPPLIER
        # ====================================================

        "supplier": [

            "supplier",
            "supplier_id",
            "supplier_name",
            "supplier_code",
            "vendor",
            "vendor_id",
            "vendor_name",
            "vendor_code"

        ],


        # ====================================================
        # LEAD TIME
        # ====================================================

        "lead_time": [

            "lead_time",
            "lead_days",
            "delivery_time",
            "delivery_days",
            "supplier_lead_time",
            "supplier_delivery_time"

        ],


        # ====================================================
        # TEMPERATURE
        # ====================================================

        "temperature": [

            "temperature",
            "temp",
            "avg_temperature",
            "average_temperature",
            "mean_temperature"

        ],


        # ====================================================
        # HOLIDAY
        # ====================================================

        "holiday": [

            "holiday",
            "holiday_flag",
            "is_holiday",
            "holiday_status",
            "holiday_indicator"

        ],


        # ====================================================
        # FESTIVAL
        # ====================================================

        "festival": [

            "festival",
            "festival_flag",
            "is_festival",
            "festival_name",
            "festival_status"

        ]

    }


    # ========================================================
    # FIND MATCHING COLUMNS
    # ========================================================

    mapping = {}


    # ========================================================
    # PREVENT SAME COLUMN BEING USED MULTIPLE TIMES
    # ========================================================

    used_columns = set()


    # ========================================================
    # PROCESS EACH FIELD
    # ========================================================

    for field, possible_names in column_patterns.items():

        mapping[field] = None


        # ====================================================
        # EXACT MATCH
        # ====================================================

        for original, normalized in normalized_columns.items():

            if original in used_columns:

                continue


            if normalized in possible_names:

                mapping[field] = original

                used_columns.add(
                    original
                )

                break


        # ====================================================
        # PARTIAL MATCH
        # ====================================================

        if mapping[field] is None:

            for original, normalized in normalized_columns.items():

                if original in used_columns:

                    continue


                for possible_name in possible_names:

                    if (
                        possible_name in normalized
                        or
                        normalized in possible_name
                    ):

                        mapping[field] = original

                        used_columns.add(
                            original
                        )

                        break


                if mapping[field] is not None:

                    break


    # ========================================================
    # ORDER-RELATED SUMMARY
    # ========================================================

    order_columns = {

        "order_id":
            mapping.get("order_id"),

        "order_name":
            mapping.get("order_name"),

        "transaction_id":
            mapping.get("transaction_id"),

        "invoice_id":
            mapping.get("invoice_id")

    }


    # ========================================================
    # CHECK WHETHER ORDER INFORMATION EXISTS
    # ========================================================

    order_detected = any(

        value is not None

        for value in order_columns.values()

    )


    mapping["order_detected"] = (
        order_detected
    )


    mapping["order_columns"] = (
        order_columns
    )


    # ========================================================
    # RETURN MAPPING
    # ========================================================

    return mapping


# ============================================================
# DATABASE
# ============================================================

db.init_app(
    app
)


# ============================================================
# LOGIN MANAGER
# ============================================================

login_manager = LoginManager()

login_manager.init_app(
    app
)

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


            user_role = (
                current_user.role
            )


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


        # ====================================================
        # NAME VALIDATION
        # ====================================================

        if not name:

            flash(
                "Please enter your name.",
                "danger"
            )

            return redirect(
                url_for("register")
            )


        # ====================================================
        # EMAIL VALIDATION
        # ====================================================

        if not email:

            flash(
                "Please enter your email.",
                "danger"
            )

            return redirect(
                url_for("register")
            )


        # ====================================================
        # PASSWORD VALIDATION
        # ====================================================

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


        # ====================================================
        # ALLOWED ROLES
        # ====================================================

        allowed_roles = {

            "Admin",
            "Inventory Manager",
            "Business Analyst"

        }


        if role not in allowed_roles:

            role = "Business Analyst"


        # ====================================================
        # CHECK EXISTING USER
        # ====================================================

        existing_user = (
            User.query
            .filter_by(
                email=email
            )
            .first()
        )


        if existing_user:

            flash(
                "An account with this email already exists.",
                "warning"
            )

            return redirect(
                url_for("register")
            )


        # ====================================================
        # HASH PASSWORD
        # ====================================================

        hashed_password = (
            generate_password_hash(
                password
            )
        )


        # ====================================================
        # CREATE USER
        # ====================================================

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

        email = request.form.get(
            "email",
            ""
        ).strip().lower()


        password = request.form.get(
            "password",
            ""
        )


        user = (
            User.query
            .filter_by(
                email=email
            )
            .first()
        )


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
# DATASET MANAGEMENT
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
# ============================================================

@app.route(
    "/upload-dataset",
    methods=["POST"]
)
@login_required
@role_required("Admin")
def upload_dataset():

    # ========================================================
    # DATASET TYPE
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
    # FILE CHECK
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
    # FILE FORMAT
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
    # UNIQUE FILE NAME
    # ========================================================

    base_name, extension = (
        os.path.splitext(
            filename
        )
    )


    upload_filename = (
        filename
    )


    counter = 1


    while os.path.exists(
        os.path.join(
            UPLOAD_FOLDER,
            upload_filename
        )
    ):

        upload_filename = (

            f"{base_name}_"
            f"{counter}"
            f"{extension}"

        )

        counter += 1


    filename = (
        upload_filename
    )


    upload_path = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    # ========================================================
    # SAVE FILE
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
    # READ CSV
    # ========================================================

    try:

        try:

            df = pd.read_csv(
                upload_path,
                low_memory=False
            )

        except UnicodeDecodeError:

            df = pd.read_csv(
                upload_path,
                encoding="latin1",
                low_memory=False
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
    # EMPTY DATAFRAME CHECK
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
    # CLEAN COLUMN NAMES
    # ========================================================

    df.columns = (

        df.columns
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(
            " ",
            "_",
            regex=False
        )
        .str.replace(
            "-",
            "_",
            regex=False
        )

    )


    # ========================================================
    # AUTOMATIC COLUMN MAPPING
    # ========================================================

    column_mapping = (
        automatic_column_mapping(
            df.columns.tolist()
        )
    )


    # ========================================================
    # DATASET STATISTICS
    # ========================================================

    total_rows = len(
        df
    )


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
    # SAVE RAW DATASET
    # ========================================================

    raw_filename = (

        f"{dataset_type}_"
        f"{filename}"

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
            "Dataset analyzed but raw copy could not be saved: "
            f"{str(e)}",
            "warning"
        )


    # ========================================================
    # DATA CLEANING
    # ========================================================

    cleaned_filename = (

        f"cleaned_"
        f"{dataset_type}_"
        f"{filename}"

    )


    cleaned_path = os.path.join(
        CLEANED_FOLDER,
        cleaned_filename
    )


    cleaned_success = False


    try:

        if clean_dataset is not None:

            cleaned_result = (
                clean_dataset(
                    df.copy()
                )
            )


            if isinstance(
                cleaned_result,
                pd.DataFrame
            ):

                cleaned_df = (
                    cleaned_result
                )

            else:

                cleaned_df = (
                    df.copy()
                )

        else:

            cleaned_df = (
                df.copy()
            )


        cleaned_df.to_csv(
            cleaned_path,
            index=False
        )


        cleaned_success = True


    except Exception as e:

        cleaned_df = (
            df.copy()
        )


        flash(
            "Dataset uploaded successfully, "
            "but automatic cleaning could not be completed: "
            f"{str(e)}",
            "warning"
        )


    # ========================================================
    # CLEANED DATASET STATISTICS
    # ========================================================

    cleaned_missing_values = int(

        cleaned_df.isnull()
        .sum()
        .sum()

    )


    cleaned_duplicate_rows = int(

        cleaned_df.duplicated()
        .sum()

    )


    # ========================================================
    # COLUMN LIST
    # ========================================================

    columns = list(
        df.columns
    )


    # ========================================================
    # PREVIEW
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
    # MAPPING COUNTS
    # ========================================================

    mapped_count = sum(

        1

        for key, value
        in column_mapping.items()

        if (
            key not in {
                "order_detected",
                "order_columns"
            }
            and
            value is not None
        )

    )


    total_mapping_fields = (

        len(column_mapping)
        - 2

    )


    # ========================================================
    # ORDER INFORMATION
    # ========================================================

    order_detected = (
        column_mapping.get(
            "order_detected",
            False
        )
    )


    order_columns = (
        column_mapping.get(
            "order_columns",
            {}
        )
    )


    # ========================================================
    # FORECASTING READINESS
    # ========================================================

    forecasting_ready = (

        column_mapping.get(
            "date"
        ) is not None

        and

        (

            column_mapping.get(
                "demand"
            ) is not None

            or

            column_mapping.get(
                "sales"
            ) is not None

        )

    )


    # ========================================================
    # SHOW RESULTS
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

        total_mapping_fields=(
            total_mapping_fields
        ),

        cleaned_success=(
            cleaned_success
        ),

        cleaned_missing_values=(
            cleaned_missing_values
        ),

        cleaned_duplicate_rows=(
            cleaned_duplicate_rows
        ),

        cleaned_filename=(
            cleaned_filename
        ),

        forecasting_ready=(
            forecasting_ready
        ),

        order_detected=(
            order_detected
        ),

        order_columns=(
            order_columns
        )

    )


# ============================================================
# FORECASTING PAGE
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

    # ========================================================
    # DEFAULT VALUES
    # ========================================================

    total_products = 0

    low_stock = 0

    overstock = 0

    reorder_required = 0

    average_daily_demand = 0

    average_lead_time = 0

    safety_stock = 0

    reorder_point = 0

    inventory_data = []

    inventory_available = False


    # ========================================================
    # FIND INVENTORY DATASET
    # ========================================================

    try:

        inventory_files = []


        # ====================================================
        # CHECK CLEANED FOLDER
        # ====================================================

        if os.path.exists(
            CLEANED_FOLDER
        ):

            for file_name in os.listdir(
                CLEANED_FOLDER
            ):

                if (

                    file_name.lower().endswith(
                        ".csv"
                    )

                    and

                    "inventory"
                    in file_name.lower()

                ):

                    inventory_files.append(
                        file_name
                    )


        # ====================================================
        # CHECK RAW FOLDER
        # ====================================================

        if (

            not inventory_files

            and

            os.path.exists(
                RAW_FOLDER
            )

        ):

            for file_name in os.listdir(
                RAW_FOLDER
            ):

                if (

                    file_name.lower().endswith(
                        ".csv"
                    )

                    and

                    "inventory"
                    in file_name.lower()

                ):

                    inventory_files.append(
                        file_name
                    )


        # ====================================================
        # PROCESS INVENTORY DATASET
        # ====================================================

        if inventory_files:

            selected_file = (
                inventory_files[-1]
            )


            cleaned_path = os.path.join(

                CLEANED_FOLDER,

                selected_file

            )


            raw_path = os.path.join(

                RAW_FOLDER,

                selected_file

            )


            if os.path.exists(
                cleaned_path
            ):

                inventory_path = (
                    cleaned_path
                )

            else:

                inventory_path = (
                    raw_path
                )


            # =================================================
            # LOAD CSV
            # =================================================

            df = pd.read_csv(

                inventory_path,

                low_memory=False

            )


            # =================================================
            # CLEAN COLUMN NAMES
            # =================================================

            df.columns = (

                df.columns
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(
                    " ",
                    "_",
                    regex=False
                )
                .str.replace(
                    "-",
                    "_",
                    regex=False
                )

            )


            inventory_available = True


            # =================================================
            # PRODUCT COLUMN
            # =================================================

            product_column = None


            possible_product_columns = [

                "product",
                "product_id",
                "product_name",
                "item",
                "item_id",
                "item_name",
                "sku",
                "sku_id"

            ]


            for column in (
                possible_product_columns
            ):

                if column in df.columns:

                    product_column = (
                        column
                    )

                    break


            if product_column:

                total_products = int(

                    df[
                        product_column
                    ]
                    .nunique()

                )

            else:

                total_products = len(
                    df
                )


            # =================================================
            # INVENTORY / STOCK COLUMN
            # =================================================

            inventory_column = None


            possible_inventory_columns = [

                "inventory",
                "current_stock",
                "stock",
                "stock_level",
                "stock_quantity",
                "inventory_level",
                "available_stock"

            ]


            for column in (
                possible_inventory_columns
            ):

                if column in df.columns:

                    inventory_column = (
                        column
                    )

                    break


            # =================================================
            # DEMAND COLUMN
            # =================================================

            demand_column = None


            possible_demand_columns = [

                "demand",
                "units_sold",
                "unit_sold",
                "quantity",
                "qty",
                "sales_quantity",
                "units",
                "units_demanded"

            ]


            for column in (
                possible_demand_columns
            ):

                if column in df.columns:

                    demand_column = (
                        column
                    )

                    break


            # =================================================
            # LEAD TIME COLUMN
            # =================================================

            lead_time_column = None


            possible_lead_time_columns = [

                "lead_time",
                "lead_days",
                "delivery_time",
                "delivery_days",
                "supplier_lead_time"

            ]


            for column in (
                possible_lead_time_columns
            ):

                if column in df.columns:

                    lead_time_column = (
                        column
                    )

                    break


            # =================================================
            # REORDER THRESHOLD
            # =================================================

            reorder_threshold_column = None


            possible_threshold_columns = [

                "reorder_threshold",
                "reorder_point",
                "minimum_stock",
                "min_stock"

            ]


            for column in (
                possible_threshold_columns
            ):

                if column in df.columns:

                    reorder_threshold_column = (
                        column
                    )

                    break


            # =================================================
            # REORDER QUANTITY
            # =================================================

            reorder_quantity_column = None


            possible_quantity_columns = [

                "reorder_qty",
                "reorder_quantity",
                "order_quantity"

            ]


            for column in (
                possible_quantity_columns
            ):

                if column in df.columns:

                    reorder_quantity_column = (
                        column
                    )

                    break


            # =================================================
            # CONVERT NUMERIC COLUMNS
            # =================================================

            if inventory_column:

                df[
                    inventory_column
                ] = pd.to_numeric(

                    df[
                        inventory_column
                    ],

                    errors="coerce"

                )


            if demand_column:

                df[
                    demand_column
                ] = pd.to_numeric(

                    df[
                        demand_column
                    ],

                    errors="coerce"

                )


            if lead_time_column:

                df[
                    lead_time_column
                ] = pd.to_numeric(

                    df[
                        lead_time_column
                    ],

                    errors="coerce"

                )


            if reorder_threshold_column:

                df[
                    reorder_threshold_column
                ] = pd.to_numeric(

                    df[
                        reorder_threshold_column
                    ],

                    errors="coerce"

                )


            if reorder_quantity_column:

                df[
                    reorder_quantity_column
                ] = pd.to_numeric(

                    df[
                        reorder_quantity_column
                    ],

                    errors="coerce"

                )


            # =================================================
            # INVENTORY ANALYSIS
            # =================================================

            if inventory_column:

                stock_series = (

                    df[
                        inventory_column
                    ]
                    .dropna()

                )


                if len(
                    stock_series
                ) > 0:

                    # =========================================
                    # LOW STOCK
                    # =========================================

                    if reorder_threshold_column:

                        low_stock = int(

                            (

                                df[
                                    inventory_column
                                ]

                                <

                                df[
                                    reorder_threshold_column
                                ]

                            )
                            .sum()

                        )


                    # =========================================
                    # REORDER REQUIRED
                    # =========================================

                    if reorder_threshold_column:

                        reorder_required = int(

                            (

                                df[
                                    inventory_column
                                ]

                                <=

                                df[
                                    reorder_threshold_column
                                ]

                            )
                            .sum()

                        )


                    # =========================================
                    # OVERSTOCK
                    # =========================================

                    if demand_column:

                        demand_values = (

                            df[
                                demand_column
                            ]
                            .dropna()

                        )


                        if len(
                            demand_values
                        ) > 0:

                            average_demand_value = (

                                demand_values.mean()

                            )


                            if (

                                pd.notna(
                                    average_demand_value
                                )

                                and

                                average_demand_value > 0

                            ):

                                overstock_threshold = (

                                    average_demand_value
                                    *
                                    30

                                )


                                overstock = int(

                                    (

                                        df[
                                            inventory_column
                                        ]

                                        >

                                        overstock_threshold

                                    )
                                    .sum()

                                )


            # =================================================
            # AVERAGE DAILY DEMAND
            # =================================================

            if demand_column:

                demand_values = (

                    df[
                        demand_column
                    ]
                    .dropna()

                )


                if len(
                    demand_values
                ) > 0:

                    average_daily_demand = round(

                        float(
                            demand_values.mean()
                        ),

                        2

                    )


            # =================================================
            # LEAD TIME
            # =================================================

            if lead_time_column:

                lead_values = (

                    df[
                        lead_time_column
                    ]
                    .dropna()

                )


                if len(
                    lead_values
                ) > 0:

                    average_lead_time = round(

                        float(
                            lead_values.mean()
                        ),

                        2

                    )


            # =================================================
            # SAFETY STOCK
            # =================================================

            if average_daily_demand > 0:

                safety_stock = round(

                    average_daily_demand
                    *
                    0.20,

                    2

                )


            # =================================================
            # REORDER POINT
            # =================================================

            if (

                average_daily_demand > 0

                and

                average_lead_time > 0

            ):

                reorder_point = round(

                    (

                        average_daily_demand
                        *
                        average_lead_time

                    )

                    +

                    safety_stock,

                    2

                )


            # =================================================
            # PRODUCT-LEVEL INVENTORY TABLE
            # =================================================

            if inventory_column:

                for index, row in (
                    df.head(100).iterrows()
                ):

                    product_name = (
                        "Product"
                    )


                    if product_column:

                        product_name = str(

                            row[
                                product_column
                            ]

                        )


                    current_stock = row[

                        inventory_column

                    ]


                    if pd.isna(
                        current_stock
                    ):

                        current_stock = 0


                    current_stock = float(
                        current_stock
                    )


                    # =========================================
                    # STATUS
                    # =========================================

                    status = (
                        "Healthy"
                    )


                    if (

                        reorder_threshold_column

                        and

                        pd.notna(

                            row[
                                reorder_threshold_column
                            ]

                        )

                    ):

                        threshold = float(

                            row[
                                reorder_threshold_column
                            ]

                        )


                        if current_stock <= threshold:

                            status = (
                                "Reorder Required"
                            )


                        elif current_stock <= (
                            threshold * 1.5
                        ):

                            status = (
                                "Low Stock"
                            )


                    elif current_stock <= 0:

                        status = (
                            "Out of Stock"
                        )


                    # =========================================
                    # RECOMMENDED ORDER
                    # =========================================

                    recommended_order = 0


                    if reorder_quantity_column:

                        actual_reorder_quantity = (

                            row[
                                reorder_quantity_column
                            ]

                        )


                        if pd.notna(
                            actual_reorder_quantity
                        ):

                            recommended_order = round(

                                float(
                                    actual_reorder_quantity
                                ),

                                2

                            )


                    inventory_data.append({

                        "product":
                            product_name,

                        "stock":
                            round(
                                current_stock,
                                2
                            ),

                        "status":
                            status,

                        "recommended_order":
                            recommended_order

                    })


    except Exception as e:

        flash(
            f"Inventory analysis could not be completed: {str(e)}",
            "warning"
        )


    # ========================================================
    # RENDER INVENTORY PAGE
    # ========================================================

    return render_template(

        "inventory.html",

        total_products=
            total_products,

        low_stock=
            low_stock,

        overstock=
            overstock,

        reorder_required=
            reorder_required,

        average_daily_demand=
            average_daily_demand,

        average_lead_time=
            average_lead_time,

        safety_stock=
            safety_stock,

        reorder_point=
            reorder_point,

        inventory_data=
            inventory_data,

        inventory_available=
            inventory_available

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
# SMART ALERTS
# ============================================================

@app.route("/alerts")
@login_required
def alerts():

    return render_template(
        "alerts.html"
    )


# ============================================================
# REPORTS
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