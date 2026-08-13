# ============================================================
# RETAIL DEMAND FORECASTING SYSTEM
# Flask Backend
# ============================================================

import os
import pandas as pd
import re

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


# Create folders automatically

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
# AUTOMATIC COLUMN MAPPING ENGINE
# ============================================================

def automatic_column_mapping(columns):

    """
    Automatically identifies important retail dataset
    columns based on common column-name variations.

    Example:

    Order_Date      -> date
    Product_Name    -> product
    Units_Sold      -> demand
    Selling_Price   -> price
    Store_ID        -> store
    Discount        -> discount
    """

    # --------------------------------------------------------
    # Normalize original column names
    # --------------------------------------------------------

    normalized_columns = {}

    for column in columns:

        original = str(column).strip()

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

        normalized_columns[original] = normalized


    # --------------------------------------------------------
    # Possible names for important retail fields
    # --------------------------------------------------------

    column_patterns = {

        "date": [
            "date",
            "order_date",
            "sales_date",
            "sale_date",
            "transaction_date",
            "invoice_date",
            "purchase_date",
            "datetime",
            "timestamp"
        ],

        "product": [
            "product",
            "product_name",
            "product_id",
            "item",
            "item_name",
            "item_id",
            "sku",
            "sku_id",
            "product_code"
        ],

        "sales": [
            "sales",
            "sale",
            "sales_amount",
            "sale_amount",
            "revenue",
            "total_sales",
            "sales_value",
            "turnover"
        ],

        "demand": [
            "demand",
            "units_sold",
            "unit_sold",
            "quantity",
            "qty",
            "sales_quantity",
            "units",
            "units_demanded"
        ],

        "store": [
            "store",
            "store_id",
            "store_name",
            "shop",
            "shop_id",
            "branch",
            "branch_id",
            "location"
        ],

        "price": [
            "price",
            "unit_price",
            "selling_price",
            "sale_price",
            "product_price",
            "cost",
            "unit_cost"
        ],

        "discount": [
            "discount",
            "discount_rate",
            "discount_percentage",
            "discount_percent",
            "markdown"
        ],

        "promotion": [
            "promotion",
            "promotion_status",
            "promo",
            "promotional",
            "campaign",
            "offer",
            "promotion_flag"
        ],

        "category": [
            "category",
            "product_category",
            "item_category",
            "product_type",
            "department",
            "segment"
        ],

        "inventory": [
            "inventory",
            "stock",
            "current_stock",
            "stock_level",
            "stock_quantity",
            "inventory_level",
            "available_stock"
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
            "lead_days",
            "delivery_time",
            "delivery_days",
            "supplier_lead_time"
        ],

        "temperature": [
            "temperature",
            "temp",
            "avg_temperature",
            "average_temperature"
        ],

        "holiday": [
            "holiday",
            "holiday_flag",
            "is_holiday",
            "holiday_status"
        ],

        "festival": [
            "festival",
            "festival_flag",
            "is_festival",
            "festival_name"
        ]
    }


    # --------------------------------------------------------
    # Find matching columns
    # --------------------------------------------------------

    mapping = {}

    for field, possible_names in column_patterns.items():

        mapping[field] = None


        # ----------------------------------------------------
        # First: exact matching
        # ----------------------------------------------------

        for original, normalized in normalized_columns.items():

            if normalized in possible_names:

                mapping[field] = original

                break


        # ----------------------------------------------------
        # Second: partial matching
        # ----------------------------------------------------

        if mapping[field] is None:

            for original, normalized in normalized_columns.items():

                for possible_name in possible_names:

                    if (
                        possible_name in normalized
                        or
                        normalized in possible_name
                    ):

                        mapping[field] = original

                        break


                if mapping[field] is not None:

                    break


    return mapping


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

            # User is not logged in
            if not current_user.is_authenticated:

                flash(
                    "Please login to continue.",
                    "warning"
                )

                return redirect(
                    url_for("login")
                )


            # Get current user's role
            user_role = current_user.role


            # Check permission
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

        if user and check_password_hash(
            user.password,
            password
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
    # 9. AUTOMATIC COLUMN MAPPING
    # ========================================================

    column_mapping = automatic_column_mapping(
        df.columns.tolist()
    )


    # ========================================================
    # 10. BASIC DATASET STATISTICS
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
            f"Dataset analyzed but raw copy could not be saved: {str(e)}",
            "warning"
        )


    # ========================================================
    # 12. GET COLUMN NAMES
    # ========================================================

    columns = list(
        df.columns
    )


    # ========================================================
    # 13. CREATE PREVIEW
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
    # 14. SHOW ANALYSIS + COLUMN MAPPING
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

        column_mapping=column_mapping

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

    return render_template(
        "inventory.html"
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