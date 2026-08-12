import os
import pandas as pd
from flask import request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

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

from config import Config
from database.models import db, User


# =========================================================
# FLASK APPLICATION
# =========================================================

app = Flask(__name__)

app.config.from_object(Config)


# =========================================================
# DATABASE
# =========================================================

db.init_app(app)


# =========================================================
# LOGIN MANAGER
# =========================================================

login_manager = LoginManager()

login_manager.init_app(app)

login_manager.login_view = "login"

login_manager.login_message = (
    "Please login to access this page."
)


@login_manager.user_loader
def load_user(user_id):

    return User.query.get(int(user_id))


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    if current_user.is_authenticated:

        return redirect(
            url_for("dashboard")
        )

    return redirect(
        url_for("login")
    )


# =========================================================
# REGISTER
# =========================================================

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


        # -------------------------------------------------
        # VALIDATION
        # -------------------------------------------------

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


        # -------------------------------------------------
        # CHECK EXISTING USER
        # -------------------------------------------------

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


        # -------------------------------------------------
        # HASH PASSWORD
        # -------------------------------------------------

        hashed_password = generate_password_hash(
            password
        )


        # -------------------------------------------------
        # CREATE USER
        # -------------------------------------------------

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


        return redirect(
            url_for("login")
        )


    return render_template(
        "register.html"
    )


# =========================================================
# LOGIN
# =========================================================

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


        user = User.query.filter_by(
            email=email
        ).first()


        # -------------------------------------------------
        # VERIFY LOGIN
        # -------------------------------------------------

        if user and check_password_hash(
            user.password,
            password
        ):

            login_user(user)


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


# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
@login_required
def dashboard():

    return render_template(
        "dashboard.html",
        user=current_user
    )


# =========================================================
# LOGOUT
# =========================================================

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


# =========================================================
# CREATE DATABASE TABLES
# =========================================================

with app.app_context():

    db.create_all()


# =========================================================
# START APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )


# ============================================================
# DATASET UPLOAD CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RAW_FOLDER, exist_ok=True)
os.makedirs(CLEANED_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Maximum upload size: 100 MB
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


ALLOWED_EXTENSIONS = {"csv"}


def allowed_file(filename):
    """
    Check whether the uploaded file is a CSV file.
    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# ============================================================
# DATASET MANAGEMENT PAGE
# ============================================================

@app.route("/datasets")
def datasets():

    return render_template(
        "dataset_upload.html"
    )

# ============================================================
# DATASET UPLOAD
# ============================================================

@app.route("/upload-dataset", methods=["POST"])
def upload_dataset():

    # --------------------------------------------------------
    # Get dataset type
    # --------------------------------------------------------

    dataset_type = request.form.get(
        "dataset_type",
        ""
    ).strip()


    # --------------------------------------------------------
    # Validate dataset type
    # --------------------------------------------------------

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

        return render_template(
            "dataset_upload.html",
            error="Please select a valid dataset type."
        )


    # --------------------------------------------------------
    # Check file
    # --------------------------------------------------------

    if "dataset" not in request.files:

        return render_template(
            "dataset_upload.html",
            error="No dataset file was selected."
        )


    file = request.files["dataset"]


    if file.filename == "":

        return render_template(
            "dataset_upload.html",
            error="Please select a CSV file."
        )


    # --------------------------------------------------------
    # Check extension
    # --------------------------------------------------------

    if not allowed_file(file.filename):

        return render_template(
            "dataset_upload.html",
            error="Invalid file format. Please upload a CSV file."
        )


    # --------------------------------------------------------
    # Secure filename
    # --------------------------------------------------------

    filename = secure_filename(
        file.filename
    )


    # --------------------------------------------------------
    # Save uploaded file
    # --------------------------------------------------------

    upload_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )


    try:

        file.save(upload_path)

    except Exception as e:

        return render_template(
            "dataset_upload.html",
            error=f"Could not save the uploaded file: {str(e)}"
        )


    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            upload_path,
            low_memory=False
        )

    except UnicodeDecodeError:

        try:

            df = pd.read_csv(
                upload_path,
                encoding="latin1",
                low_memory=False
            )

        except Exception as e:

            return render_template(
                "dataset_upload.html",
                error=f"Unable to read CSV encoding: {str(e)}"
            )

    except pd.errors.EmptyDataError:

        return render_template(
            "dataset_upload.html",
            error="The uploaded CSV file is empty."
        )

    except pd.errors.ParserError as e:

        return render_template(
            "dataset_upload.html",
            error=f"CSV parsing error. Please check the file structure: {str(e)}"
        )

    except Exception as e:

        return render_template(
            "dataset_upload.html",
            error=f"Could not process the CSV file: {str(e)}"
        )


    # --------------------------------------------------------
    # Check empty dataset
    # --------------------------------------------------------

    if df.empty:

        return render_template(
            "dataset_upload.html",
            error="The CSV file contains no usable rows."
        )


    # --------------------------------------------------------
    # Clean column names temporarily
    # --------------------------------------------------------

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]


    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    rows = len(df)

    columns = len(df.columns)

    missing_values = int(
        df.isnull().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )


    # --------------------------------------------------------
    # Dataset preview
    # --------------------------------------------------------

    preview_df = df.head(10)

    preview_html = preview_df.to_html(
        classes="table table-bordered table-hover",
        index=False
    )


    # --------------------------------------------------------
    # Save a copy into raw folder
    # --------------------------------------------------------

    raw_filename = (
        dataset_type
        + "_"
        + filename
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

    except Exception:
        pass


    # --------------------------------------------------------
    # Prepare result
    # --------------------------------------------------------

    dataset_info = {

        "filename": filename,

        "dataset_type": dataset_type,

        "rows": rows,

        "columns": columns,

        "missing": missing_values,

        "duplicates": duplicate_rows,

        "column_names": list(df.columns),

        "preview": preview_html

    }


    return render_template(

        "dataset_upload.html",

        success="Dataset uploaded and analyzed successfully.",

        dataset_info=dataset_info

    )


