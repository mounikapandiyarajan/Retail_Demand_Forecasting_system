# ============================================================
# RETAIL DEMAND FORECASTING SYSTEM
# Flask Configuration
# ============================================================

import os

from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.dirname(__file__)
)


# ============================================================
# CONFIGURATION CLASS
# ============================================================

class Config:

    # --------------------------------------------------------
    # SECRET KEY
    # --------------------------------------------------------

    SECRET_KEY = os.getenv(
        "SECRET_KEY",
        "retail-demand-forecasting-secret-key-change-this"
    )

    # --------------------------------------------------------
    # MYSQL DATABASE
    # --------------------------------------------------------

    MYSQL_USER = os.getenv(
        "MYSQL_USER",
        "root"
    )

    MYSQL_PASSWORD = os.getenv(
        "MYSQL_PASSWORD",
        ""
    )

    MYSQL_HOST = os.getenv(
        "MYSQL_HOST",
        "localhost"
    )

    MYSQL_PORT = os.getenv(
        "MYSQL_PORT",
        "3306"
    )

    MYSQL_DATABASE = os.getenv(
        "MYSQL_DATABASE",
        "retail_demand_forecasting"
    )

    # --------------------------------------------------------
    # SQLALCHEMY DATABASE URL
    # --------------------------------------------------------

    SQLALCHEMY_DATABASE_URI = (
        f"mysql+pymysql://"
        f"{MYSQL_USER}:"
        f"{MYSQL_PASSWORD}@"
        f"{MYSQL_HOST}:"
        f"{MYSQL_PORT}/"
        f"{MYSQL_DATABASE}"
    )

    # --------------------------------------------------------
    # DISABLE SQLALCHEMY TRACKING
    # --------------------------------------------------------

    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --------------------------------------------------------
    # FILE UPLOAD LIMIT
    # --------------------------------------------------------

    MAX_CONTENT_LENGTH = 100 * 1024 * 1024