# ============================================================
# RETAIL DEMAND FORECASTING SYSTEM
# Database Models
# ============================================================

from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime


# ============================================================
# DATABASE OBJECT
# ============================================================

db = SQLAlchemy()


# ============================================================
# USER MODEL
# ============================================================

class User(UserMixin, db.Model):

    __tablename__ = "users"

    # --------------------------------------------------------
    # PRIMARY KEY
    # --------------------------------------------------------

    id = db.Column(
        db.Integer,
        primary_key=True
    )

    # --------------------------------------------------------
    # USER NAME
    # --------------------------------------------------------

    name = db.Column(
        db.String(100),
        nullable=False
    )

    # --------------------------------------------------------
    # EMAIL
    # --------------------------------------------------------

    email = db.Column(
        db.String(150),
        unique=True,
        nullable=False,
        index=True
    )

    # --------------------------------------------------------
    # PASSWORD
    # --------------------------------------------------------

    password = db.Column(
        db.String(255),
        nullable=False
    )

    # --------------------------------------------------------
    # ROLE
    # --------------------------------------------------------

    role = db.Column(
        db.String(50),
        nullable=False,
        default="Business Analyst"
    )

    # --------------------------------------------------------
    # CREATED DATE
    # --------------------------------------------------------

    created_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    # --------------------------------------------------------
    # UPDATED DATE
    # --------------------------------------------------------

    updated_at = db.Column(
        db.DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

    # --------------------------------------------------------
    # REPRESENTATION
    # --------------------------------------------------------

    def __repr__(self):

        return (
            f"<User "
            f"id={self.id} "
            f"name='{self.name}' "
            f"email='{self.email}' "
            f"role='{self.role}'>"
        )