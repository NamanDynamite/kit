"""
SQLAlchemy database setup and ORM models for KitchenMind.
"""

from sqlalchemy import (
    create_engine, Column, String, Integer, Float, Boolean, DateTime, Text, Enum, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, backref
import enum
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://kitchenmind:password@localhost:5432/kitchenmind"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Enums
class DietaryPreferenceEnum(enum.Enum):
    VEG = "VEG"
    NON_VEG = "NON_VEG"

class PlanStatusEnum(enum.Enum):
    pending = "pending"
    confirmed = "confirmed"
    cancelled = "cancelled"

# Tables
class Role(Base):
    __tablename__ = "roles"
    role_id = Column(String, primary_key=True)  # 'user', 'trainer', 'admin' only
    role_name = Column(String, nullable=False)
    description = Column(String)
    user = relationship("User", back_populates="role")

class User(Base):
    __tablename__ = "user"
    user_id = Column(String, primary_key=True)
    name = Column(String)
    email = Column(String, unique=True)
    login_identifier = Column(String, unique=True)
    password_hash = Column(String)
    auth_type = Column(String)
    otp_hash = Column(String)
    otp_expires_at = Column(DateTime)
    otp_verified = Column(Boolean, default=False)
    role_id = Column(String, ForeignKey("roles.role_id"))
    dietary_preference = Column(Enum(DietaryPreferenceEnum))
    rating_score = Column(Float, default=0.0)
    credit = Column(Float, default=0.0)
    created_at = Column(DateTime)
    last_login_at = Column(DateTime)
    role = relationship("Role", back_populates="user")
    is_super_admin = Column(Boolean, default=False)
    created_by = Column(String)  # user_id of creator (admin)
    admin_action_type = Column(String)  # last admin action type (if admin)
    admin_action_target_type = Column(String)  # last admin action target type
    admin_action_target_id = Column(String)  # last admin action target id
    admin_action_description = Column(Text)  # last admin action description
    admin_action_created_at = Column(DateTime)  # last admin action timestamp
    sessions = relationship("Session", back_populates="user")
    feedbacks = relationship("Feedback", back_populates="user")
    point_logs = relationship("PointLog", back_populates="user")
    token_transactions = relationship("TokenTransaction", back_populates="user")
    event_plans = relationship("EventPlan", back_populates="user")
    recipes = relationship("Recipe", back_populates="creator")


class Session(Base):
    __tablename__ = "sessions"
    session_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("user.user_id"))
    created_at = Column(DateTime)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    ip_address = Column(String)
    user_agent = Column(String)
    user = relationship("User", back_populates="sessions")


class Recipe(Base):
    __tablename__ = "recipes"
    recipe_id = Column(String, primary_key=True)
    dish_name = Column(String)
    servings = Column(Integer, nullable=False, default=1)
    current_version_id = Column(String, nullable=True)  # Removed FK to break circular dependency
    created_by = Column(String, ForeignKey("user.user_id"))
    is_published = Column(Boolean, default=False)
    created_at = Column(DateTime)
    creator = relationship("User", back_populates="recipes")
    versions = relationship("RecipeVersion", back_populates="recipe")
    feedbacks = relationship("Feedback", back_populates="recipe")
    recipe_score = relationship("RecipeScore", uselist=False, back_populates="recipe")
    token_transactions = relationship("TokenTransaction", back_populates="recipe")

class RecipeVersion(Base):
    __tablename__ = "recipe_versions"
    version_id = Column(String, primary_key=True)
    recipe_id = Column(String, ForeignKey("recipes.recipe_id"))
    submitted_by = Column(String, ForeignKey("user.user_id"))
    submitted_at = Column(DateTime)
    status = Column(String)
    validator_confidence = Column(Float)
    base_servings = Column(Integer)
    avg_rating = Column(Float)
    recipe = relationship("Recipe", back_populates="versions")
    ingredients = relationship("Ingredient", back_populates="version")
    steps = relationship("Step", back_populates="version")
    validations = relationship("Validation", back_populates="version")

class Ingredient(Base):
    __tablename__ = "ingredients"
    ingredient_id = Column(String, primary_key=True)
    version_id = Column(String, ForeignKey("recipe_versions.version_id"))
    name = Column(String)
    quantity = Column(Float)
    unit = Column(String)
    notes = Column(String)
    version = relationship("RecipeVersion", back_populates="ingredients")

class Step(Base):
    __tablename__ = "steps"
    step_id = Column(String, primary_key=True)
    version_id = Column(String, ForeignKey("recipe_versions.version_id"))
    step_order = Column(Integer)
    instruction = Column(Text)
    minutes = Column(Integer)
    version = relationship("RecipeVersion", back_populates="steps")

class Validation(Base):
    __tablename__ = "validations"
    validation_id = Column(String, primary_key=True)
    version_id = Column(String, ForeignKey("recipe_versions.version_id"))
    validated_at = Column(DateTime)
    approved = Column(Boolean)
    feedback = Column(Text)
    version = relationship("RecipeVersion", back_populates="validations")

class Feedback(Base):
    __tablename__ = "feedbacks"
    feedback_id = Column(String, primary_key=True)
    recipe_id = Column(String, ForeignKey("recipes.recipe_id"))
    user_id = Column(String, ForeignKey("user.user_id"))
    created_at = Column(DateTime)
    rating = Column(Integer)
    comment = Column(Text)
    flagged = Column(Boolean, default=False)
    is_revised = Column(Boolean, default=False)
    revised_at = Column(DateTime)
    recipe = relationship("Recipe", back_populates="feedbacks")
    user = relationship("User", back_populates="feedbacks")

class TokenTransaction(Base):
    __tablename__ = "token_transactions"
    tx_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("user.user_id"))
    recipe_id = Column(String, ForeignKey("recipes.recipe_id"))
    date = Column(DateTime)
    tokens = Column(Float)
    reason = Column(String)
    related_id = Column(String)
    user = relationship("User", back_populates="token_transactions")
    recipe = relationship("Recipe", back_populates="token_transactions")

class PointLog(Base):
    __tablename__ = "point_logs"
    log_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("user.user_id"))
    activity_type = Column(String)
    quantity = Column(Integer)
    points = Column(Integer)
    created_at = Column(DateTime)
    user = relationship("User", back_populates="point_logs")

class RecipeScore(Base):
    __tablename__ = "recipe_scores"
    score_id = Column(String, primary_key=True)
    recipe_id = Column(String, ForeignKey("recipes.recipe_id"))
    user_rating_score = Column(Float)
    validator_confidence_score = Column(Float)
    ingredient_authenticity_score = Column(Float)
    serving_scalability_score = Column(Float)
    popularity_score = Column(Float)
    ai_confidence_score = Column(Float)
    final_score = Column(Float)
    calculated_at = Column(DateTime)
    recipe = relationship("Recipe", back_populates="recipe_score")



class EventPlan(Base):
    __tablename__ = "event_plans"
    event_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("user.user_id"))
    event_date = Column(DateTime)
    guest_count = Column(Integer)
    budget = Column(Float)
    preferences = Column(Text)
    plan_status = Column(Enum(PlanStatusEnum))
    user = relationship("User", back_populates="event_plans")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    Base.metadata.create_all(bind=engine)
