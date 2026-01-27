# --- Recipe Score Calculation Utility ---
def update_recipe_score(db, recipe_id, ai_scores=None, popularity=None, version_id=None):
    """
    Update or create RecipeScore for a recipe.
    By default uses the latest version, but when rating we pass the rated version_id
    so the average reflects that specific version.
    ai_scores: dict with keys 'ingredient_authenticity_score',
        'serving_scalability_score', 'ai_confidence_score' (all 0-5 scale).
    popularity: float (0-5 scale, calculated from user interactions)
    version_id: optional, when provided the score is computed for that version
    """
    from sqlalchemy import func
    from Module.utils_time import get_india_time
    
    # 1. Choose target version: explicit version_id (if provided) else latest
    if version_id:
        target_version = db.query(RecipeVersion).filter(RecipeVersion.version_id == version_id).first()
        if not target_version:
            raise ValueError(f"No recipe version found for version_id={version_id}")
        target_version_id = target_version.version_id
        print(f"[DEBUG] update_recipe_score: recipe_id={recipe_id} target_version_id={target_version_id} (explicit)")
    else:
        from sqlalchemy import desc
        latest_version = db.query(RecipeVersion).filter(RecipeVersion.recipe_id == recipe_id).order_by(desc(RecipeVersion.submitted_at)).first()
        if not latest_version:
            raise ValueError(f"No recipe version found for recipe_id={recipe_id}")
        target_version_id = latest_version.version_id
        print(f"[DEBUG] update_recipe_score: recipe_id={recipe_id} target_version_id={target_version_id} (latest)")
    
    # 2. Calculate rating from LATEST VERSION ONLY (both on 0-5 scale)
    avg_rating = 0.0
    feedback_avg = db.query(func.avg(Feedback.rating)).filter(Feedback.version_id == target_version_id).scalar() or 0.0
    # Ensure feedback_avg is a float (func.avg may return Decimal)
    try:
        feedback_avg = float(feedback_avg)
        # Both feedback and rating are on 0-5 scale, no conversion needed
        avg_rating = max(0, min(5, feedback_avg))
    except Exception:
        avg_rating = 0.0
    print(f"[DEBUG] update_recipe_score: feedback_avg={feedback_avg} avg_rating={avg_rating}")

    # 3. Get or create RecipeScore (one per recipe_version pair - immutable)
    score = db.query(RecipeScore).filter(
        RecipeScore.recipe_id == recipe_id,
        RecipeScore.version_id == target_version_id
    ).first()
    if not score:
        import uuid
        score = RecipeScore(score_id=str(uuid.uuid4()), recipe_id=recipe_id, version_id=target_version_id)
        db.add(score)
        db.flush()  # Ensure score exists in session before calculating final_score
        print(f"[DEBUG] update_recipe_score: created new RecipeScore score_id={score.score_id} for version_id={target_version_id}")
    else:
        print(f"[DEBUG] update_recipe_score: updating existing RecipeScore score_id={score.score_id} for version_id={target_version_id}")

    # 4. Set scores (all on 0-5 scale)
    score.rating = avg_rating  # avg_rating calculated from latest version feedback
    if ai_scores:
        score.ingredient_authenticity_score = ai_scores.get('ingredient_authenticity_score', 0)
        score.serving_scalability_score = ai_scores.get('serving_scalability_score', 0)
        score.ai_confidence_score = ai_scores.get('ai_confidence_score', 0)
    # Only update popularity_score when explicitly provided; otherwise preserve existing
    if popularity is not None:
        score.popularity_score = popularity
    print(f"[DEBUG] update_recipe_score: score.rating={score.rating} ia={score.ingredient_authenticity_score} ss={score.serving_scalability_score} pop={score.popularity_score} ai_conf={score.ai_confidence_score}")

    # 5. Calculate final_score (all scores now on 0-5 scale)
    weights = {
        'rating': 0.2,
        'ingredient_authenticity_score': 0.2,
        'serving_scalability_score': 0.15,
        'popularity_score': 0.1,
        'ai_confidence_score': 0.35
    }
    final = (
        (score.rating or 0) * weights['rating'] +
        (score.ingredient_authenticity_score or 0) * weights['ingredient_authenticity_score'] +
        (score.serving_scalability_score or 0) * weights['serving_scalability_score'] +
        (score.popularity_score or 0) * weights['popularity_score'] +
        (score.ai_confidence_score or 0) * weights['ai_confidence_score']
    )
    score.final_score = final
    score.calculated_at = get_india_time()
    print(f"[DEBUG] update_recipe_score: final_score={score.final_score} calculated_at={score.calculated_at}")
    db.commit()
    db.refresh(score)
    print(f"[DEBUG] update_recipe_score: refreshed score.rating={score.rating} final_score={score.final_score}")
    
    # Note: Each (recipe_id, version_id) pair has its own RecipeScore row (immutable).
    # This preserves version history and prevents retroactive score changes.
    return score

def update_trainer_rating_score(db, trainer_id):
    """
    Calculate and update the trainer's rating_score based on average rating 
    from all feedback received on their recipes (across all versions).
    
    trainer_id: The user_id of the trainer whose rating should be updated
    """
    from sqlalchemy import func
    
    # Calculate average rating from all feedback on recipes created by this trainer
    # Join: User (trainer) -> Recipe -> RecipeVersion -> Feedback
    avg_rating = db.query(func.avg(Feedback.rating)).join(
        RecipeVersion, Feedback.version_id == RecipeVersion.version_id
    ).join(
        Recipe, RecipeVersion.recipe_id == Recipe.recipe_id
    ).filter(
        Recipe.created_by == trainer_id
    ).scalar()
    
    # Get the trainer user object
    trainer = db.query(User).filter(User.user_id == trainer_id).first()
    if not trainer:
        raise ValueError(f"No user found with user_id={trainer_id}")
    
    # Update rating_score (convert to float, handle None case)
    if avg_rating is not None:
        try:
            trainer.rating_score = float(avg_rating)
            # Ensure rating_score is within 0-5 range
            trainer.rating_score = max(0, min(5, trainer.rating_score))
        except Exception:
            trainer.rating_score = 0.0
    else:
        # No feedback yet, set to 0
        trainer.rating_score = 0.0
    
    print(f"[DEBUG] update_trainer_rating_score: trainer_id={trainer_id} avg_rating={avg_rating} rating_score={trainer.rating_score}")
    db.commit()
    db.refresh(trainer)
    
    return trainer.rating_score

"""
SQLAlchemy database setup and ORM models for KitchenMind.
"""

import sqlalchemy as sa
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

from sqlalchemy import create_engine

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,      # Checks connection before use, auto-reconnects
    pool_size=10,            # Number of connections to keep in pool
    max_overflow=20,         # Extra connections allowed above pool_size
    pool_recycle=1800        # Recycle connections every 30 min
)
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
    phone_number = Column(String)
    email = Column(String, unique=True)
    password_hash = Column(String)
    auth_type = Column(String)
    otp_hash = Column(String)
    otp_expires_at = Column(DateTime(timezone=True))
    otp_verified = Column(Boolean, default=False)
    role_id = Column(String, ForeignKey("roles.role_id"))
    dietary_preference = Column(Enum(DietaryPreferenceEnum))
    rating_score = Column(Float, default=0.0)
    credit = Column(Float, default=0.0)
    created_at = Column(DateTime(timezone=True))
    last_login_at = Column(DateTime(timezone=True))
    role = relationship("Role", back_populates="user")
    is_super_admin = Column(Boolean, default=False)
    created_by = Column(String)  # user_id of creator (admin)
    admin_action_type = Column(String)  # last admin action type (if admin)
    admin_action_target_type = Column(String)  # last admin action target type
    admin_action_target_id = Column(String)  # last admin action target id
    admin_action_description = Column(Text)  # last admin action description
    admin_action_created_at = Column(DateTime(timezone=True))  # last admin action timestamp
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
    created_at = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    is_active = Column(Boolean, default=True)
    ip_address = Column(String)
    user_agent = Column(String)
    user = relationship("User", back_populates="sessions")


class Recipe(Base):
    __tablename__ = "recipes"
    __table_args__ = (
        sa.UniqueConstraint(
            "dish_name", "created_by", "is_published",
            name="uq_recipe_dish_creator_published"
        ),
    )
    recipe_id = Column(String, primary_key=True)
    version_id = Column(String, ForeignKey("recipe_versions.version_id"), nullable=True)
    dish_name = Column(String)
    servings = Column(Integer, nullable=False, default=1)
    created_by = Column(String, ForeignKey("user.user_id"))
    is_published = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True))
    creator = relationship("User", back_populates="recipes")
    versions = relationship("RecipeVersion", foreign_keys="RecipeVersion.recipe_id", back_populates="recipe")
    # feedbacks relationship removed; now on RecipeVersion
    recipe_score = relationship("RecipeScore", uselist=False, back_populates="recipe")
    token_transactions = relationship("TokenTransaction", back_populates="recipe")

class RecipeVersion(Base):
    __tablename__ = "recipe_versions"
    version_id = Column(String, primary_key=True)
    recipe_id = Column(String, ForeignKey("recipes.recipe_id"))
    submitted_by = Column(String, ForeignKey("user.user_id"))
    submitted_at = Column(DateTime(timezone=True))
    status = Column(String)
    ai_confidence_score = Column(Float)
    base_servings = Column(Integer)
    views = Column(Integer, default=0)  # Track views per version
    recipe = relationship("Recipe", foreign_keys="RecipeVersion.recipe_id", back_populates="versions")
    ingredients = relationship("Ingredient", back_populates="version")
    steps = relationship("Step", back_populates="version")
    validations = relationship("Validation", back_populates="version")
    feedbacks = relationship("Feedback", back_populates="version")

class Ingredient(Base):
    __tablename__ = "ingredients"
    ingredient_id = Column(String, primary_key=True)
    version_id = Column(String, ForeignKey("recipe_versions.version_id"))
    name = Column(String)
    quantity = Column(Float)
    unit = Column(String)
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
    validated_at = Column(DateTime(timezone=True))
    approved = Column(Boolean)
    feedback = Column(Text)
    version = relationship("RecipeVersion", back_populates="validations")

class Feedback(Base):
    __tablename__ = "feedbacks"
    feedback_id = Column(String, primary_key=True)
    version_id = Column(String, ForeignKey("recipe_versions.version_id"))
    user_id = Column(String, ForeignKey("user.user_id"))
    created_at = Column(DateTime(timezone=True))
    rating = Column(Integer)
    comment = Column(Text)
    flagged = Column(Boolean, default=False)
    is_revised = Column(Boolean, default=False)
    revised_at = Column(DateTime(timezone=True))
    version = relationship("RecipeVersion", back_populates="feedbacks")
    user = relationship("User", back_populates="feedbacks")

class TokenTransaction(Base):
    __tablename__ = "token_transactions"
    tx_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("user.user_id"))
    recipe_id = Column(String, ForeignKey("recipes.recipe_id"))
    date = Column(DateTime(timezone=True))
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
    created_at = Column(DateTime(timezone=True))
    user = relationship("User", back_populates="point_logs")

class RecipeScore(Base):

    __tablename__ = "recipe_scores"
    score_id = Column(String, primary_key=True)
    recipe_id = Column(String, ForeignKey("recipes.recipe_id"))
    version_id = Column(String, ForeignKey("recipe_versions.version_id"), nullable=True)
    rating = Column(Float)
    ingredient_authenticity_score = Column(Float)
    serving_scalability_score = Column(Float)
    popularity_score = Column(Float)
    ai_confidence_score = Column(Float)
    final_score = Column(Float)
    calculated_at = Column(DateTime(timezone=True))
    recipe = relationship("Recipe", back_populates="recipe_score")
    version = relationship("RecipeVersion")



class EventPlan(Base):
    __tablename__ = "event_plans"
    event_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("user.user_id"))
    event_date = Column(DateTime(timezone=True))
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
