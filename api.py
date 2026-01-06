"""
FastAPI application for KitchenMind recipe synthesis system.
Provides REST API for recipe management, synthesis, and event planning.
"""
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
import os
from Module.ai_validation import ai_validate_recipe
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
import uuid
from datetime import datetime

app = FastAPI()

from Module.database import get_db, init_db, Recipe, Role, User
from Module.database import DietaryPreferenceEnum
from Module.repository_postgres import PostgresRecipeRepository
from Module.controller import KitchenMind
from Module.vector_store import MockVectorStore
from Module.scoring import ScoringEngine
# Admin Profile and Session Endpoints
# (The following code was incorrectly indented and placed outside any function or class. 
# If you need to implement admin profile creation, please define it inside an endpoint or function.)

# Admin Profile and Action Log Endpoints
from Module.models import AdminProfile, AdminActionLog



class AdminProfileCreate(BaseModel):
    name: str
    email: str

class AdminProfileResponse(BaseModel):
    admin_id: str
    name: str
    email: str
    created_at: str

class AdminActionLogCreate(BaseModel):
    admin_id: str
    action_type: str
    details: str

class AdminActionLogResponse(BaseModel):
    action_id: str
    admin_id: str
    action_type: str
    timestamp: str
    details: str




from sqlalchemy.orm import Session as OrmSession
from sqlalchemy import exists
from Module.database import User

@app.post("/admin_profiles", response_model=AdminProfileResponse)
def create_admin_profile(profile: AdminProfileCreate, db: OrmSession = Depends(get_db)):
    import uuid
    from datetime import datetime
    # Check if email already exists
    exists_query = db.query(exists().where(User.email == profile.email)).scalar()
    if exists_query:
        raise HTTPException(status_code=409, detail="Admin with this email already exists")
    admin_id = str(uuid.uuid4())
    created_at = datetime.utcnow()
    # Create admin user in User table with role_id='admin'
    db_admin = User(
        user_id=admin_id,
        name=profile.name,
        email=profile.email,
        login_identifier=profile.email,
        password_hash="",  # Set empty or handle securely
        auth_type="admin",
        role_id="admin",
        dietary_preference=None,
        rating_score=0.0,
        credit=0.0,
        created_at=created_at,
        last_login_at=created_at,
        is_super_admin=True
    )
    db.add(db_admin)
    db.commit()
    db.refresh(db_admin)
    return AdminProfileResponse(
        admin_id=db_admin.user_id,
        name=db_admin.name,
        email=db_admin.email,
        created_at=str(db_admin.created_at)
    )


@app.get("/admin_profiles/{admin_id}", response_model=AdminProfileResponse)
def get_admin_profile(admin_id: str, db: OrmSession = Depends(get_db)):
    admin = db.query(User).filter(User.user_id == admin_id, User.role_id == "admin").first()
    if not admin:
        raise HTTPException(status_code=404, detail="Admin profile not found")
    return AdminProfileResponse(
        admin_id=admin.user_id,
        name=admin.name,
        email=admin.email,
        created_at=str(admin.created_at)
    )


from sqlalchemy.orm import Session as OrmSession

@app.post("/admin_actions", response_model=AdminActionLogResponse)
def create_admin_action(action: AdminActionLogCreate, db: OrmSession = Depends(get_db)):
    import uuid
    from datetime import datetime
    from Module.database import AdminActionLog
    action_id = str(uuid.uuid4())
    timestamp = datetime.utcnow()
    log = AdminActionLog(
        action_id=action_id,
        admin_id=action.admin_id,
        action_type=action.action_type,
        timestamp=timestamp,
        details=action.details
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return AdminActionLogResponse(
        action_id=log.action_id,
        admin_id=log.admin_id,
        action_type=log.action_type,
        timestamp=str(log.timestamp),
        details=log.details
    )


@app.get("/admin_actions/{action_id}", response_model=AdminActionLogResponse)
def get_admin_action(action_id: str, db: OrmSession = Depends(get_db)):
    from Module.database import AdminActionLog
    log = db.query(AdminActionLog).filter(AdminActionLog.action_id == action_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Admin action not found")
    return AdminActionLogResponse(
        action_id=log.action_id,
        admin_id=log.admin_id,
        action_type=log.action_type,
        timestamp=str(log.timestamp),
        details=log.details
    )

class SessionCreate(BaseModel):
    user_id: str

class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: str

@app.post("/session", response_model=SessionResponse)
def create_session(session: SessionCreate, db: Session = Depends(get_db)):
    # Check if user exists
    user = db.query(User).filter(User.user_id == session.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    from Module.database import Session as DBSession
    import uuid
    from datetime import datetime
    now = datetime.utcnow()
    db_session = DBSession(
        session_id=str(uuid.uuid4()),
        user_id=session.user_id,
        created_at=now
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return SessionResponse(
        session_id=db_session.session_id,
        user_id=db_session.user_id,
        created_at=str(db_session.created_at)
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use dependency injection)
km_instance = None


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class IngredientCreate(BaseModel):
    """Schema for creating an ingredient."""
    name: str
    quantity: float
    unit: str


class RecipeCreate(BaseModel):
    """Schema for creating a recipe."""
    title: str
    ingredients: List[IngredientCreate]
    steps: List[str]
    servings: int


class RecipeResponse(BaseModel):
    """Schema for recipe response."""
    id: str
    title: str
    servings: int
    approved: bool
    popularity: int
    avg_rating: float



class UserCreate(BaseModel):
    """Schema for creating a user (new schema)."""
    name: str
    email: str
    login_identifier: str
    password_hash: str
    auth_type: str
    role_id: str
    dietary_preference: str



class UserResponse(BaseModel):
    user_id: str
    name: str
    email: str
    login_identifier: str
    role_id: str
    dietary_preference: str
    rating_score: float
    credit: float
    created_at: str = None
    last_login_at: str = None
# ============================================================================
# Role Management Endpoints
# ============================================================================

class RoleCreate(BaseModel):
    role_id: str
    role_name: str
    description: str = None

class RoleResponse(BaseModel):
    role_id: str
    role_name: str
    description: str = None

# Fetch a role by role_id
@app.get("/roles/{role_id}", response_model=RoleResponse)
def get_role(role_id: str, db: Session = Depends(get_db)):
    db_role = db.query(Role).filter(Role.role_id == role_id).first()
    if not db_role:
        raise HTTPException(status_code=404, detail=f"Role '{role_id}' not found.")
    return RoleResponse(
        role_id=db_role.role_id,
        role_name=db_role.role_name,
        description=db_role.description
    )

# Fetch a user by email
@app.get("/user/email/{email}", response_model=UserResponse)
def get_user_by_email(email: str, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == email).first()
    if not db_user:
        raise HTTPException(status_code=404, detail=f"User with email '{email}' not found.")
    return UserResponse(
        user_id=db_user.user_id,
        name=db_user.name,
        email=db_user.email,
        login_identifier=db_user.login_identifier,
        role_id=db_user.role_id,
        dietary_preference=db_user.dietary_preference.value if db_user.dietary_preference else None,
        rating_score=db_user.rating_score,
        credit=db_user.credit,
        created_at=str(db_user.created_at) if db_user.created_at else None,
        last_login_at=str(db_user.last_login_at) if db_user.last_login_at else None
    )

@app.post("/roles", response_model=RoleResponse)
def create_role(role: RoleCreate, db: Session = Depends(get_db)):
    existing = db.query(Role).filter(Role.role_id == role.role_id).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Role with role_id '{role.role_id}' already exists.")
    db_role = Role(
        role_id=role.role_id,
        role_name=role.role_name,
        description=role.description
    )
    db.add(db_role)
    db.commit()
    db.refresh(db_role)
    return RoleResponse(
        role_id=db_role.role_id,
        role_name=db_role.role_name,
        description=db_role.description
    )

@app.get("/roles", response_model=list[RoleResponse])
def list_roles(db: Session = Depends(get_db)):
    roles = db.query(Role).all()
    return [RoleResponse(role_id=r.role_id, role_name=r.role_name, description=r.description) for r in roles]


class RecipeSynthesisRequest(BaseModel):
    """Schema for recipe synthesis request."""
    dish_name: str
    servings: int = 2
    top_k: int = 10
    reorder: bool = True


class RecipeValidationRequest(BaseModel):
    """Schema for recipe validation."""
    approved: bool
    feedback: Optional[str] = None
    confidence: float = 0.8


class EventPlanRequest(BaseModel):
    """Schema for event planning request."""
    event_name: str
    guest_count: int
    budget_per_person: float
    dietary: Optional[str] = None


# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
def startup_event():
    """Initialize database on startup."""
    global km_instance
    init_db()
    km_instance = KitchenMind()
    print("✓ Database initialized")
    print("✓ KitchenMind instance created")


# ============================================================================
# Health Check
# ============================================================================

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "message": "KitchenMind API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "database": "connected",
        "api": "running"
    }


# ============================================================================
# User Management Endpoints
# ============================================================================



@app.post("/user", response_model=UserResponse)
def create_user(user: UserCreate, db: Session = Depends(get_db)):
    # Check if role exists
    role = db.query(Role).filter(Role.role_id == user.role_id).first()
    if not role:
        raise HTTPException(status_code=400, detail=f"Role '{user.role_id}' does not exist.")

    # Check for duplicate email or login_identifier
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=409, detail=f"User with email '{user.email}' already exists.")
    if db.query(User).filter(User.login_identifier == user.login_identifier).first():
        raise HTTPException(status_code=409, detail=f"User with login_identifier '{user.login_identifier}' already exists.")

    # Convert dietary_preference to Enum
    try:
        dietary_pref = DietaryPreferenceEnum[user.dietary_preference] if user.dietary_preference in DietaryPreferenceEnum.__members__ else DietaryPreferenceEnum(user.dietary_preference)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid dietary_preference: {user.dietary_preference}")

    now = datetime.utcnow()
    db_user = User(
        user_id=str(uuid.uuid4()),
        name=user.name,
        email=user.email,
        login_identifier=user.login_identifier,
        password_hash=user.password_hash,
        auth_type=user.auth_type,
        role_id=user.role_id,
        dietary_preference=dietary_pref,
        rating_score=0.0,
        credit=0.0,
        created_at=now,
        last_login_at=now
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return UserResponse(
        user_id=db_user.user_id,
        name=db_user.name,
        email=db_user.email,
        login_identifier=db_user.login_identifier,
        role_id=db_user.role_id,
        dietary_preference=db_user.dietary_preference.value if db_user.dietary_preference else None,
        rating_score=db_user.rating_score,
        credit=db_user.credit,
        created_at=str(db_user.created_at) if db_user.created_at else None,
        last_login_at=str(db_user.last_login_at) if db_user.last_login_at else None
    )



@app.get("/user/{user_id}", response_model=UserResponse)
def get_user(user_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return UserResponse(
        user_id=user.user_id,
        name=user.name,
        email=user.email,
        login_identifier=user.login_identifier,
        role_id=user.role_id,
        dietary_preference=user.dietary_preference,
        rating_score=user.rating_score,
        credit=user.credit,
        created_at=str(user.created_at) if user.created_at else None,
        last_login_at=str(user.last_login_at) if user.last_login_at else None
    )


# ============================================================================
# Recipe Management Endpoints
# ============================================================================

@app.post("/recipe", response_model=RecipeResponse)
def submit_recipe(
    recipe: RecipeCreate,
    trainer_id: str = Query(...),
    db: Session = Depends(get_db)
):
    """Submit a new recipe (trainer only)."""
    print("[DEBUG] ENTER submit_recipe endpoint")
    print(f"[DEBUG] incoming recipe: {recipe}")
    print(f"[DEBUG] incoming trainer_id: {trainer_id}")
    print(f"[DEBUG] trainer_id: {trainer_id}")
    print(f"[DEBUG] recipe: {recipe}")
    trainer = db.query(User).filter(User.user_id == trainer_id).first()
    print(f"[DEBUG] trainer: {trainer}")
    if not trainer:
        raise HTTPException(status_code=404, detail="Trainer not found")

    print(f"[DEBUG] trainer.role: {trainer.role}, type: {type(trainer.role)}")
    # Handle SQLAlchemy Role object, Enum, or string
    trainer_role = trainer.role
    if hasattr(trainer_role, 'role_id'):
        trainer_role = trainer_role.role_id
    elif hasattr(trainer_role, 'value'):
        trainer_role = trainer_role.value
    print(f"[DEBUG] normalized trainer_role: {trainer_role}")
    if str(trainer_role).lower() not in ["trainer", "admin"]:
        raise HTTPException(status_code=403, detail="Only trainers can submit recipes")


    # Convert ingredients to IngredientCreate objects if they are dicts
    from pydantic import parse_obj_as
    if recipe.ingredients and isinstance(recipe.ingredients[0], dict):
        ingredients_obj = parse_obj_as(List[IngredientCreate], recipe.ingredients)
    else:
        ingredients_obj = recipe.ingredients

    try:
        # Create and save recipe using repository directly
        postgres_repo = PostgresRecipeRepository(db)
        print(f"[DEBUG] PostgresRecipeRepository created: {postgres_repo}")
        recipe_obj = postgres_repo.create_recipe(
            title=recipe.title,
            ingredients=ingredients_obj,
            steps=recipe.steps,
            servings=recipe.servings,
            submitted_by=trainer.user_id
        )
        print(f"[DEBUG] recipe_obj returned: {recipe_obj}")
        print(f"[DEBUG] recipe_obj.id: {getattr(recipe_obj, 'id', None)}")
        # Sync to in-memory store for synthesis
        try:
            from Module.models import Recipe, Ingredient
            # Convert DB recipe to in-memory Recipe model
            mem_recipe = Recipe(
                id=recipe_obj.id,
                title=recipe_obj.title,
                ingredients=[Ingredient(name=ing.name, quantity=ing.quantity, unit=ing.unit) for ing in recipe_obj.ingredients],
                steps=recipe_obj.steps,
                servings=recipe_obj.servings,
                metadata=getattr(recipe_obj, 'metadata', {}),
                ratings=getattr(recipe_obj, 'ratings', []),
                validator_confidence=getattr(recipe_obj, 'validator_confidence', 0.0),
                popularity=getattr(recipe_obj, 'popularity', 0),
                approved=recipe_obj.approved,
                rejection_suggestions=getattr(recipe_obj, 'rejection_suggestions', [])
            )
            km_instance.recipes.add(mem_recipe)
            km_instance.vstore.index(mem_recipe)
            print(f"[DEBUG] Synced recipe to in-memory store: {mem_recipe.id}")
        except Exception as sync_e:
            print(f"[ERROR] Failed to sync recipe to in-memory store: {sync_e}")
        # Ensure ingredients and steps are in the expected format
        response = RecipeResponse(
            id=recipe_obj.id,
            title=recipe_obj.title,
            servings=recipe_obj.servings,
            approved=recipe_obj.approved,
            popularity=getattr(recipe_obj, 'popularity', 0),
            avg_rating=recipe_obj.avg_rating() if hasattr(recipe_obj, 'avg_rating') else 0.0
        )
        print(f"[DEBUG] RecipeResponse: {response}")
        return response
    except Exception as e:
        print(f"[ERROR] submit_recipe exception: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/recipes", response_model=List[RecipeResponse])
def list_recipes(
    approved_only: bool = Query(True),
    db: Session = Depends(get_db)
):
    """List recipes, optionally filtering by approval status."""
    postgres_repo = PostgresRecipeRepository(db)
    recipes = postgres_repo.approved() if approved_only else postgres_repo.list()
    response = [
        RecipeResponse(
            id=r.id,
            title=r.title,
            servings=r.servings,
            approved=r.approved,
            popularity=getattr(r, "popularity", 0),
            avg_rating=r.avg_rating() if hasattr(r, "avg_rating") else 0.0
        )
        for r in recipes
    ]
    print(f"[DEBUG] list_recipes response: {response}")
    return response


# ============================================================================
# Recipe Synthesis Endpoints
# ============================================================================

@app.post("/recipe/synthesize", response_model=RecipeResponse)
def synthesize_recipe(
    request: RecipeSynthesisRequest,
    user_id: str = Query(...),
    db: Session = Depends(get_db)
):
    """Synthesize multiple recipes into one."""
    print(f"[DEBUG] synthesize_recipe called with user_id: {user_id}, request: {request}")
    # Fetch user from the database
    from Module.database import User as DBUser
    user = db.query(DBUser).filter(DBUser.user_id == user_id).first()
    print(f"[DEBUG] user: {user}")
    print(f"[DEBUG] user_id: {user.user_id}, credit: {getattr(user, 'credit', None)}")
    if not user:
        print("[ERROR] User not found")
        raise HTTPException(status_code=404, detail="User not found")

    try:
        # Synthesize recipe using controller, then save using repository
        result = km_instance.request_recipe(
            user,
            request.dish_name,
            request.servings,
            request.top_k,
            request.reorder
        )
        print(f"[DEBUG] Synthesis result: {result}")
        postgres_repo = PostgresRecipeRepository(db)
        print(f"[DEBUG] PostgresRecipeRepository created: {postgres_repo}")
        recipe_obj = postgres_repo.create_recipe(
            title=result.title,
            ingredients=[{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} for ing in result.ingredients],
            steps=result.steps,
            servings=result.servings,
            submitted_by=user.user_id
        )
        print(f"[DEBUG] recipe_obj returned: {recipe_obj}")
        print(f"[DEBUG] recipe_obj.id: {getattr(recipe_obj, 'id', None)}")
        # Ensure ingredients and steps are in the expected format
        response = RecipeResponse(
            id=recipe_obj.id,
            title=recipe_obj.title,
            servings=recipe_obj.servings,
            approved=getattr(recipe_obj, "approved", False),
            popularity=getattr(recipe_obj, "popularity", 0),
            avg_rating=recipe_obj.validator_confidence if hasattr(recipe_obj, "validator_confidence") else 0.0
        )
        print(f"[DEBUG] Synthesize response: {response}")
        return response
    except LookupError as e:
        print(f"[ERROR] synthesize_recipe LookupError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"[ERROR] synthesize_recipe Exception: {e}")
        raise HTTPException(status_code=500, detail=f"Synthesis error: {str(e)}")


# ============================================================================
# Event Planning Endpoints
# ============================================================================

@app.post("/event/plan")
def plan_event(request: EventPlanRequest):
    """Plan an event with recipes."""
    try:
        plan = km_instance.event_plan(
            request.event_name,
            request.guest_count,
            request.budget_per_person,
            request.dietary
        )
        return plan
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/recipes/pending")
def get_pending_recipes(db: Session = Depends(get_db)):
    """Get all pending (unapproved) recipes."""
    postgres_repo = PostgresRecipeRepository(db)
    recipes = postgres_repo.pending()
    
    return [
        {
            "id": r.id,
            "title": r.title,
            "servings": r.servings,
            "submitted_by": r.metadata.get("submitted_by", "unknown")
        }
        for r in recipes
    ]


# ============================================================================
# Recipe Validation (Approve/Reject) Endpoint (FIXED: validator_id as query param)
# ============================================================================
@app.post("/recipe/{recipe_id}/validate", response_model=RecipeResponse)
def validate_recipe(
    recipe_id: str,
    request: RecipeValidationRequest,
    validator_id: str = Query(...),
    db: Session = Depends(get_db)
):
    print(f"[DEBUG] validate_recipe called with recipe_id={recipe_id}, validator_id={validator_id}, request={request}")
    postgres_repo = PostgresRecipeRepository(db)
    recipe = postgres_repo.get(recipe_id)
    if not recipe:
        print("[DEBUG] Recipe not found")
        raise HTTPException(status_code=404, detail="Recipe not found")

    # Optionally, check validator exists
    validator = db.query(User).filter(User.user_id == validator_id).first()
    print(f"[DEBUG] validator: {validator}")
    if not validator:
        print("[DEBUG] Validator not found")
        raise HTTPException(status_code=404, detail="Validator not found")
    if hasattr(validator, 'role'):
        role = validator.role.role_id if hasattr(validator.role, 'role_id') else validator.role
        print(f"[DEBUG] validator role: {role}")
        if str(role).lower() != "validator":
            print("[DEBUG] Not a validator role")
            raise HTTPException(status_code=403, detail="Only validators can approve/reject recipes")

    # Update recipe approval status
    recipe.approved = request.approved
    # Optionally, store feedback/confidence in metadata or another table
    recipe.metadata["validation_feedback"] = request.feedback
    recipe.validator_confidence = request.confidence
    print(f"[DEBUG] Updating recipe: approved={recipe.approved}, feedback={request.feedback}, confidence={request.confidence}")
    postgres_repo.update(recipe)

    response = RecipeResponse(
        id=recipe.id,
        title=recipe.title,
        servings=recipe.servings,
        approved=recipe.approved,
        popularity=getattr(recipe, "popularity", 0),
        avg_rating=recipe.avg_rating() if hasattr(recipe, "avg_rating") else 0.0
    )
    print(f"[DEBUG] validate_recipe response: {response}")
    return response


# ============================================================================
# Get Single Recipe Endpoint
# ============================================================================

# AI Review endpoint: Review recipe by AI (OpenAI), auto-approve if accuracy > 90%, else reject, with feedback
@app.post("/recipe/{recipe_id}/ai_review", response_model=RecipeResponse)
def ai_review_recipe(recipe_id: str, db: Session = Depends(get_db)):
    """Review a recipe using OpenAI. Approve if accuracy > 90%, else reject, with feedback."""
    postgres_repo = PostgresRecipeRepository(db)
    recipe = postgres_repo.get(recipe_id)
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")

    # Use ai_validate_recipe utility (calls OpenAI)
    approved, feedback, confidence = ai_validate_recipe(
        recipe.title,
        [f"{ing.name} {ing.quantity} {ing.unit}" for ing in recipe.ingredients],
        recipe.steps
    )
    recipe.approved = approved
    # Save feedback and confidence in metadata if possible
    if hasattr(recipe, 'metadata'):
        recipe.metadata['ai_feedback'] = feedback
        recipe.metadata['ai_confidence'] = confidence
    postgres_repo.update(recipe)
    # Sync to in-memory store for synthesis
    try:
        from Module.models import Recipe, Ingredient
        mem_recipe = Recipe(
            id=recipe.id,
            title=recipe.title,
            ingredients=[Ingredient(name=ing.name, quantity=ing.quantity, unit=ing.unit) for ing in recipe.ingredients],
            steps=recipe.steps,
            servings=recipe.servings,
            metadata=getattr(recipe, 'metadata', {}),
            ratings=getattr(recipe, 'ratings', []),
            validator_confidence=getattr(recipe, 'validator_confidence', 0.0),
            popularity=getattr(recipe, 'popularity', 0),
            approved=recipe.approved,
            rejection_suggestions=getattr(recipe, 'rejection_suggestions', [])
        )
        km_instance.recipes.add(mem_recipe)
        km_instance.vstore.index(mem_recipe)
        print(f"[DEBUG] Synced recipe to in-memory store: {mem_recipe.id}")
    except Exception as sync_e:
        print(f"[ERROR] Failed to sync recipe to in-memory store: {sync_e}")
    return RecipeResponse(
        id=recipe.id,
        title=recipe.title,
        servings=recipe.servings,
        approved=recipe.approved,
        popularity=getattr(recipe, "popularity", 0),
        avg_rating=recipe.avg_rating() if hasattr(recipe, "avg_rating") else 0.0
    )

@app.get("/recipe/{recipe_id}", response_model=RecipeResponse)
def get_single_recipe(recipe_id: str, db: Session = Depends(get_db)):
    print(f"[DEBUG] get_single_recipe called with recipe_id={recipe_id}")
    postgres_repo = PostgresRecipeRepository(db)
    recipe = postgres_repo.get(recipe_id)
    if not recipe:
        print("[DEBUG] Recipe not found")
        raise HTTPException(status_code=404, detail="Not Found")
    response = RecipeResponse(
        id=recipe.id,
        title=recipe.title,
        servings=recipe.servings,
        approved=recipe.approved,
        popularity=getattr(recipe, "popularity", 0),
        avg_rating=recipe.avg_rating() if hasattr(recipe, "avg_rating") else 0.0
    )
    print(f"[DEBUG] get_single_recipe response: {response}")
    return response


# ============================================================================
# Rate Recipe Endpoint
# ============================================================================
@app.post("/recipe/{recipe_id}/rate")
def rate_recipe(recipe_id: str, user_id: str = Query(...), rating: float = Query(...), db: Session = Depends(get_db)):
    print(f"[DEBUG] rate_recipe called with recipe_id={recipe_id}, user_id={user_id}, rating={rating}")
    postgres_repo = PostgresRecipeRepository(db)
    recipe = postgres_repo.get(recipe_id)
    if not recipe:
        print("[DEBUG] Recipe not found")
        raise HTTPException(status_code=404, detail="Not Found")
    # Optionally, check user exists
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        print("[DEBUG] User not found")
        raise HTTPException(status_code=404, detail="User not found")
    # Add or update rating
    postgres_repo.add_rating(recipe_id, user_id, rating)
    print(f"[DEBUG] Rating added/updated for recipe {recipe_id} by user {user_id}")
    # Return recipe info as response
    return {
        "id": recipe.id,
        "title": recipe.title,
        "servings": recipe.servings,
        "approved": recipe.approved,
        "popularity": getattr(recipe, "popularity", 0),
        "avg_rating": recipe.avg_rating() if hasattr(recipe, "avg_rating") else 0.0
    }

# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    print("[API ERROR] Exception occurred:")
    print(f"Request: {request.method} {request.url}")
    print(f"Exception: {exc!r}")
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )
