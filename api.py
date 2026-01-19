from fastapi import FastAPI, Depends
from typing import List
from flask import request
from pydantic import BaseModel
from sqlalchemy.orm import Session
from Module.database import get_db, init_db, Recipe, Role, User
from Module.database import DietaryPreferenceEnum
import hashlib
from Module.token_utils import create_access_token, create_refresh_token, decode_token

app = FastAPI()
STATIC_OTP = "123456"

class RecipeResponse(BaseModel):
    """Schema for recipe response."""
    recipe_id: str
    version_id: str = None
    title: str
    servings: int
    approved: bool
    popularity: int
    ingredients: list = []
    steps: list = []



# Public recipe search endpoint (no login required, returns up to 2 recipes)
@app.get("/public/recipes", response_model=List[RecipeResponse])
def public_recipes(db: Session = Depends(get_db)):
    recipes = db.query(Recipe).filter(Recipe.is_published == True).limit(2).all()
    result = []
    for r in recipes:
        result.append(RecipeResponse(
            recipe_id=r.recipe_id,
            version_id=getattr(r, 'current_version_id', None),
            title=getattr(r, 'title', getattr(r, 'dish_name', None)),
            servings=r.servings,
            approved=getattr(r, 'is_published', False),
            popularity=getattr(r, 'popularity', 0),
            ingredients=[{'name': ing.name, 'quantity': ing.quantity, 'unit': ing.unit} for ing in getattr(r, 'ingredients', [])],
            steps=getattr(r, 'steps', [])
        ))
    return result
# Login request/response schemas
class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    email: str
    message: str

# Login endpoint with OTP
@app.post("/login", response_model=LoginResponse)
def login_user(request: LoginRequest, db: Session = Depends(get_db)):
    # Find user by email
    db_user = db.query(User).filter(User.email == request.email).first()
    password_hash = hashlib.sha256(request.password.encode()).hexdigest()
    if not db_user or db_user.password_hash != password_hash:
        return LoginResponse(email=request.email, message="Invalid email or password.")
    # Store OTP and expiry in DB
    from datetime import datetime, timedelta
    db_user.otp_hash = hashlib.sha256(STATIC_OTP.encode()).hexdigest()
    db_user.otp_expires_at = datetime.utcnow() + timedelta(days=60)
    db_user.otp_verified = False
    db.commit()
    print(f"[OTP] For login {request.email}: {STATIC_OTP}")
    return LoginResponse(email=request.email, message="OTP sent to your email. Please verify.")
from fastapi import status
import random
from typing import Optional
# Temporary in-memory store for OTPs and pending users (for demo)
# Registration request/response schemas
class RegisterRequest(BaseModel):
    first_name: str
    last_name: str
    email: str
    phone_number: str
    password: str
    role: str

class RegisterResponse(BaseModel):
    email: str
    message: str

class OTPVerifyRequest(BaseModel):
    email: str
    otp: str

class OTPVerifyResponse(BaseModel):
    data: dict
    message: str
    status: str
    token: str

# Registration endpoint (no OTP)
@app.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
def register_user(request: RegisterRequest, db: Session = Depends(get_db)):
    # Check for duplicate email or phone
    existing_user = db.query(User).filter(User.email == request.email).first()
    if existing_user:
        return RegisterResponse(email=request.email, message="User already exists.")
    import uuid
    from datetime import datetime
    user_id = str(uuid.uuid4())
    db_user = User(
        user_id=user_id,
        name=f"{request.first_name} {request.last_name}",
        email=request.email,
        login_identifier=request.email,
        
        password_hash=hashlib.sha256(request.password.encode()).hexdigest(),
        auth_type="email",
        role_id=request.role,
        phone_number=request.phone_number,
        dietary_preference=None,
        rating_score=0.0,
        credit=0.0,
        created_at=datetime.utcnow(),
        last_login_at=datetime.utcnow()
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return RegisterResponse(email=request.email, message="Registration complete. User created.")

# OTP verification endpoint (login only)
@app.post("/verify-otp", response_model=OTPVerifyResponse)
def verify_otp(request: OTPVerifyRequest, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == request.email).first()
    otp_hash = hashlib.sha256(request.otp.encode()).hexdigest()
    if not db_user or db_user.otp_hash != otp_hash:
        return OTPVerifyResponse(
            data={},
            message="Invalid or expired OTP.",
            status="fail",
            token=""
        )
    from datetime import timezone
    if db_user.otp_expires_at and db_user.otp_expires_at < datetime.now(timezone.utc):
        return OTPVerifyResponse(
            data={},
            message="OTP expired.",
            status="fail",
            token=""
        )
    db_user.otp_verified = True
    db.commit()
    # Split name into first and last (if possible)
    name_parts = db_user.name.split(" ", 1)
    first_name = name_parts[0]
    last_name = name_parts[1] if len(name_parts) > 1 else ""
    user_data = {
        "user_id": db_user.user_id,
        "first_name": first_name,
        "last_name": last_name,
        "email": db_user.email,
        "phone_number": getattr(db_user, "phone_number", ""),
        "role": db_user.role_id
    }
    # Generate tokens
    payload = {"user_id": db_user.user_id, "role": db_user.role_id, "email": db_user.email}
    access_token = create_access_token(payload)
    refresh_token = create_refresh_token(payload)
    return OTPVerifyResponse(
        data={
            "refresh_token": refresh_token,
            "access_token": access_token
        },
        message="Login successful",
        status="success",
        token=access_token
    )

# Add refresh token endpoint
from fastapi import HTTPException
from fastapi import Request

class RefreshRequest(BaseModel):
    refresh_token: str

@app.post("/refresh-token")
def refresh_token_endpoint(request: RefreshRequest):
    try:
        payload = decode_token(request.refresh_token)
        # Optionally: check if refresh token is in DB/session and not revoked
        new_access_token = create_access_token({"user_id": payload["user_id"], "role": payload["role"], "email": payload["email"]})
        return {
            "success": True,
            "message": "Token refreshed",
            "data": {
                "access_token": new_access_token
            }
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

# Dependency for protected routes
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = decode_token(credentials.credentials)
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# Example protected route
@app.get("/protected")
def protected_route(user=Depends(get_current_user)):
    return {"message": "You are authenticated", "user": user}

# PATCH endpoint to update user fields (including admin_action_* fields)
from fastapi import Body, Depends
from typing import Any
from pydantic import BaseModel


class UserUpdate(BaseModel):
    name: str = None
    email: str = None
    login_identifier: str = None
    password_hash: str = None
    auth_type: str = None
    role_id: str = None
    dietary_preference: str = None
    rating_score: float = None
    credit: float = None
    last_login_at: str = None
    is_super_admin: bool = None
    created_by: str = None
    admin_action_type: str = None
    admin_action_target_type: str = None
    admin_action_target_id: str = None
    admin_action_description: str = None
    admin_action_created_at: str = None

# Import get_db and related models early so they are available for endpoints
from Module.database import get_db, init_db, Recipe, Role, User
from Module.database import DietaryPreferenceEnum

# UserResponse must be defined before any endpoint that uses it
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
    name: Optional[str] = None
    email: Optional[str] = None
    login_identifier: Optional[str] = None
    password_hash: Optional[str] = None
    auth_type: Optional[str] = None
    role_id: Optional[str] = None
    dietary_preference: Optional[str] = None
    rating_score: Optional[float] = None
    credit: Optional[float] = None
    last_login_at: Optional[str] = None
    is_super_admin: Optional[bool] = None
    created_by: Optional[str] = None
    admin_action_type: Optional[str] = None
    admin_action_target_type: Optional[str] = None
    admin_action_target_id: Optional[str] = None
    admin_action_description: Optional[str] = None
    admin_action_created_at: Optional[str] = None

class UserUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    login_identifier: Optional[str] = None
    password_hash: Optional[str] = None
    auth_type: Optional[str] = None
    role_id: Optional[str] = None
    dietary_preference: Optional[str] = None
    rating_score: Optional[float] = None
    credit: Optional[float] = None
    last_login_at: Optional[str] = None
    is_super_admin: Optional[bool] = None
    created_by: Optional[str] = None
    admin_action_type: Optional[str] = None
    admin_action_target_type: Optional[str] = None
    admin_action_target_id: Optional[str] = None
    admin_action_description: Optional[str] = None
    admin_action_created_at: Optional[str] = None

# UserResponse must be defined before any endpoint that uses it
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
    name: Optional[str] = None
    email: Optional[str] = None
    login_identifier: Optional[str] = None
    password_hash: Optional[str] = None
    auth_type: Optional[str] = None
    role_id: Optional[str] = None
    dietary_preference: Optional[str] = None
    rating_score: Optional[float] = None
    credit: Optional[float] = None
    last_login_at: Optional[str] = None
    is_super_admin: Optional[bool] = None
    created_by: Optional[str] = None
    admin_action_type: Optional[str] = None
    admin_action_target_type: Optional[str] = None
    admin_action_target_id: Optional[str] = None
    admin_action_description: Optional[str] = None
    admin_action_created_at: Optional[str] = None

@app.patch("/user/{user_id}", response_model=UserResponse)
def update_user(user_id: str, user_update: UserUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    update_data = user_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        if field == "dietary_preference" and value is not None:
            try:
                value = DietaryPreferenceEnum[value] if value in DietaryPreferenceEnum.__members__ else DietaryPreferenceEnum(value)
            except Exception:
                raise HTTPException(status_code=400, detail=f"Invalid dietary_preference: {value}")
        setattr(user, field, value)
    db.commit()
    db.refresh(user)
    return UserResponse(
        user_id=user.user_id,
        name=user.name,
        email=user.email,
        login_identifier=user.login_identifier,
        role_id=user.role_id,
        dietary_preference=user.dietary_preference.value if user.dietary_preference else None,
        rating_score=user.rating_score,
        credit=user.credit,
        created_at=str(user.created_at) if user.created_at else None,
        last_login_at=str(user.last_login_at) if user.last_login_at else None
    )
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
from Module.models import AdminProfile



class AdminProfileCreate(BaseModel):
    name: str
    email: str

class AdminProfileResponse(BaseModel):
    admin_id: str
    name: str
    email: str
    created_at: str





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


class SessionCreate(BaseModel):
    user_id: str

class SessionResponse(BaseModel):
    session_id: str
    user_id: str
    created_at: str

from fastapi import Request

@app.post("/session", response_model=SessionResponse)
def create_session(session: SessionCreate, db: Session = Depends(get_db), request: Request = None):
    print(f"[DEBUG] create_session called with: {session}")
    # Check if user exists
    user = db.query(User).filter(User.user_id == session.user_id).first()
    print(f"[DEBUG] User lookup for session: {user}")
    if not user:
        print(f"[DEBUG] User not found for session: {session.user_id}")
        raise HTTPException(status_code=404, detail="User not found")
    from Module.database import Session as DBSession
    import uuid
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    # Extract user agent and IP address from request
    user_agent = None
    ip_address = None
    if request is not None:
        user_agent = request.headers.get("user-agent")
        # Try X-Forwarded-For first, then fallback to client.host
        ip_address = request.headers.get("x-forwarded-for")
        if not ip_address and request.client:
            ip_address = request.client.host
    expires_at = now + timedelta(hours=1)
    db_session = DBSession(
        session_id=str(uuid.uuid4()),
        user_id=session.user_id,
        created_at=now,
        expires_at=expires_at,
        is_active=True,
        ip_address=ip_address,
        user_agent=user_agent
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    print(f"[DEBUG] Session created: {db_session.session_id} for user: {db_session.user_id}")
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
    recipe_id: str
    version_id: str = None
    title: str
    servings: int
    approved: bool
    popularity: int
    # avg_rating removed from model
    ingredients: list = []
    steps: list = []



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
    ingredients: Optional[List[IngredientCreate]] = None
    steps: Optional[List[str]] = None


class RecipeValidationRequest(BaseModel):
    """Schema for recipe validation."""
    approved: bool
    feedback: Optional[str] = None
    confidence: float = 0.8


class EventPlanResponse(BaseModel):
    event: str
    user_id: str
    guest_count: int
    budget_per_person: float
    dietary: Optional[str] = None

class EventPlanRequest(BaseModel):
    """Schema for event planning request."""
    user_id: str
    event_name: str
    guest_count: int
    budget_per_person: float
    dietary: Optional[str] = None

@app.post("/event/plan", response_model=EventPlanResponse)
def plan_event(request: EventPlanRequest):
    """Plan an event and return details."""
    # Here you can add logic to store or process the event
    return EventPlanResponse(
        event=request.event_name,
        user_id=request.user_id,
        guest_count=request.guest_count,
        budget_per_person=request.budget_per_person,
        dietary=request.dietary
    )


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
    print(f"[DEBUG] create_user called with: {user}")
    # Check if role exists
    role = db.query(Role).filter(Role.role_id == user.role_id).first()
    print(f"[DEBUG] Role lookup for '{user.role_id}': {role}")
    if not role:
        print(f"[DEBUG] Role '{user.role_id}' does not exist.")
        raise HTTPException(status_code=400, detail=f"Role '{user.role_id}' does not exist.")

    # Check for duplicate email or login_identifier
    if db.query(User).filter(User.email == user.email).first():
        print(f"[DEBUG] Duplicate email: {user.email}")
        raise HTTPException(status_code=409, detail=f"User with email '{user.email}' already exists.")
    if db.query(User).filter(User.login_identifier == user.login_identifier).first():
        print(f"[DEBUG] Duplicate login_identifier: {user.login_identifier}")
        raise HTTPException(status_code=409, detail=f"User with login_identifier '{user.login_identifier}' already exists.")

    # Convert dietary_preference to Enum
    try:
        dietary_pref = DietaryPreferenceEnum[user.dietary_preference] if user.dietary_preference in DietaryPreferenceEnum.__members__ else DietaryPreferenceEnum(user.dietary_preference)
        print(f"[DEBUG] dietary_preference enum: {dietary_pref}")
    except Exception as e:
        print(f"[DEBUG] Invalid dietary_preference: {user.dietary_preference}, error: {e}")
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
    print(f"[DEBUG] User created: {db_user.user_id}")
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
                recipe_id=recipe_obj.id,
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
        # Get version_id from the database
        version_id = None
        from Module.database import Recipe as DBRecipe
        db_recipe = db.query(DBRecipe).filter(DBRecipe.recipe_id == recipe_obj.id).first()
        if db_recipe and hasattr(db_recipe, 'current_version_id'):
            version_id = db_recipe.current_version_id
        response = RecipeResponse(
            recipe_id=recipe_obj.id,
            version_id=version_id,
            title=recipe_obj.title,
            servings=recipe_obj.servings,
            approved=recipe_obj.approved,
            popularity=getattr(recipe_obj, 'popularity', 0),
            # avg_rating removed from response
            ingredients=[{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} for ing in getattr(recipe_obj, 'ingredients', [])],
            steps=getattr(recipe_obj, 'steps', [])
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
    response = []
    for r in recipes:
        version_id = None
        from Module.database import Recipe as DBRecipe
        db_recipe = db.query(DBRecipe).filter(DBRecipe.recipe_id == r.id).first()
        if db_recipe and hasattr(db_recipe, 'current_version_id'):
            version_id = db_recipe.current_version_id
        response.append(RecipeResponse(
            recipe_id=r.id,
            version_id=version_id,
            title=r.title,
            servings=r.servings,
            approved=r.approved,
            popularity=getattr(r, "popularity", 0),
            # avg_rating removed from response
            ingredients=[i for i in getattr(r, 'ingredients', [])],
            steps=[s for s in getattr(r, 'steps', [])]
        ))
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
            # Use keyword arguments to avoid too many positional arguments
            kwargs = {
                'user': user,
                'dish_name': request.dish_name,
                'servings': request.servings
            }
            if request.ingredients is not None:
                kwargs['ingredients'] = request.ingredients
            result = km_instance.request_recipe(**kwargs)
            print(f"[DEBUG] Before ensure_recipe_dataclass: type={type(result)}, dir={dir(result)}, repr={repr(result)}")
            from Module.controller import ensure_recipe_dataclass
            result = ensure_recipe_dataclass(result)
            print(f"[DEBUG] After ensure_recipe_dataclass: type={type(result)}, dir={dir(result)}, repr={repr(result)}")
            if not hasattr(result, 'ingredients') or not isinstance(result.ingredients, (list, tuple)):
                print(f"[ERROR] Synthesized recipe has no 'ingredients' attribute after ensure_recipe_dataclass. type={type(result)}, dir={dir(result)}, repr={repr(result)}")
                raise HTTPException(status_code=500, detail="Synthesized recipe has no 'ingredients' attribute after conversion.")
            postgres_repo = PostgresRecipeRepository(db)
            print(f"[DEBUG] PostgresRecipeRepository created: {postgres_repo}")
            try:
                ings = result.ingredients
            except Exception as attr_e:
                print(f"[ERROR] Exception accessing result.ingredients: {attr_e}")
                print(f"[ERROR] result type: {type(result)}; dir: {dir(result)}; repr: {repr(result)}")
                raise HTTPException(status_code=500, detail=f"Synthesized recipe object has no 'ingredients' attribute: {attr_e}")
            recipe_obj = postgres_repo.create_recipe(
                title=result.title,
                ingredients=[{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} for ing in ings],
                steps=result.steps,
                servings=result.servings,
                submitted_by=user.user_id,
                approved=getattr(result, 'approved', False)
            )
            print(f"[DEBUG] recipe_obj returned: {recipe_obj}")
            print(f"[DEBUG] recipe_obj.id: {getattr(recipe_obj, 'id', None)}")
            # Get version_id from the database
            version_id = None
            from Module.database import Recipe as DBRecipe
            db_recipe = db.query(DBRecipe).filter(DBRecipe.recipe_id == recipe_obj.id).first()
            if db_recipe and hasattr(db_recipe, 'current_version_id'):
                version_id = db_recipe.current_version_id
            response = RecipeResponse(
                recipe_id=recipe_obj.id,
                version_id=version_id,
                title=recipe_obj.title,
                servings=recipe_obj.servings,
                approved=recipe_obj.approved,
                popularity=getattr(recipe_obj, 'popularity', 0),
                # avg_rating removed from response
                ingredients=[{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} for ing in getattr(recipe_obj, 'ingredients', [])],
                steps=getattr(recipe_obj, 'steps', [])
            )
            print(f"[DEBUG] RecipeResponse: {response}")
            return response
    except Exception as e:
        print(f"[ERROR] synthesize_recipe exception: {e}")
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


# ============================================================================
# Get Single Recipe Endpoint
# ============================================================================

# AI Review endpoint: Review recipe by AI (OpenAI), auto-approve if accuracy > 90%, else reject, with feedback
from pydantic import BaseModel

class ValidationResponse(BaseModel):
    validation_id: str
    version_id: str
    validated_at: str
    approved: bool
    feedback: str

@app.post("/recipe/version/{version_id}/validate", response_model=ValidationResponse)
def ai_review_recipe(version_id: str, db: Session = Depends(get_db)):
    """Review a recipe version using OpenAI. Approve if accuracy > 90%, else reject, with feedback."""
    # Find the recipe version
    from Module.database import RecipeVersion, Validation
    all_versions = db.query(RecipeVersion.version_id).all()
    print(f"[DEBUG] All available version_ids: {[v[0] for v in all_versions]}")
    version = db.query(RecipeVersion).filter(RecipeVersion.version_id == version_id).first()
    if not version:
        print(f"[DEBUG] Provided version_id '{version_id}' not found in recipe_versions table.")
        raise HTTPException(status_code=404, detail="Recipe version not found")
    # Find the parent recipe
    recipe = version.recipe
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found for this version")

    # Use ai_validate_recipe utility (calls OpenAI)
    approved, feedback, confidence = ai_validate_recipe(
        recipe.dish_name,
        [f"{ing.name} {ing.quantity} {ing.unit}" for ing in version.ingredients],
        [step.instruction for step in version.steps]
    )

    # Save validation record (only required columns)
    import uuid
    from datetime import datetime
    validation = Validation(
        validation_id=str(uuid.uuid4()),
        version_id=version_id,
        validated_at=datetime.utcnow(),
        approved=approved,
        feedback=feedback
    )
    db.add(validation)
    # Also update the Recipe's is_published field if approved
    if approved:
        recipe.is_published = True
    db.commit()
    db.refresh(validation)
    db.refresh(version)
    db.refresh(recipe)

    # --- Update recipe_scores table ---
    from Module.scoring import ScoringEngine
    from Module.database import update_recipe_score
    scorer = ScoringEngine()
    # Build a mock recipe object for scoring
    class MockRecipe:
        def __init__(self, recipe, version):
            self.recipe_id = recipe.recipe_id
            self.dish_name = recipe.dish_name
            self.servings = recipe.servings
            self.popularity = getattr(recipe, 'popularity', 0)
            self.validator_confidence = confidence
            self.metadata = {'ai_confidence': confidence}
            self.ingredients = version.ingredients
            self.steps = version.steps
        def avg_rating(self):
            return 0.0
    mock_recipe = MockRecipe(recipe, version)
    ai_scores = {
        'validator_confidence_score': scorer.score(mock_recipe),
        'ingredient_authenticity_score': scorer.ingredient_authenticity_score(mock_recipe),
        'serving_scalability_score': scorer.serving_scalability_score(mock_recipe),
        'ai_confidence_score': scorer.ai_confidence_score(mock_recipe)
    }
    popularity_score = scorer.popularity_score(mock_recipe)
    update_recipe_score(db, recipe.recipe_id, ai_scores=ai_scores, popularity=popularity_score)

    return ValidationResponse(
        validation_id=validation.validation_id,
        version_id=validation.version_id,
        validated_at=validation.validated_at.isoformat(),
        approved=validation.approved,
        feedback=validation.feedback
    )


@app.get("/recipe/version/{version_id}", response_model=RecipeResponse)
def get_single_recipe_by_version(version_id: str, db: Session = Depends(get_db)):
    print(f"[DEBUG] get_single_recipe_by_version called with version_id={version_id}")
    from Module.database import RecipeVersion
    version = db.query(RecipeVersion).filter(RecipeVersion.version_id == version_id).first()
    if not version:
        print("[DEBUG] Recipe version not found")
        raise HTTPException(status_code=404, detail="Recipe version not found")
    recipe = version.recipe
    if not recipe:
        print("[DEBUG] Parent recipe not found for version")
        raise HTTPException(status_code=404, detail="Parent recipe not found for version")
    # Determine approval status from latest Validation record
    from Module.database import Validation
    latest_validation = (
        db.query(Validation)
        .filter(Validation.version_id == version.version_id)
        .order_by(Validation.validated_at.desc())
        .first()
    )
    approved = latest_validation.approved if latest_validation and latest_validation.approved is not None else False
    # Fetch scores from recipe_scores table if available
    from Module.database import RecipeScore
    score = db.query(RecipeScore).filter(RecipeScore.recipe_id == recipe.recipe_id).first()
    avg_rating = score.user_rating_score if score and score.user_rating_score is not None else 0.0
    popularity = score.popularity_score if score and score.popularity_score is not None else getattr(recipe, "popularity", 0)
    response = RecipeResponse(
        recipe_id=recipe.recipe_id,
        version_id=version.version_id,
        title=recipe.dish_name,
        servings=version.base_servings if hasattr(version, 'base_servings') and version.base_servings else getattr(recipe, 'servings', 1),
        approved=approved,
        popularity=popularity,
        # avg_rating removed from response
        ingredients=[{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} for ing in getattr(version, 'ingredients', [])],
        steps=[step.instruction for step in sorted(getattr(version, 'steps', []), key=lambda x: x.step_order)]
    )
    print(f"[DEBUG] get_single_recipe_by_version response: {response}")
    return response


# ============================================================================
# Rate Recipe Endpoint
# ============================================================================

from fastapi import Body

@app.post("/recipe/version/{version_id}/rate")
def rate_recipe(
    version_id: str,
    user_id: str = Query(...),
    rating: float = Query(...),
    comment: str = Body(default=None, embed=True),
    db: Session = Depends(get_db)
):
    print(f"[DEBUG] rate_recipe called with version_id={version_id}, user_id={user_id}, rating={rating}, comment={comment}")
    from Module.database import RecipeVersion
    version = db.query(RecipeVersion).filter(RecipeVersion.version_id == version_id).first()
    if not version:
        print("[DEBUG] Recipe version not found")
        raise HTTPException(status_code=404, detail="Recipe version not found")
    recipe_id = version.recipe_id
    from Module.repository_postgres import PostgresRecipeRepository
    postgres_repo = PostgresRecipeRepository(db)
    # Optionally, check user exists
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        print("[DEBUG] User not found")
        raise HTTPException(status_code=404, detail="User not found")
    # Add or update rating and comment
    feedback = postgres_repo.add_rating(version_id, user_id, rating, comment)
    print(f"[DEBUG] Rating and comment added/updated for version {version_id} by user {user_id}")
    # Update recipe_scores
    from Module.database import update_recipe_score, RecipeScore
    update_recipe_score(db, recipe_id)
    # Fetch updated scores
    score = db.query(RecipeScore).filter(RecipeScore.recipe_id == recipe_id).first()
    avg_rating = score.user_rating_score if score and score.user_rating_score is not None else 0.0
    return {
        "recipe_id": recipe_id,
        "version_id": version_id,
        "title": version.recipe.dish_name if version.recipe else None,
        "servings": version.base_servings if hasattr(version, 'base_servings') and version.base_servings else None,
        "approved": version.recipe.is_published if version.recipe else None,
        "avg_rating": round(avg_rating, 2),
        "comment": feedback.comment
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
