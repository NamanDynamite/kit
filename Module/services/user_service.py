import hashlib
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from Module.database import User, DietaryPreferenceEnum
from Module.schemas.user import RegisterRequest, UserUpdate, UserResponse

class UserService:
    """Service for user-related business logic."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def register_user(self, request: RegisterRequest) -> dict:
        """Register a new user."""
        existing_user = self.db.query(User).filter(User.email == request.email).first()
        if existing_user:
            raise ValueError("User already exists")
        
        db_user = User(
            user_id=str(uuid.uuid4()),
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
        self.db.add(db_user)
        
        try:
            self.db.commit()
            self.db.refresh(db_user)
        except IntegrityError as e:
            self.db.rollback()
            if "user_role_id_fkey" in str(e.orig):
                raise ValueError("Role must be one of: user, trainer, admin.")
            raise
        
        return {"email": request.email}
    
    def update_user(self, user_id: str, user_update: UserUpdate) -> UserResponse:
        """Update user information."""
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise ValueError("User not found")
        
        update_data = user_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            if field == "dietary_preference" and value is not None:
                try:
                    value = DietaryPreferenceEnum[value] if value in DietaryPreferenceEnum.__members__ else DietaryPreferenceEnum(value)
                except Exception:
                    raise ValueError(f"Invalid dietary_preference: {value}")
            setattr(user, field, value)
        
        self.db.commit()
        self.db.refresh(user)
        
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
    
    def get_user(self, user_id: str) -> UserResponse:
        """Get user by ID."""
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise ValueError("User not found")
        
        return UserResponse(
            user_id=user.user_id,
            name=user.name,
            email=user.email,
            login_identifier=user.login_identifier,
            role_id=user.role_id,
            dietary_preference=user.dietary_preference.value if hasattr(user.dietary_preference, 'value') else user.dietary_preference,
            rating_score=user.rating_score,
            credit=user.credit,
            created_at=str(user.created_at) if user.created_at else None,
            last_login_at=str(user.last_login_at) if user.last_login_at else None
        )
    
    def get_user_by_email(self, email: str) -> UserResponse:
        """Get user by email."""
        db_user = self.db.query(User).filter(User.email == email).first()
        if not db_user:
            raise ValueError(f"User with email '{email}' not found")
        
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
