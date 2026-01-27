import hashlib
import uuid
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from Module.database import User, DietaryPreferenceEnum
from Module.schemas.user import RegisterRequest, UserUpdate, UserResponse, UserProfileResponse
from Module.utils_time import format_datetime_ampm as format_dt, get_india_time

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
            password_hash=hashlib.sha256(request.password.encode()).hexdigest(),
            auth_type="email",
            role_id=request.role,
            phone_number=request.phone_number,
            dietary_preference=None,
            rating_score=0.0,
            credit=0.0,
            created_at=get_india_time(),
            last_login_at=get_india_time()
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
        
        # Handle first_name and last_name by combining them into name
        first_name = update_data.pop("first_name", None)
        last_name = update_data.pop("last_name", None)
        
        if first_name is not None or last_name is not None:
            # Get current first and last names
            current_name_parts = user.name.split(' ', 1) if user.name else ['', '']
            current_first = current_name_parts[0] if len(current_name_parts) > 0 else ''
            current_last = current_name_parts[1] if len(current_name_parts) > 1 else ''
            
            # Update with new values or keep existing
            new_first = first_name if first_name is not None else current_first
            new_last = last_name if last_name is not None else current_last
            
            # Combine back into name
            user.name = f"{new_first} {new_last}".strip()
        
        # Update other fields
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
            role_id=user.role_id,
            dietary_preference=user.dietary_preference.value if user.dietary_preference else None,
            rating_score=user.rating_score,
            credit=user.credit,
            created_at=format_dt(user.created_at) if user.created_at else None,
            last_login_at=format_dt(user.last_login_at) if user.last_login_at else None
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
            role_id=user.role_id,
            dietary_preference=user.dietary_preference.value if hasattr(user.dietary_preference, 'value') else user.dietary_preference,
            rating_score=user.rating_score,
            credit=user.credit,
            created_at=format_dt(user.created_at) if user.created_at else None,
            last_login_at=format_dt(user.last_login_at) if user.last_login_at else None
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
            role_id=db_user.role_id,
            dietary_preference=db_user.dietary_preference.value if db_user.dietary_preference else None,
            rating_score=db_user.rating_score,
            credit=db_user.credit,
            created_at=format_dt(db_user.created_at) if db_user.created_at else None,
            last_login_at=format_dt(db_user.last_login_at) if db_user.last_login_at else None
        )
    
    def get_user_profile(self, user_id: str) -> UserProfileResponse:
        """Get user profile (first_name, last_name, phone_number)."""
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise ValueError("User not found")
        
        # Split name into first_name and last_name
        name_parts = user.name.split(' ', 1) if user.name else ['', '']
        first_name = name_parts[0] if len(name_parts) > 0 else ''
        last_name = name_parts[1] if len(name_parts) > 1 else ''
        
        return UserProfileResponse(
            first_name=first_name,
            last_name=last_name,
            phone_number=user.phone_number
        )
