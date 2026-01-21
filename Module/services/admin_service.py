import uuid
from datetime import datetime, timedelta
from sqlalchemy import exists
from sqlalchemy.orm import Session
from fastapi import Request

from Module.database import User
from Module.schemas.admin import AdminProfileCreate, AdminProfileResponse, SessionCreate, SessionResponse

class AdminService:
    """Service for admin-related business logic."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_admin_profile(self, profile: AdminProfileCreate) -> AdminProfileResponse:
        """Create a new admin profile."""
        exists_query = self.db.query(exists().where(User.email == profile.email)).scalar()
        if exists_query:
            raise ValueError("Admin with this email already exists")
        
        admin_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        db_admin = User(
            user_id=admin_id,
            name=profile.name,
            email=profile.email,
            login_identifier=profile.email,
            password_hash="",
            auth_type="admin",
            role_id="admin",
            dietary_preference=None,
            rating_score=0.0,
            credit=0.0,
            created_at=created_at,
            last_login_at=created_at,
            is_super_admin=True
        )
        self.db.add(db_admin)
        self.db.commit()
        self.db.refresh(db_admin)
        
        return AdminProfileResponse(
            admin_id=db_admin.user_id,
            name=db_admin.name,
            email=db_admin.email,
            created_at=str(db_admin.created_at)
        )
    
    def get_admin_profile(self, admin_id: str) -> AdminProfileResponse:
        """Get admin profile by ID."""
        admin = self.db.query(User).filter(User.user_id == admin_id, User.role_id == "admin").first()
        if not admin:
            raise ValueError("Admin profile not found")
        
        return AdminProfileResponse(
            admin_id=admin.user_id,
            name=admin.name,
            email=admin.email,
            created_at=str(admin.created_at)
        )
    
    def create_session(self, session: SessionCreate, request: Request = None) -> SessionResponse:
        """Create a new session."""
        user = self.db.query(User).filter(User.user_id == session.user_id).first()
        if not user:
            raise ValueError(f"User not found: {session.user_id}")
        if not user.email or not user.role_id:
            raise ValueError("User profile incomplete (missing email or role)")
        
        from Module.database import Session as DBSession
        now = datetime.utcnow()
        user_agent = None
        ip_address = None
        
        if request is not None:
            user_agent = request.headers.get("user-agent")
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
        self.db.add(db_session)
        self.db.commit()
        self.db.refresh(db_session)
        
        return SessionResponse(
            session_id=db_session.session_id,
            user_id=db_session.user_id,
            created_at=str(db_session.created_at)
        )
