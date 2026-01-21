import hashlib
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session

from Module.database import User
from Module.token_utils import create_access_token, create_refresh_token, decode_token
from Module.schemas.auth import LoginRequest, OTPVerifyRequest, RefreshRequest

STATIC_OTP = "123456"

class AuthService:
    """Service for authentication-related business logic."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def login_user(self, request: LoginRequest) -> dict:
        """Handle user login with OTP."""
        db_user = self.db.query(User).filter(User.email == request.email).first()
        
        if not db_user:
            raise ValueError("We could not find an account associated with this email")
        
        password_hash = hashlib.sha256(request.password.encode()).hexdigest()
        if db_user.password_hash != password_hash:
            raise PermissionError("Invalid email or password")
        
        db_user.otp_hash = hashlib.sha256(STATIC_OTP.encode()).hexdigest()
        db_user.otp_expires_at = datetime.utcnow() + timedelta(days=60)
        db_user.otp_verified = False
        self.db.commit()
        
        print(f"[OTP] For login {request.email}: {STATIC_OTP}")
        return {"email": request.email}
    
    def verify_otp(self, request: OTPVerifyRequest) -> dict:
        """Verify OTP and return user data with tokens."""
        db_user = self.db.query(User).filter(User.email == request.email).first()
        
        if not db_user:
            raise ValueError("Email not found. Please register first.")
        
        otp_hash = hashlib.sha256(request.otp.encode()).hexdigest()
        if db_user.otp_hash != otp_hash:
            raise PermissionError("Invalid OTP. Please check and try again.")
        
        if db_user.otp_expires_at and db_user.otp_expires_at < datetime.now(timezone.utc):
            raise PermissionError("OTP expired.")
        
        db_user.otp_verified = True
        self.db.commit()
        
        name_parts = db_user.name.split(" ", 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""
        
        payload = {"user_id": db_user.user_id, "role": db_user.role_id, "email": db_user.email}
        access_token = create_access_token(payload)
        refresh_token = create_refresh_token(payload)
        
        return {
            "user_id": db_user.user_id,
            "first_name": first_name,
            "last_name": last_name,
            "email": db_user.email,
            "phone_number": getattr(db_user, "phone_number", ""),
            "role": db_user.role_id,
            "access_token": access_token,
            "refresh_token": refresh_token
        }
    
    def refresh_token(self, request: RefreshRequest) -> dict:
        """Refresh access token."""
        payload = decode_token(request.refresh_token)
        new_access_token = create_access_token({
            "user_id": payload["user_id"], 
            "role": payload["role"], 
            "email": payload["email"]
        })
        return {"access_token": new_access_token}
    
    def validate_token(self, token: str) -> dict:
        """Validate and decode token."""
        return decode_token(token)
