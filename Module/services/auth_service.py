import hashlib
import uuid
from datetime import datetime, timedelta, timezone
from sqlalchemy.orm import Session
from zoneinfo import ZoneInfo

from Module.database import User
from Module.utils_time import get_india_time
from Module.token_utils import (
    create_access_token,
    create_refresh_token,
    decode_token,
    REFRESH_TOKEN_EXPIRE_DAYS,
)
from Module.schemas.auth import LoginRequest, OTPVerifyRequest, RefreshRequest, ChangePasswordRequest

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
        db_user.otp_expires_at = get_india_time() + timedelta(days=60)
        db_user.otp_verified = False
        self.db.commit()
        
        print(f"[OTP] For login {request.email}: {STATIC_OTP}")
        return {"email": request.email}
    
    def verify_otp(self, request: OTPVerifyRequest, request_ctx=None) -> dict:
        """Verify OTP and return user data with tokens; records session with client info when available."""
        db_user = self.db.query(User).filter(User.email == request.email).first()
        
        if not db_user:
            raise ValueError("Email not found. Please register first.")
        
        otp_hash = hashlib.sha256(request.otp.encode()).hexdigest()
        if db_user.otp_hash != otp_hash:
            raise PermissionError("Invalid OTP. Please check and try again.")
        
        if db_user.otp_expires_at and db_user.otp_expires_at < get_india_time():
            raise PermissionError("OTP expired.")
        
        db_user.otp_verified = True
        
        name_parts = db_user.name.split(" ", 1)
        first_name = name_parts[0]
        last_name = name_parts[1] if len(name_parts) > 1 else ""
        
        payload = {"user_id": db_user.user_id, "role": db_user.role_id, "email": db_user.email}
        access_token_data = create_access_token(payload)
        refresh_token_data = create_refresh_token(payload)

        # Record session aligned to refresh token lifetime
        from Module.database import Session as DBSession
        refresh_expires_at = get_india_time() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)

        ip_address = None
        user_agent = None
        if request_ctx is not None:
            user_agent = request_ctx.headers.get("user-agent")
            ip_address = request_ctx.headers.get("x-forwarded-for")
            if not ip_address and getattr(request_ctx, "client", None):
                ip_address = request_ctx.client.host

        db_session = DBSession(
            session_id=str(uuid.uuid4()),
            user_id=db_user.user_id,
            created_at=get_india_time(),
            expires_at=refresh_expires_at,
            is_active=True,
            ip_address=ip_address,
            user_agent=user_agent,
        )
        self.db.add(db_session)
        self.db.commit()
        self.db.refresh(db_session)
        
        return {
            "user_id": db_user.user_id,
            "first_name": first_name,
            "last_name": last_name,
            "email": db_user.email,
            "phone_number": getattr(db_user, "phone_number", ""),
            "role": db_user.role_id,
            "access_token": access_token_data["token"],
            "access_token_expires_at": access_token_data["expires_at"],
            "refresh_token": refresh_token_data["token"],
            "refresh_token_expires_at": refresh_token_data["expires_at"]
        }
    
    def refresh_token(self, request: RefreshRequest) -> dict:
        """Refresh access token."""
        payload = decode_token(request.refresh_token)
        new_access_token_data = create_access_token({
            "user_id": payload["user_id"], 
            "role": payload["role"], 
            "email": payload["email"]
        })
        return {
            "access_token": new_access_token_data["token"],
            "access_token_expires_at": new_access_token_data["expires_at"]
        }
    
    def validate_token(self, token: str) -> dict:
        """Validate and decode token."""
        return decode_token(token)
    
    def change_password(self, user_id: str, request: ChangePasswordRequest) -> dict:
        """Change user password after verifying old password."""
        db_user = self.db.query(User).filter(User.user_id == user_id).first()
        
        if not db_user:
            raise ValueError("User not found")
        
        # Verify old password
        old_password_hash = hashlib.sha256(request.old_password.encode()).hexdigest()
        if db_user.password_hash != old_password_hash:
            raise PermissionError("Current password is incorrect")
        
        # Validate new password format
        import re
        if len(request.new_password) < 8 or len(request.new_password) > 128:
            raise ValueError("New password must be 8-128 characters, include uppercase, lowercase, digit, and special character.")
        pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,128}$'
        if not re.match(pattern, request.new_password):
            raise ValueError("New password must be 8-128 characters, include uppercase, lowercase, digit, and special character.")
        
        # Check if new password is same as old
        if request.old_password == request.new_password:
            raise ValueError("New password must be different from the current password")
        
        # Check common passwords
        common_passwords = {"password", "123456", "12345678", "qwerty", "abc123", "111111", "123123", "letmein", "welcome", "admin", "iloveyou", "monkey", "login", "passw0rd", "starwars"}
        if request.new_password.lower() in common_passwords:
            raise ValueError("Password is too common. Please choose a stronger password.")
        
        # Update password
        new_password_hash = hashlib.sha256(request.new_password.encode()).hexdigest()
        db_user.password_hash = new_password_hash
        self.db.commit()
        
        return {"message": "Password changed successfully"}
