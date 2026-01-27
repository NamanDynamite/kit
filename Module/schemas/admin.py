from pydantic import BaseModel, field_validator, EmailStr
import uuid
import re

class AdminProfileCreate(BaseModel):
    """Schema for creating admin profile."""
    name: str
    email: str = "user@example.com"

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name: 2-100 chars, letters/spaces/hyphens/apostrophes."""
        if not v or not v.strip():
            raise ValueError("name cannot be empty")
        v = v.strip()
        if len(v) < 2 or len(v) > 100:
            raise ValueError("name must be 2-100 characters")
        if not re.match(r"^[A-Za-z\s\-']+$", v):
            raise ValueError("name can only contain letters, spaces, hyphens, and apostrophes")
        return v

    @field_validator('email')
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email format and normalize to lowercase."""
        if not v or not v.strip():
            raise ValueError("email cannot be empty")
        v = v.strip().lower()
        if len(v) > 255:
            raise ValueError("email must be 255 characters or less")
        email_regex = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_regex, v):
            raise ValueError("invalid email format")
        return v

class AdminProfileResponse(BaseModel):
    """Schema for admin profile response."""
    admin_id: str
    name: str
    email: str
    created_at: str

class SessionCreate(BaseModel):
    """Schema for creating session."""
    user_id: str

    @field_validator('user_id')
    @classmethod
    def validate_user_id_format(cls, v: str) -> str:
        """Validate that user_id is a valid UUID."""
        if not v or not v.strip():
            raise ValueError('User ID cannot be empty')
        try:
            uuid.UUID(v)
        except (ValueError, AttributeError):
            raise ValueError('Invalid user ID format. Please provide a valid user identifier.')
        return v

class SessionResponse(BaseModel):
    """Schema for session response."""
    session_id: str
    user_id: str
    created_at: str
