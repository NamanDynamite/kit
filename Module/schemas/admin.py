from pydantic import BaseModel, field_validator
import uuid

class AdminProfileCreate(BaseModel):
    """Schema for creating admin profile."""
    name: str
    email: str

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
