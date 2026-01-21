from typing import Optional
from pydantic import BaseModel

class RoleCreate(BaseModel):
    """Schema for creating a role."""
    role_id: str
    role_name: str
    description: Optional[str] = None

class RoleResponse(BaseModel):
    """Schema for role response."""
    role_id: str
    role_name: str
    description: Optional[str] = None
