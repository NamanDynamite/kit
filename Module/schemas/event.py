from typing import Optional
from pydantic import BaseModel

class EventPlanRequest(BaseModel):
    """Schema for event planning request."""
    user_id: str
    event_name: str
    guest_count: int
    budget_per_person: float
    dietary: Optional[str] = None

class EventPlanResponse(BaseModel):
    """Schema for event planning response."""
    event: str
    user_id: str
    guest_count: int
    budget_per_person: float
    dietary: Optional[str] = None
