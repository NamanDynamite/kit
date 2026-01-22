from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from Module.database import get_db
from Module.routers.base import api_router
from Module.routers.auth import get_current_user
from Module.schemas.event import EventPlanRequest, EventPlanResponse
from Module.services.event_service import EventService

@api_router.post("/event/plan", include_in_schema=False)
def plan_event(
    request: EventPlanRequest, 
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    try:
        service = EventService(db)
        result = service.plan_event(request)
        return {
            "status": True,
            "message": "Event planned successfully.",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
