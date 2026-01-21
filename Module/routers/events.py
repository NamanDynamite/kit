from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from Module.database import get_db
from Module.routers.base import api_router
from Module.schemas.event import EventPlanRequest, EventPlanResponse
from Module.services.event_service import EventService

@api_router.post("/event/plan")
def plan_event(request: EventPlanRequest, db: Session = Depends(get_db)):
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
