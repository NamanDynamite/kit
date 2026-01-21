from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from Module.database import get_db
from Module.routers.base import api_router
from Module.schemas.admin import AdminProfileCreate, AdminProfileResponse, SessionCreate, SessionResponse
from Module.services.admin_service import AdminService

@api_router.post("/admin_profiles")
def create_admin_profile(profile: AdminProfileCreate, db: Session = Depends(get_db)):
    try:
        service = AdminService(db)
        result = service.create_admin_profile(profile)
        return {
            "status": True,
            "message": "Admin profile created successfully.",
            "data": result
        }
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/admin_profiles/{admin_id}")
def get_admin_profile(admin_id: str, db: Session = Depends(get_db)):
    try:
        service = AdminService(db)
        result = service.get_admin_profile(admin_id)
        return {
            "status": True,
            "message": "Admin profile fetched successfully.",
            "data": result
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/session")
def create_session(session: SessionCreate, db: Session = Depends(get_db), request: Request = None):
    try:
        if not session.user_id or session.user_id.strip() == "":
            raise HTTPException(status_code=400, detail="user_id is required and cannot be empty")
        service = AdminService(db)
        result = service.create_session(session, request)
        return {
            "status": True,
            "message": "Session created successfully.",
            "data": result
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
