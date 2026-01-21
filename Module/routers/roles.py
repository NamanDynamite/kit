from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from Module.database import get_db
from Module.routers.base import api_router
from Module.schemas.role import RoleCreate, RoleResponse
from Module.services.role_service import RoleService

@api_router.get("/roles/{role_id}")
def get_role(role_id: str, db: Session = Depends(get_db)):
    try:
        service = RoleService(db)
        result = service.get_role(role_id)
        return {
            "status": True,
            "message": "Role fetched successfully.",
            "data": result
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/roles")
def create_role(role: RoleCreate, db: Session = Depends(get_db)):
    try:
        service = RoleService(db)
        result = service.create_role(role)
        return {
            "status": True,
            "message": "Role created successfully.",
            "data": result
        }
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/roles")
def list_roles(db: Session = Depends(get_db)):
    service = RoleService(db)
    result = service.list_roles()
    return {
        "status": True,
        "message": "Roles fetched successfully.",
        "data": result
    }
