from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from Module.database import get_db
from Module.routers.base import api_router
from Module.routers.auth import get_current_user
from Module.schemas.role import RoleCreate, RoleResponse
from Module.schemas.recipe import ApiResponse
from Module.services.role_service import RoleService

@api_router.get("/roles/{role_id}", response_model=ApiResponse)
def get_role(
    role_id: str, 
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Admin-only authorization
        if current_user.get("role") != "admin":
            # Log unauthorized access attempt (could be expanded with real logging)
            # logger.warning(f"Unauthorized access to /roles/{{role_id}} by user: {current_user.get('id')}")
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to view this role. Please contact your administrator."
            )

        service = RoleService(db)
        result = service.get_role(role_id)
        return ApiResponse(status=True, message="Role fetched successfully.", data=result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail="Not found.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error.")

@api_router.post("/roles", response_model=ApiResponse)
def create_role(
    role: RoleCreate, 
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Admin-only authorization
        if current_user.get("role") != "admin":
            raise PermissionError("Access denied. Only administrators can create roles.")
        
        service = RoleService(db)
        result = service.create_role(role)
        return ApiResponse(status=True, message="Role created successfully.", data=result)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/roles", response_model=ApiResponse)
def list_roles(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    service = RoleService(db)
    result = service.list_roles()
    return ApiResponse(status=True, message="Roles fetched successfully.", data=result)
