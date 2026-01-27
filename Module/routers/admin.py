from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from Module.database import get_db
from Module.routers.base import api_router
from Module.routers.auth import get_current_user

# Dependency to check admin role before request body validation
from fastapi import Security
def admin_required(current_user: dict = Security(get_current_user)):
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Access denied. Only administrators can perform this action.")
    return current_user
from Module.schemas.admin import AdminProfileCreate, AdminProfileResponse
from Module.schemas.recipe import ApiResponse
from Module.services.admin_service import AdminService


@api_router.post("/admin_profiles", response_model=ApiResponse)
def create_admin_profile(
    profile: AdminProfileCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(admin_required)
):
    try:
        result = AdminService(db).create_admin_profile(profile)
        return ApiResponse(status=True, message="Admin profile created successfully.", data=result)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/admin_profiles/{admin_id}", response_model=ApiResponse)
def get_admin_profile(
    admin_id: str, 
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Admin-only authorization
        if current_user.get("role") != "admin":
            raise PermissionError("Access denied. Only administrators can view admin profiles.")
        
        result = AdminService(db).get_admin_profile(admin_id)
        return ApiResponse(status=True, message="Admin profile fetched successfully.", data=result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Session creation is now handled automatically during OTP verification; explicit /session endpoint removed.
