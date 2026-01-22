from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from Module.database import get_db
from Module.routers.base import api_router
from Module.routers.auth import get_current_user
from Module.schemas.user import RegisterRequest, UserUpdate, UserResponse
from Module.services.user_service import UserService

@api_router.post("/register", status_code=status.HTTP_201_CREATED)
def register_user(request: RegisterRequest, db: Session = Depends(get_db)):
    try:
        service = UserService(db)
        result = service.register_user(request)
        return {
            "status": True,
            "message": "Registration complete. User created.",
            "data": result
        }
    except ValueError as e:
        if "already exists" in str(e):
            raise HTTPException(status_code=409, detail=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.patch("/user/{user_id}")
def update_user(
    user_id: str, 
    user_update: UserUpdate, 
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Manual field-order validation for friendly messages
        if user_update.name is not None:
            if len(user_update.name.strip()) == 0:
                raise HTTPException(status_code=422, detail="Name cannot be empty.")
            if len(user_update.name.strip()) < 2:
                raise HTTPException(status_code=422, detail="Name must be at least 2 characters long.")

        if user_update.email is not None:
            if len(user_update.email.strip()) == 0:
                raise HTTPException(status_code=422, detail="Email must be a valid email address (e.g., user@example.com).")

        if user_update.dietary_preference is not None:
            if len(user_update.dietary_preference.strip()) == 0:
                raise HTTPException(status_code=422, detail="Dietary preference is required. Use one of: VEG, NON_VEG.")

        # Validate user_id format first
        import uuid as uuid_module
        try:
            uuid_module.UUID(user_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid user ID format. Please provide a valid UUID (e.g., 020b5621-937d-4c82-8752-959c512fgh78)."
            )
        
        # Authorization: only the user themselves or an admin can update
        current_user_id = current_user.get("user_id")
        current_user_role = current_user.get("role")
        
        if current_user_id != user_id and current_user_role != "admin":
            raise HTTPException(
                status_code=403, 
                detail="You do not have permission to update this user. Only the account owner or an admin can update user information."
            )
        
        service = UserService(db)
        result = service.update_user(user_id, user_update)
        return {
            "status": True,
            "message": "User updated successfully.",
            "data": result
        }
    except HTTPException:
        raise
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/user/{user_id}")
def get_user(
    user_id: str, 
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    try:
        # Validate user_id format first
        import uuid as uuid_module
        try:
            uuid_module.UUID(user_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid user ID format. Please provide a valid UUID (e.g., 020b5621-937d-4c82-8752-959c512eee57)."
            )
        
        # Authorization: only the user themselves or an admin can view
        current_user_id = current_user.get("user_id")
        current_user_role = current_user.get("role")
        
        if current_user_id != user_id and current_user_role != "admin":
            raise HTTPException(
                status_code=403, 
                detail="You do not have permission to view this user's profile. Only the account owner or an admin can view user information."
            )
        
        service = UserService(db)
        result = service.get_user(user_id)
        return {
            "status": True,
            "message": "User fetched successfully.",
            "data": result
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/user/email/{email}")
def get_user_by_email(email: str, db: Session = Depends(get_db)):
    try:
        service = UserService(db)
        result = service.get_user_by_email(email)
        return {
            "status": True,
            "message": "User fetched successfully.",
            "data": result
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
