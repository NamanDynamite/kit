from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from Module.database import get_db
from Module.routers.base import api_router
from Module.routers.auth import get_current_user
from Module.schemas.user import RegisterRequest, UserUpdate, UserResponse
from Module.schemas.recipe import ApiResponse
from Module.services.user_service import UserService

@api_router.post("/register", status_code=status.HTTP_201_CREATED, response_model=ApiResponse)
def register_user(request: RegisterRequest, db: Session = Depends(get_db)):
    try:
        service = UserService(db)
        result = service.register_user(request)
        return ApiResponse(status=True, message="Registration complete. User created.", data=result)
    except ValueError as e:
        if "already exists" in str(e):
            raise HTTPException(status_code=409, detail=str(e))
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.patch("/user/{user_id}", response_model=ApiResponse)
def update_user(
    user_id: str, 
    user_update: UserUpdate, 
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    try:
        # FIRST: Validate user_id format
        import uuid as uuid_module
        import re
        try:
            uuid_module.UUID(user_id)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid user ID format. Please provide a valid UUID (e.g., 020b5621-937d-4c82-8752-959c512fgh78)."
            )
        
        # SECOND: Authorization check
        current_user_id = current_user.get("user_id")
        current_user_role = current_user.get("role")
        
        if current_user_id != user_id and current_user_role != "admin":
            raise HTTPException(
                status_code=403, 
                detail="You do not have permission to update this user. Only the account owner or an admin can update user information."
            )
        
        # THIRD: Validate request body fields in order: first_name -> last_name -> phone_number -> dietary_preference
        
        # Validate first_name
        if user_update.first_name is not None:
            first_name = user_update.first_name.strip()
            if len(first_name) == 0:
                raise HTTPException(status_code=422, detail="First name cannot be empty.")
            if len(first_name) < 2:
                raise HTTPException(status_code=422, detail="First name must be at least 2 characters.")
            if len(first_name) > 30:
                raise HTTPException(status_code=422, detail="First name must be at most 30 characters.")
            if not re.match(r"^[A-Za-z\s'-]+$", first_name):
                raise HTTPException(status_code=422, detail="First name may only contain letters, spaces, hyphens, or apostrophes.")

        # Validate last_name
        if user_update.last_name is not None:
            last_name = user_update.last_name.strip()
            if len(last_name) == 0:
                raise HTTPException(status_code=422, detail="Last name cannot be empty.")
            if len(last_name) < 2:
                raise HTTPException(status_code=422, detail="Last name must be at least 2 characters.")
            if len(last_name) > 30:
                raise HTTPException(status_code=422, detail="Last name must be at most 30 characters.")
            if not re.match(r"^[A-Za-z\s'-]+$", last_name):
                raise HTTPException(status_code=422, detail="Last name may only contain letters, spaces, hyphens, or apostrophes.")

        # Validate phone_number
        if user_update.phone_number is not None:
            phone_number = user_update.phone_number.strip()
            if len(phone_number) == 0:
                raise HTTPException(status_code=422, detail="Phone number cannot be empty.")
            if not re.match(r'^(?:\+91)?[6-9]\d{9}$', phone_number):
                raise HTTPException(status_code=422, detail="Phone number must be a valid Indian mobile number (10 digits, may start with +91, and must start with 6-9).")
            normalized_phone = phone_number
            if normalized_phone.startswith('+91'):
                normalized_phone = normalized_phone[3:]
            # Reject repeated digits and obvious sequences
            if len(set(normalized_phone)) == 1:
                raise HTTPException(status_code=422, detail="Phone number looks like a placeholder (repeated digits). Please provide a real contact number.")
            if normalized_phone in {"1234567890", "9876543210"}:
                raise HTTPException(status_code=422, detail="Phone number looks like a placeholder (sequential digits). Please provide a real contact number.")

        # Validate dietary_preference
        if user_update.dietary_preference is not None:
            dietary_preference = user_update.dietary_preference.strip()
            if len(dietary_preference) == 0:
                raise HTTPException(status_code=422, detail="Dietary preference cannot be empty.")
            if dietary_preference.upper() not in {"VEG", "NON_VEG"}:
                raise HTTPException(status_code=422, detail="Dietary preference must be one of: VEG, NON_VEG.")
        
        # FOURTH: Update user
        service = UserService(db)
        result = service.update_user(user_id, user_update)
        return ApiResponse(status=True, message="User updated successfully.", data=result)
    except HTTPException:
        raise
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/user/{user_id}", response_model=ApiResponse)
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
        result = service.get_user_profile(user_id)
        return ApiResponse(status=True, message="User fetched successfully.", data=result)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/user/email/{email}", response_model=ApiResponse)
def get_user_by_email(email: str, db: Session = Depends(get_db)):
    try:
        service = UserService(db)
        result = service.get_user_by_email(email)
        return ApiResponse(status=True, message="User fetched successfully.", data=result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
