from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from Module.database import get_db
from Module.routers.base import api_router
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
def update_user(user_id: str, user_update: UserUpdate, db: Session = Depends(get_db)):
    try:
        service = UserService(db)
        result = service.update_user(user_id, user_update)
        return {
            "status": True,
            "message": "User updated successfully.",
            "data": result
        }
    except ValueError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/user/{user_id}")
def get_user(user_id: str, db: Session = Depends(get_db)):
    try:
        service = UserService(db)
        result = service.get_user(user_id)
        return {
            "status": True,
            "message": "User fetched successfully.",
            "data": result
        }
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
