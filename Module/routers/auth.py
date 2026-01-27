from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from Module.database import get_db
from Module.routers.base import api_router
from Module.schemas.auth import LoginRequest, OTPVerifyRequest, RefreshRequest, ChangePasswordRequest
from Module.schemas.recipe import ApiResponse
from Module.services.auth_service import AuthService
from Module.token_utils import format_expiration_time

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    try:
        service = AuthService(db)
        return service.validate_token(credentials.credentials)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception:
        raise HTTPException(status_code=401, detail="Your session has expired or the token is invalid. Please log in again to get a new access token.")

@api_router.post("/login", response_model=ApiResponse)
def login_user(request: LoginRequest, db: Session = Depends(get_db)):
    try:
        service = AuthService(db)
        result = service.login_user(request)
        return ApiResponse(status=True, message="OTP sent to your email. Please verify.", data=result)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/verify-otp", response_model=ApiResponse)
def verify_otp(request: OTPVerifyRequest, db: Session = Depends(get_db), http_request: Request = None):
    try:
        service = AuthService(db)
        user_data = service.verify_otp(request, http_request)
        return ApiResponse(status=True, message="Welcome! You've successfully logged in.", data=user_data)
    except ValueError as e:
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/refresh-token", response_model=ApiResponse)
def refresh_token_endpoint(request: RefreshRequest, db: Session = Depends(get_db)):
    try:
        service = AuthService(db)
        tokens = service.refresh_token(request)
        return ApiResponse(status=True, message="Token refreshed", data=tokens)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@api_router.get("/protected", response_model=ApiResponse)
def protected_route(user=Depends(get_current_user)):
    user_data = user.copy()
    if "exp" in user_data:
        user_data["token_expires_at"] = format_expiration_time(user_data["exp"])
        del user_data["exp"]
    return ApiResponse(status=True, message="You are authenticated", data={"user": user_data})

@api_router.post("/change-password", include_in_schema=False, response_model=ApiResponse)
def change_password(
    request: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    try:
        user_id = current_user.get("user_id")
        service = AuthService(db)
        result = service.change_password(user_id, request)
        return ApiResponse(status=True, message="Password changed successfully.", data=result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
