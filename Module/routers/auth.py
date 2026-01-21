from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from Module.database import get_db
from Module.routers.base import api_router
from Module.schemas.auth import LoginRequest, OTPVerifyRequest, RefreshRequest
from Module.services.auth_service import AuthService

security = HTTPBearer()

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    try:
        service = AuthService(db)
        return service.validate_token(credentials.credentials)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

@api_router.post("/login")
def login_user(request: LoginRequest, db: Session = Depends(get_db)):
    try:
        service = AuthService(db)
        result = service.login_user(request)
        return {
            "status": True,
            "message": "OTP sent to your email. Please verify.",
            "data": result
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/verify-otp")
def verify_otp(request: OTPVerifyRequest, db: Session = Depends(get_db)):
    try:
        service = AuthService(db)
        user_data = service.verify_otp(request)
        return {
            "status": True,
            "message": "Welcome! You've successfully logged in.",
            "data": user_data
        }
    except ValueError as e:
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/refresh-token")
def refresh_token_endpoint(request: RefreshRequest, db: Session = Depends(get_db)):
    try:
        service = AuthService(db)
        tokens = service.refresh_token(request)
        return {
            "status": True,
            "message": "Token refreshed",
            "data": tokens
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))

@api_router.get("/protected")
def protected_route(user=Depends(get_current_user)):
    return {
        "status": True,
        "message": "You are authenticated",
        "data": {"user": user}
    }
