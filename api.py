from fastapi import FastAPI, Depends, APIRouter
from typing import List
from pydantic import BaseModel, EmailStr, constr, validator
from sqlalchemy.orm import Session
from Module.database import get_db, init_db, Recipe, Role, User
from Module.database import DietaryPreferenceEnum
import hashlib
from Module.token_utils import create_access_token, create_refresh_token, decode_token

# Import router modules so their endpoints register on the shared router
from Module.routers import auth, public, users, roles, admin, recipes, events


app = FastAPI()
from Module.routers.base import api_router
STATIC_OTP = "123456"

from fastapi import status, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

@app.exception_handler(RequestValidationError)
async def custom_validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    
    # Priority 1: Check query parameters (like trainer_id) first
    for err in errors:
        loc = err.get("loc", [])
        if len(loc) > 0 and loc[0] == "query":
            field_name = loc[-1] if loc else None
            msg = err.get("msg", "")
            if field_name == "trainer_id":
                return JSONResponse(status_code=422, content={"status": False, "message": "Invalid trainer ID format. Please provide a valid user identifier."})
    
    # Priority 2: Check body fields
    for err in errors:
        loc = err.get("loc", [])
        msg = err.get("msg", "")
        err_type = err.get("type", "")
        
        # Skip query params (already handled above)
        if len(loc) > 0 and loc[0] == "query":
            continue
        
        # Recipe field errors - check by field name in location path
        field_name = loc[-1] if loc else None
        
        # Quantity validation
        if field_name == "quantity" and ("greater than" in msg or err_type == "greater_than"):
            return JSONResponse(status_code=422, content={"status": False, "message": "Ingredient quantity must be greater than 0."})
        
        # Servings validation
        if field_name == "servings" and ("greater than" in msg or err_type == "greater_than_equal"):
            return JSONResponse(status_code=422, content={"status": False, "message": "Servings must be between 1 and 100."})
        
        # Title validation
        if field_name == "title":
            if "at least 3" in msg or err_type == "string_too_short":
                return JSONResponse(status_code=422, content={"status": False, "message": "Recipe title must be at least 3 characters long."})
            clean_msg = msg.replace("Value error, ", "").strip()
            return JSONResponse(status_code=422, content={"status": False, "message": clean_msg})
        
        # Steps validation
        if field_name == "steps":
            if "at least 1" in msg or err_type == "too_short":
                return JSONResponse(status_code=422, content={"status": False, "message": "At least one cooking step is required."})
            clean_msg = msg.replace("Value error, ", "").strip()
            return JSONResponse(status_code=422, content={"status": False, "message": clean_msg})
        
        # Ingredients validation
        if field_name == "ingredients":
            if "at least 1" in msg or err_type == "too_short":
                return JSONResponse(status_code=422, content={"status": False, "message": "At least one ingredient is required."})
        
        # Ingredient name/unit validation
        if field_name in ["name", "unit"] and "ingredients" in str(loc):
            clean_msg = msg.replace("Value error, ", "").strip()
            return JSONResponse(status_code=422, content={"status": False, "message": clean_msg})
        
        # First name errors
        if field_name == "first_name":
            if "First name may only contain letters, spaces, hyphens, or apostrophes." in msg:
                return JSONResponse(status_code=422, content={"status": False, "message": "First name may only contain letters, spaces, hyphens, or apostrophes."})
            if err_type == "string_too_short":
                return JSONResponse(status_code=422, content={"status": False, "message": "First name must be at least 2 characters."})
        
        # Last name errors
        if field_name == "last_name":
            if "Last name may only contain letters, spaces, hyphens, or apostrophes." in msg:
                return JSONResponse(status_code=422, content={"status": False, "message": "Last name may only contain letters, spaces, hyphens, or apostrophes."})
            if err_type == "string_too_short":
                return JSONResponse(status_code=422, content={"status": False, "message": "Last name must be at least 2 characters."})
        
        # Email errors
        if field_name == "email":
            if ("value is not a valid email address" in msg or "An email address must have an @-sign." in msg or "email address" in msg):
                return JSONResponse(status_code=422, content={"status": False, "message": "Email must be a valid email address (e.g., user@example.com)."})
        
        # Phone number errors
        if field_name == "phone_number":
            if ("Phone number must be a valid Indian mobile number" in msg or "string does not match regex" in msg or "ensure this value has at least" in msg or "ensure this value has at most" in msg):
                return JSONResponse(status_code=422, content={"status": False, "message": "Phone number must be a valid Indian mobile number (10 digits, may start with +91, and must start with 6-9)."})
        
        # Password errors
        if field_name == "password":
            if ("Password must be 8-128 characters" in msg or "Password must be at least 8 characters" in msg or "Password must be a string." in msg or "Password is too common" in msg or "String should have at least" in msg):
                return JSONResponse(status_code=422, content={"status": False, "message": "Password must be at least 8 characters and include an uppercase letter, a lowercase letter, a digit, and a special character."})
        
        # Role errors
        if field_name == "role":
            if ("Role must be one of: user, trainer, admin." in msg):
                return JSONResponse(status_code=422, content={"status": False, "message": "Role must be one of: user, trainer, admin."})
        
        # User ID / Trainer ID errors
        if field_name in ["user_id", "trainer_id"]:
            clean_msg = msg.replace("Value error, ", "").strip()
            if not clean_msg or "badly formed" in msg.lower() or "invalid" in msg.lower():
                return JSONResponse(status_code=422, content={"status": False, "message": "Invalid user ID format. Please provide a valid user identifier."})
            return JSONResponse(status_code=422, content={"status": False, "message": clean_msg})
    
    # Fallback - clean "Value error, " prefix from any validation message
    if errors:
        err = errors[0]
        msg = err.get("msg", "Invalid input.").replace("Value error, ", "").strip()
        return JSONResponse(status_code=422, content={"status": False, "message": msg})
    return JSONResponse(status_code=422, content={"status": False, "message": "Invalid input. Please check your data and try again."})



# Models and endpoints moved to Module/routers/*.py

"""
FastAPI application for KitchenMind recipe synthesis system.
Provides REST API for recipe management, synthesis, and event planning.
"""
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
import os
from Module.ai_validation import ai_validate_recipe
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
import uuid
from datetime import datetime


from Module.database import get_db, init_db, Recipe, Role, User
from Module.database import DietaryPreferenceEnum
from Module.repository_postgres import PostgresRecipeRepository
from Module.controller import KitchenMind
from Module.vector_store import MockVectorStore
from Module.scoring import ScoringEngine

# Global state (in production, use dependency injection)
km_instance = None

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Startup and Shutdown Events
# ============================================================================

@app.on_event("startup")
def startup_event():
    """Initialize database on startup."""
    global km_instance
    init_db()
    km_instance = KitchenMind()
    print("✓ Database initialized")
    print("✓ KitchenMind instance created")


# ============================================================================
# Health Check
# ============================================================================

@app.get("/")
def read_root():
    """Health check endpoint."""
    return {
        "message": "KitchenMind API",
        "version": "1.0.0",
        "status": "online",
        "docs": "/docs"
    }


@app.get("/health")
def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "database": "connected",
        "api": "running"
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTPException to return custom error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": False,
            "message": exc.detail
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    print("[API ERROR] Exception occurred:")
    print(f"Request: {request.method} {request.url}")
    print(f"Exception: {exc!r}")
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={
            "status": False,
            "message": "Internal server error"
        }
    )


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )

app.include_router(api_router)