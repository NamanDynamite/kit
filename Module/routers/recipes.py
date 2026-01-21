from typing import List
import uuid

from fastapi import Body, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from Module.database import get_db, User
from Module.routers.base import api_router
from Module.schemas.recipe import (
    RecipeCreate, RecipeResponse, RecipeSynthesisRequest, ValidationResponse
)
from Module.services.recipe_service import RecipeService

@api_router.post("/recipe")
def submit_recipe(
    recipe: RecipeCreate,
    trainer_id: str = Query(..., description="Trainer user ID (UUID)"),
    db: Session = Depends(get_db)
):
    try:
        # Validate trainer_id format
        if not trainer_id or trainer_id.strip() == "":
            raise HTTPException(status_code=400, detail="trainer_id is required")
        try:
            uuid.UUID(trainer_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid trainer ID format. Please provide a valid user identifier.")
        
        # Validate trainer exists
        trainer = db.query(User).filter(User.user_id == trainer_id).first()
        if not trainer:
            raise HTTPException(status_code=404, detail="Trainer user not found")
        
        service = RecipeService(db)
        result = service.submit_recipe(recipe, trainer_id)
        return {
            "status": True,
            "message": "Recipe submitted successfully.",
            "data": result
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        print(f"[ERROR] submit_recipe exception: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while submitting the recipe. Please try again.")

@api_router.get("/recipes")
def list_recipes(
    approved_only: bool = Query(True),
    db: Session = Depends(get_db)
):
    service = RecipeService(db)
    response = service.list_recipes(approved_only)
    return {
        "status": True,
        "message": "Recipes fetched successfully.",
        "data": response
    }

@api_router.post("/recipe/synthesize")
def synthesize_recipe(
    request: RecipeSynthesisRequest,
    user_id: str = Query(...),
    db: Session = Depends(get_db)
):
    try:
        service = RecipeService(db)
        return service.synthesize_recipe(request, user_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print(f"[ERROR] synthesize_recipe exception: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while synthesizing the recipe. Please try again later.")

@api_router.get("/recipes/pending")
def get_pending_recipes(db: Session = Depends(get_db)):
    service = RecipeService(db)
    return service.get_pending_recipes()

@api_router.post("/recipe/version/{version_id}/validate")
def ai_review_recipe(version_id: str, db: Session = Depends(get_db)):
    try:
        service = RecipeService(db)
        return service.validate_recipe(version_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@api_router.get("/recipe/version/{version_id}")
def get_single_recipe_by_version(version_id: str, db: Session = Depends(get_db)):
    try:
        service = RecipeService(db)
        return service.get_recipe_by_version(version_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@api_router.post("/recipe/version/{version_id}/rate")
def rate_recipe(
    version_id: str,
    user_id: str = Query(...),
    rating: float = Query(...),
    comment: str = Body(default=None, embed=True),
    db: Session = Depends(get_db)
):
    try:
        service = RecipeService(db)
        return service.rate_recipe(version_id, user_id, rating, comment)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
