from typing import List

from fastapi import Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from Module.database import get_db, Recipe
from Module.routers.base import api_router

class RecipeResponse(BaseModel):
    recipe_id: str
    version_id: str | None = None
    title: str
    servings: int
    approved: bool
    popularity: int
    ingredients: list = []
    steps: list = []

@api_router.get("/public/recipes")
def public_recipes(db: Session = Depends(get_db)):
    recipes = db.query(Recipe).filter(Recipe.is_published == True).limit(2).all()
    result = []
    for r in recipes:
        result.append(RecipeResponse(
            recipe_id=r.recipe_id,
            version_id=getattr(r, 'current_version_id', None),
            title=getattr(r, 'title', getattr(r, 'dish_name', None)),
            servings=r.servings,
            approved=getattr(r, 'is_published', False),
            popularity=getattr(r, 'popularity', 0),
            ingredients=[{'name': ing.name, 'quantity': ing.quantity, 'unit': ing.unit} for ing in getattr(r, 'ingredients', [])],
            steps=getattr(r, 'steps', [])
        ))
    return {
        "status": True,
        "message": "Public recipes fetched successfully.",
        "data": result
    }
