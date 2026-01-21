from typing import List, Optional
from pydantic import BaseModel, field_validator, Field
import re

class IngredientCreate(BaseModel):
    """Schema for creating an ingredient."""
    name: str = Field(..., min_length=2, max_length=50, description="Ingredient name (2-50 chars)")
    quantity: float = Field(..., gt=0, description="Quantity must be greater than 0")
    unit: str = Field(..., min_length=1, max_length=20, description="Unit of measurement (1-20 chars)")

    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate ingredient name contains only letters, numbers, spaces, hyphens."""
        if not re.match(r"^[a-zA-Z0-9\s\-\.,']+$", v):
            raise ValueError('Ingredient name can only contain letters, numbers, spaces, hyphens, dots, commas, and apostrophes')
        return v.strip()

    @field_validator('unit')
    @classmethod
    def validate_unit(cls, v: str) -> str:
        """Validate unit contains only letters and abbreviations."""
        if not re.match(r"^[a-zA-Z0-9\s\-\.]+$", v):
            raise ValueError('Unit can only contain letters, numbers, spaces, hyphens, and dots')
        return v.strip().lower()

class RecipeCreate(BaseModel):
    """Schema for creating a recipe."""
    title: str = Field(..., min_length=3, max_length=100, description="Recipe title (3-100 chars)")
    ingredients: List[IngredientCreate] = Field(..., min_length=1, description="At least 1 ingredient required")
    steps: List[str] = Field(..., min_length=1, description="At least 1 step required")
    servings: int = Field(..., ge=1, le=100, description="Servings must be 1-100")

    @field_validator('title')
    @classmethod
    def validate_title(cls, v: str) -> str:
        """Validate recipe title format."""
        if not re.match(r"^[a-zA-Z0-9\s\-\.,'&()]+$", v):
            raise ValueError('Recipe title can only contain letters, numbers, spaces, hyphens, dots, commas, apostrophes, ampersands, and parentheses')
        return v.strip()

    @field_validator('steps')
    @classmethod
    def validate_steps(cls, v: List[str]) -> List[str]:
        """Validate each step has minimum length."""
        for i, step in enumerate(v):
            if not step or len(step.strip()) < 5:
                raise ValueError(f'Step {i+1} must be at least 5 characters long')
            if len(step) > 500:
                raise ValueError(f'Step {i+1} cannot exceed 500 characters')
        return [step.strip() for step in v]

class RecipeResponse(BaseModel):
    """Schema for recipe response."""
    recipe_id: str
    version_id: Optional[str] = None
    title: str
    servings: int
    approved: bool
    popularity: int
    ingredients: list = []
    steps: list = []

class RecipeSynthesisRequest(BaseModel):
    """Schema for recipe synthesis request."""
    dish_name: str
    servings: int = 2
    ingredients: Optional[List[IngredientCreate]] = None
    steps: Optional[List[str]] = None

class ValidationResponse(BaseModel):
    """Schema for recipe validation response."""
    validation_id: str
    version_id: str
    validated_at: str
    approved: bool
    feedback: str
