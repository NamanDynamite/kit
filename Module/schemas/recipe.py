from typing import List, Optional, Generic, TypeVar, Any
from pydantic import BaseModel, field_validator, Field
import re
from Module.utils_time import format_datetime_ampm as format_dt

T = TypeVar('T')

class ApiResponse(BaseModel, Generic[T]):
    """Generic API response wrapper."""
    status: bool = Field(..., description="Response status: True on success, False on error")
    message: Optional[str] = Field(None, description="Optional message")
    data: Optional[Any] = Field(None, description="Response data (only present on success)")

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
    views: int
    ingredients: list = []
    steps: list = []

class RecipeSynthesisRequest(BaseModel):
    """Schema for recipe synthesis request."""
    dish_name: str = Field(..., min_length=3, max_length=100, description="Dish name (3-100 chars)")
    servings: int = Field(2, ge=1, le=100, description="Servings must be 1-100")

    @field_validator('dish_name')
    @classmethod
    def validate_dish_name_format(cls, v: str) -> str:
        """Validate dish name format only (content validation happens in service layer)."""
        v = v.strip()
        if not re.match(r"^[a-zA-Z0-9\s\-\.,'&()]+$", v):
            raise ValueError('Dish name can only contain letters, numbers, spaces, hyphens, dots, commas, apostrophes, ampersands, and parentheses')
        return v

class RecipeScoreResponse(BaseModel):
    """Schema for recipe score response (all scores displayed as 0-5 scale)."""
    rating: float = Field(..., description="User rating on 0-5 scale")
    ingredient_authenticity_score: float = Field(..., description="Ingredient authenticity on 0-5 scale")
    serving_scalability_score: float = Field(..., description="Serving scalability on 0-5 scale")
    popularity_score: float = Field(..., description="Popularity score on 0-5 scale")
    ai_confidence_score: float = Field(..., description="Overall AI confidence on 0-5 scale")
    final_score: float = Field(..., description="Final composite score on 0-5 scale")
    calculated_at: str

    @classmethod
    def from_db(cls, db_score):
        """Convert scores from database (already 0-5 scale) to response format."""
        if not db_score:
            return None
        return cls(
            rating=round((db_score.rating or 0), 2),
            ingredient_authenticity_score=round((db_score.ingredient_authenticity_score or 0), 2),
            serving_scalability_score=round((db_score.serving_scalability_score or 0), 2),
            popularity_score=round((db_score.popularity_score or 0), 2),
            ai_confidence_score=round((db_score.ai_confidence_score or 0), 2),
            final_score=round((db_score.final_score or 0), 2),
            calculated_at=format_dt(db_score.calculated_at) if db_score.calculated_at else None
        )

class RatingResponse(BaseModel):
    """Schema for recipe rating response."""
    recipe_id: str
    version_id: str
    user_id: str
    title: str
    servings: int
    rating: float = Field(..., description="Rating given by user on 0-5 scale")
    avg_rating: float = Field(..., description="Average rating across all users on 0-5 scale")
    comment: str
    created_at: str

class ValidationResponse(BaseModel):
    """Schema for recipe validation response."""
    validation_id: str
    version_id: str
    validated_at: str
    approved: bool
    feedback: str
