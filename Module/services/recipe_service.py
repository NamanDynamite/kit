import uuid
from datetime import datetime
from typing import List
from sqlalchemy.orm import Session
from pydantic import parse_obj_as

from Module.database import Recipe as DBRecipe, RecipeVersion, Validation, RecipeScore, User
from Module.repository_postgres import PostgresRecipeRepository
from Module.schemas.recipe import (
    RecipeCreate, RecipeResponse, RecipeSynthesisRequest, 
    ValidationResponse, IngredientCreate
)

class RecipeService:
    """Service for recipe-related business logic."""
    
    def __init__(self, db: Session):
        self.db = db
        self.repo = PostgresRecipeRepository(db)
    
    def submit_recipe(self, recipe: RecipeCreate, trainer_id: str) -> RecipeResponse:
        """Submit a new recipe (trainer only)."""
        trainer = self.db.query(User).filter(User.user_id == trainer_id).first()
        if not trainer:
            raise ValueError("Trainer not found")
        
        trainer_role = trainer.role
        if hasattr(trainer_role, 'role_id'):
            trainer_role = trainer_role.role_id
        elif hasattr(trainer_role, 'value'):
            trainer_role = trainer_role.value
        
        if str(trainer_role).lower() not in ["trainer", "admin"]:
            raise PermissionError("Only trainers can submit recipes")
        
        if recipe.ingredients and isinstance(recipe.ingredients[0], dict):
            ingredients_obj = parse_obj_as(List[IngredientCreate], recipe.ingredients)
        else:
            ingredients_obj = recipe.ingredients
        
        recipe_obj = self.repo.create_recipe(
            title=recipe.title,
            ingredients=ingredients_obj,
            steps=recipe.steps,
            servings=recipe.servings,
            submitted_by=trainer.user_id
        )
        
        # Sync to in-memory store
        try:
            from api import km_instance
            from Module.models import Recipe, Ingredient
            mem_recipe = Recipe(
                recipe_id=recipe_obj.id,
                title=recipe_obj.title,
                ingredients=[Ingredient(name=ing.name, quantity=ing.quantity, unit=ing.unit) for ing in recipe_obj.ingredients],
                steps=recipe_obj.steps,
                servings=recipe_obj.servings,
                metadata=getattr(recipe_obj, 'metadata', {}),
                ratings=getattr(recipe_obj, 'ratings', []),
                validator_confidence=getattr(recipe_obj, 'validator_confidence', 0.0),
                popularity=getattr(recipe_obj, 'popularity', 0),
                approved=recipe_obj.approved,
                rejection_suggestions=getattr(recipe_obj, 'rejection_suggestions', [])
            )
            km_instance.recipes.add(mem_recipe)
            km_instance.vstore.index(mem_recipe)
        except Exception as sync_e:
            print(f"[ERROR] Failed to sync recipe to in-memory store: {sync_e}")
        
        version_id = None
        db_recipe = self.db.query(DBRecipe).filter(DBRecipe.recipe_id == recipe_obj.id).first()
        if db_recipe and hasattr(db_recipe, 'current_version_id'):
            version_id = db_recipe.current_version_id

        # Auto-validate immediately after submission so trainers see approved recipes without extra calls
        if version_id:
            try:
                self.validate_recipe(version_id)
                # Refresh recipe_obj to reflect any approval/is_published changes
                self.db.refresh(recipe_obj)
            except Exception as val_e:
                # Keep submission successful even if validation fails; recipe remains pending
                print(f"[WARN] Auto validation failed for version {version_id}: {val_e}")
        
        return RecipeResponse(
            recipe_id=recipe_obj.id,
            version_id=version_id,
            title=recipe_obj.title,
            servings=recipe_obj.servings,
            approved=getattr(recipe_obj, 'approved', getattr(recipe_obj, 'is_published', False)),
            popularity=getattr(recipe_obj, 'popularity', 0),
            ingredients=[{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} for ing in getattr(recipe_obj, 'ingredients', [])],
            steps=getattr(recipe_obj, 'steps', [])
        )
    
    def list_recipes(self, approved_only: bool = True) -> List[RecipeResponse]:
        """List recipes, optionally filtering by approval status."""
        recipes = self.repo.approved() if approved_only else self.repo.list()
        response = []
        for r in recipes:
            version_id = None
            db_recipe = self.db.query(DBRecipe).filter(DBRecipe.recipe_id == r.id).first()
            if db_recipe and hasattr(db_recipe, 'current_version_id'):
                version_id = db_recipe.current_version_id
            response.append(RecipeResponse(
                recipe_id=r.id,
                version_id=version_id,
                title=r.title,
                servings=r.servings,
                approved=r.approved,
                popularity=getattr(r, "popularity", 0),
                ingredients=[i for i in getattr(r, 'ingredients', [])],
                steps=[s for s in getattr(r, 'steps', [])]
            ))
        return response
    
    def synthesize_recipe(self, request: RecipeSynthesisRequest, user_id: str) -> RecipeResponse:
        """Synthesize multiple recipes into one."""
        from Module.database import User as DBUser
        user = self.db.query(DBUser).filter(DBUser.user_id == user_id).first()
        if not user:
            raise ValueError("No user found with the provided user ID")
        
        from api import km_instance
        kwargs = {
            'user': user,
            'dish_name': request.dish_name,
            'servings': request.servings
        }
        if request.ingredients is not None:
            kwargs['ingredients'] = request.ingredients
        result = km_instance.request_recipe(**kwargs)
        
        from Module.controller import ensure_recipe_dataclass
        result = ensure_recipe_dataclass(result)
        
        if not hasattr(result, 'ingredients') or not isinstance(result.ingredients, (list, tuple)):
            raise RuntimeError("Problem generating the recipe")
        
        try:
            ings = result.ingredients
        except Exception:
            raise RuntimeError("Problem accessing the recipe ingredients")
        
        recipe_obj = self.repo.create_recipe(
            title=result.title,
            ingredients=[{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} for ing in ings],
            steps=result.steps,
            servings=result.servings,
            submitted_by=user.user_id,
            approved=getattr(result, 'approved', False)
        )
        
        version_id = None
        db_recipe = self.db.query(DBRecipe).filter(DBRecipe.recipe_id == recipe_obj.id).first()
        if db_recipe and hasattr(db_recipe, 'current_version_id'):
            version_id = db_recipe.current_version_id
        
        return RecipeResponse(
            recipe_id=recipe_obj.id,
            version_id=version_id,
            title=recipe_obj.title,
            servings=recipe_obj.servings,
            approved=recipe_obj.approved,
            popularity=getattr(recipe_obj, 'popularity', 0),
            ingredients=[{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} for ing in getattr(recipe_obj, 'ingredients', [])],
            steps=getattr(recipe_obj, 'steps', [])
        )
    
    def get_pending_recipes(self) -> List[dict]:
        """Get all pending (unapproved) recipes."""
        recipes = self.repo.pending()
        return [
            {
                "id": r.id,
                "title": r.title,
                "servings": r.servings,
                "submitted_by": r.metadata.get("submitted_by", "unknown")
            }
            for r in recipes
        ]
    
    def validate_recipe(self, version_id: str) -> ValidationResponse:
        """Review a recipe version using AI validation."""
        from Module.ai_validation import ai_validate_recipe
        
        version = self.db.query(RecipeVersion).filter(RecipeVersion.version_id == version_id).first()
        if not version:
            raise ValueError("No recipe version found with the provided ID")
        
        recipe = version.recipe
        if not recipe:
            raise ValueError("No parent recipe found for this version")
        
        approved, feedback, confidence = ai_validate_recipe(
            recipe.dish_name,
            [f"{ing.name} {ing.quantity} {ing.unit}" for ing in version.ingredients],
            [step.instruction for step in version.steps]
        )
        
        validation = Validation(
            validation_id=str(uuid.uuid4()),
            version_id=version_id,
            validated_at=datetime.utcnow(),
            approved=approved,
            feedback=feedback
        )
        self.db.add(validation)
        
        if approved:
            recipe.is_published = True
        
        self.db.commit()
        self.db.refresh(validation)
        self.db.refresh(version)
        self.db.refresh(recipe)
        
        # Update recipe_scores
        from Module.scoring import ScoringEngine
        from Module.database import update_recipe_score
        scorer = ScoringEngine()
        
        class MockRecipe:
            def __init__(self, recipe, version):
                self.recipe_id = recipe.recipe_id
                self.dish_name = recipe.dish_name
                self.servings = recipe.servings
                self.popularity = getattr(recipe, 'popularity', 0)
                self.validator_confidence = confidence
                self.metadata = {'ai_confidence': confidence}
                self.ingredients = version.ingredients
                self.steps = version.steps
            def avg_rating(self):
                return 0.0
        
        mock_recipe = MockRecipe(recipe, version)
        ai_scores = {
            'validator_confidence_score': scorer.score(mock_recipe),
            'ingredient_authenticity_score': scorer.ingredient_authenticity_score(mock_recipe),
            'serving_scalability_score': scorer.serving_scalability_score(mock_recipe),
            'ai_confidence_score': scorer.ai_confidence_score(mock_recipe)
        }
        popularity_score = scorer.popularity_score(mock_recipe)
        update_recipe_score(self.db, recipe.recipe_id, ai_scores=ai_scores, popularity=popularity_score)
        
        return ValidationResponse(
            validation_id=validation.validation_id,
            version_id=validation.version_id,
            validated_at=validation.validated_at.isoformat(),
            approved=validation.approved,
            feedback=validation.feedback
        )
    
    def get_recipe_by_version(self, version_id: str) -> RecipeResponse:
        """Get a single recipe by version ID."""
        version = self.db.query(RecipeVersion).filter(RecipeVersion.version_id == version_id).first()
        if not version:
            raise ValueError("No recipe version found with the provided ID")
        
        recipe = version.recipe
        if not recipe:
            raise ValueError("No parent recipe found for this version")
        
        latest_validation = (
            self.db.query(Validation)
            .filter(Validation.version_id == version.version_id)
            .order_by(Validation.validated_at.desc())
            .first()
        )
        approved = latest_validation.approved if latest_validation and latest_validation.approved is not None else False
        
        score = self.db.query(RecipeScore).filter(RecipeScore.recipe_id == recipe.recipe_id).first()
        popularity = score.popularity_score if score and score.popularity_score is not None else getattr(recipe, "popularity", 0)
        
        return RecipeResponse(
            recipe_id=recipe.recipe_id,
            version_id=version.version_id,
            title=recipe.dish_name,
            servings=version.base_servings if hasattr(version, 'base_servings') and version.base_servings else getattr(recipe, 'servings', 1),
            approved=approved,
            popularity=popularity,
            ingredients=[{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} for ing in getattr(version, 'ingredients', [])],
            steps=[step.instruction for step in sorted(getattr(version, 'steps', []), key=lambda x: x.step_order)]
        )
    
    def rate_recipe(self, version_id: str, user_id: str, rating: float, comment: str = None) -> dict:
        """Rate a recipe version."""
        version = self.db.query(RecipeVersion).filter(RecipeVersion.version_id == version_id).first()
        if not version:
            raise ValueError("No recipe version found with the provided ID")
        
        recipe_id = version.recipe_id
        
        user = self.db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise ValueError("No user found with the provided user ID")
        
        feedback = self.repo.add_rating(version_id, user_id, rating, comment)
        
        from Module.database import update_recipe_score
        update_recipe_score(self.db, recipe_id)
        
        score = self.db.query(RecipeScore).filter(RecipeScore.recipe_id == recipe_id).first()
        avg_rating = score.user_rating_score if score and score.user_rating_score is not None else 0.0
        
        return {
            "recipe_id": recipe_id,
            "version_id": version_id,
            "title": version.recipe.dish_name if version.recipe else None,
            "servings": version.base_servings if hasattr(version, 'base_servings') and version.base_servings else None,
            "approved": version.recipe.is_published if version.recipe else None,
            "avg_rating": round(avg_rating, 2),
            "comment": feedback.comment
        }
