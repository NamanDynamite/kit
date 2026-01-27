import uuid
import os
from datetime import datetime
from typing import List
from sqlalchemy.orm import Session
from pydantic import parse_obj_as

from Module.database import Recipe as DBRecipe, RecipeVersion, Validation, RecipeScore, User
from Module.repository_postgres import PostgresRecipeRepository
from Module.utils_time import format_datetime_ampm as format_dt, get_india_time
from Module.schemas.recipe import (
    RecipeCreate, RecipeResponse, RecipeSynthesisRequest, 
    ValidationResponse, IngredientCreate, RecipeScoreResponse
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
        
        # Check if a recipe with this title already exists for this trainer
        # Allow same title with different servings as separate versions, but block exact duplicate
        existing_recipe = self.db.query(DBRecipe).filter(
            DBRecipe.dish_name == recipe.title,
            DBRecipe.created_by == trainer_id
        ).first()
        
        if existing_recipe:
            # If a version with the same servings already exists, block duplicate submission
            duplicate_version = self.db.query(RecipeVersion).filter(
                RecipeVersion.recipe_id == existing_recipe.recipe_id,
                RecipeVersion.base_servings == recipe.servings
            ).first()
            if duplicate_version:
                raise ValueError("A recipe with the same title and servings already exists for this trainer. Please edit the existing recipe or choose different servings.")

            print(f"[DEBUG] Recipe '{recipe.title}' exists; adding as new version with different servings")
            # Add new version for different servings
            recipe_obj = self.repo.add_version_to_recipe(
                recipe_id=existing_recipe.recipe_id,
                ingredients=[{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} 
                            for ing in recipe.ingredients],
                steps=recipe.steps,
                servings=recipe.servings,
                submitted_by=trainer_id
            )
        else:
            print(f"[DEBUG] Creating new recipe '{recipe.title}'")
            # Create new recipe only if it doesn't exist
            if recipe.ingredients and isinstance(recipe.ingredients[0], dict):
                ingredients_obj = parse_obj_as(List[IngredientCreate], recipe.ingredients)
            else:
                ingredients_obj = recipe.ingredients
            
            recipe_obj = self.repo.create_recipe(
                title=recipe.title,
                ingredients=ingredients_obj,
                steps=recipe.steps,
                servings=recipe.servings,
                submitted_by=trainer_id
            )
        
        version_id = None
        db_recipe = self.db.query(DBRecipe).filter(DBRecipe.recipe_id == recipe_obj.id).first()
        if db_recipe and db_recipe.versions:
            version_id = db_recipe.versions[-1].version_id

        # Auto-validate immediately after submission
        approved_status = False
        if version_id:
            try:
                print(f"[DEBUG] Auto-validation starting for version_id={version_id}")
                self.validate_recipe(version_id)
                latest_validation = (
                    self.db.query(Validation)
                    .filter(Validation.version_id == version_id)
                    .order_by(Validation.validated_at.desc())
                    .first()
                )
                approved_status = latest_validation.approved if latest_validation else False
                print(f"[DEBUG] Auto-validation finished, approved={approved_status}")
            except Exception as val_e:
                print(f"[WARN] Auto validation failed: {val_e}")
                import traceback
                traceback.print_exc()
        
        # Get views from the created version
        views = 0
        if version_id:
            version = self.db.query(RecipeVersion).filter(RecipeVersion.version_id == version_id).first()
            views = version.views if version and version.views is not None else 0
        
        return RecipeResponse(
            recipe_id=recipe_obj.id,
            version_id=version_id,
            title=recipe_obj.title,
            servings=recipe.servings,
            approved=approved_status,
            views=views,
            ingredients=[{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} for ing in recipe.ingredients],
            steps=recipe.steps
        )
    
    def list_recipes(self, approved_only: bool = True) -> List[RecipeResponse]:
        """List recipes filtered by approval status.

        approved_only=True  -> only approved (published)
        approved_only=False -> only pending/unapproved
        """
        recipes = self.repo.approved() if approved_only else self.repo.pending()
        response = []
        for r in recipes:
            version_id = None
            views = 0
            db_recipe = self.db.query(DBRecipe).filter(DBRecipe.recipe_id == r.id).first()
            if db_recipe and db_recipe.versions:
                latest_version = db_recipe.versions[-1]
                version_id = latest_version.version_id
                views = latest_version.views if latest_version.views is not None else 0
            response.append(RecipeResponse(
                recipe_id=r.id,
                version_id=version_id,
                title=r.title,
                servings=r.servings,
                approved=r.approved,
                views=views,
                ingredients=[i for i in getattr(r, 'ingredients', [])],
                steps=[s for s in getattr(r, 'steps', [])]
            ))
        return response
    
    def synthesize_recipe(self, request: RecipeSynthesisRequest, user_id: str) -> RecipeResponse:
        """Synthesize multiple recipes into one."""
        print(f"[DEBUG] synthesize_recipe called with dish_name='{request.dish_name}', servings={request.servings}, user_id={user_id}")
        from Module.database import User as DBUser
        user = self.db.query(DBUser).filter(DBUser.user_id == user_id).first()
        if not user:
            raise ValueError("No user found with the provided user ID")
        
        # Check if dish_name exists in database; if not, find similar one
        dish_name = request.dish_name
        existing_recipes = self.db.query(DBRecipe).distinct(DBRecipe.dish_name).all()
        existing_dish_names = [r.dish_name for r in existing_recipes if r.dish_name]
        
        # Try exact match (case-insensitive)
        exact_match = None
        for existing_name in existing_dish_names:
            if existing_name.lower() == dish_name.lower():
                exact_match = existing_name
                break
        
        if exact_match:
            dish_name = exact_match
            print(f"[DEBUG] Found exact match (case-insensitive): {dish_name}")
        else:
            # Find similar dish name using fuzzy matching (case-insensitive)
            import difflib
            existing_names_lower = [name.lower() for name in existing_dish_names]
            close_matches = difflib.get_close_matches(dish_name.lower(), existing_names_lower, n=1, cutoff=0.5)
            if close_matches:
                # Get original name with proper casing
                matched_idx = existing_names_lower.index(close_matches[0])
                dish_name = existing_dish_names[matched_idx]
                print(f"[DEBUG] Found similar match for '{request.dish_name}': {dish_name}")
            else:
                print(f"[DEBUG] No similar match found for '{request.dish_name}'")
        
        # Update request with matched/corrected dish_name
        request.dish_name = dish_name
        
        # Validate dish_name is genuine (after user_id is verified)
        dish_name = request.dish_name.lower()
        placeholder_values = {
            "string", "test", "dish", "recipe", "food", "meal", "dish name",
            "placeholder", "example", "sample", "demo", "x", "y", "z",
            "aaa", "bbb", "ccc", "ddd", "eee", "fff", "hhh", "iii", "jjj"
        }
        
        if dish_name in placeholder_values:
            raise ValueError('Please provide a genuine dish name (e.g., "Pasta Carbonara", "Biryani", "Tacos"), not placeholder text')
        
        # Reject names that are just repeated characters
        if len(set(dish_name.replace(" ", ""))) == 1:
            raise ValueError('Dish name must contain varied characters, not just repetition')
        
        # Reject very generic single-word dishes if they're too short
        if len(request.dish_name.split()) == 1 and len(request.dish_name) < 4:
            raise ValueError('Single-word dish names must be at least 4 characters long')
        
        from api import km_instance
        kwargs = {
            'user': user,
            'dish_name': request.dish_name,
            'servings': request.servings
        }
        
        # Check if a recipe for this dish already exists (exact match across all users)
        print(f"[DEBUG] Searching for existing recipes with dish_name='{request.dish_name}'")
        
        existing_recipes = self.db.query(DBRecipe).filter(
            DBRecipe.dish_name == request.dish_name
        ).all()
        
        print(f"[DEBUG] Found {len(existing_recipes)} recipe(s) matching dish name '{request.dish_name}'")
        for r in existing_recipes:
            print(f"[DEBUG]   - {r.recipe_id}: {r.dish_name} (servings={r.servings}, published={r.is_published})")
        
        existing_recipe = None
        if existing_recipes:
            # Prefer published recipes as base, fallback to first unpublished
            existing_recipe = next((r for r in existing_recipes if r.is_published), existing_recipes[0])
            print(f"[DEBUG] Using recipe {existing_recipe.recipe_id} as base for versioning")
            
            # Check if a version with this EXACT servings already exists
            existing_version = self.db.query(RecipeVersion).filter(
                RecipeVersion.recipe_id == existing_recipe.recipe_id,
                RecipeVersion.base_servings == request.servings
            ).first()
            
            if existing_version:
                print(f"[DEBUG] Version with servings={request.servings} already exists, returning it (deduplication)")
                # Return existing version (deduplication - same dish, same servings)
                ingredients = [{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} for ing in existing_version.ingredients]
                steps = [step.instruction for step in sorted(existing_version.steps, key=lambda x: x.step_order)]
                views = existing_version.views if existing_version.views is not None else 0
                
                return RecipeResponse(
                    recipe_id=existing_recipe.recipe_id,
                    version_id=existing_version.version_id,
                    title=existing_recipe.dish_name,
                    servings=existing_version.base_servings,
                    approved=existing_recipe.is_published,
                    views=views,
                    ingredients=ingredients,
                    steps=steps
                )
            else:
                print(f"[DEBUG] No version found for servings={request.servings}, will add a new version to recipe {existing_recipe.recipe_id}")
        
        # Synthesize the recipe
        print(f"[DEBUG] Synthesizing recipe for {request.dish_name} with {request.servings} servings")
        result = km_instance.request_recipe(**kwargs)
        
        from Module.controller import ensure_recipe_dataclass
        result = ensure_recipe_dataclass(result)
        
        if not hasattr(result, 'ingredients') or not isinstance(result.ingredients, (list, tuple)):
            raise RuntimeError("Problem generating the recipe")
        
        try:
            ings = result.ingredients
        except Exception:
            raise RuntimeError("Problem accessing the recipe ingredients")
        
        # If recipe exists, add a new version to it; otherwise create new recipe
        if existing_recipe:
            print(f"[DEBUG] Adding version to existing recipe {existing_recipe.recipe_id}")
            recipe_obj = self.repo.add_version_to_recipe(
                recipe_id=existing_recipe.recipe_id,
                ingredients=[{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} for ing in ings],
                steps=result.steps,
                servings=request.servings,
                submitted_by=user.user_id
            )
            # Get the newly created version ID (latest version)
            db_recipe = self.db.query(DBRecipe).filter(DBRecipe.recipe_id == existing_recipe.recipe_id).first()
            version_id = db_recipe.versions[-1].version_id if db_recipe and db_recipe.versions else None
            print(f"[DEBUG] New version_id: {version_id}")
        else:
            print(f"[DEBUG] Creating new recipe for {request.dish_name}")
            recipe_obj = self.repo.create_recipe(
                title=request.dish_name,
                ingredients=[{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} for ing in ings],
                steps=result.steps,
                servings=request.servings,
                submitted_by=user.user_id,
                approved=getattr(result, 'approved', False)
            )
            version_id = None
            db_recipe = self.db.query(DBRecipe).filter(DBRecipe.recipe_id == recipe_obj.id).first()
            if db_recipe and db_recipe.versions:
                version_id = db_recipe.versions[-1].version_id
            print(f"[DEBUG] New recipe_id: {recipe_obj.id}, version_id: {version_id}")
        
        views = 0
        if version_id:
            version = self.db.query(RecipeVersion).filter(RecipeVersion.version_id == version_id).first()
            views = version.views if version and version.views is not None else 0
        
        return RecipeResponse(
            recipe_id=recipe_obj.id,
            version_id=version_id,
            title=recipe_obj.title,
            servings=recipe_obj.servings,
            approved=recipe_obj.approved,
            views=views,
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
        
        api_key_present = bool(os.environ.get("OPENAI_API_KEY"))
        print(f"[DEBUG] validate_recipe called for version_id={version_id}, api_key_present={api_key_present}")
        approved, feedback, confidence = ai_validate_recipe(
            recipe.dish_name,
            [f"{ing.name} {ing.quantity} {ing.unit}" for ing in version.ingredients],
            [step.instruction for step in version.steps]
        )
        print(f"[DEBUG] ai_validate_recipe returned approved={approved}, confidence={confidence}")
        
        print(f"[DEBUG] Persisting validation: version_id={version_id}, approved={approved}, confidence={confidence}")
        validation = Validation(
            validation_id=str(uuid.uuid4()),
            version_id=version_id,
            validated_at=get_india_time(),
            approved=approved,
            feedback=feedback
        )
        self.db.add(validation)
        
        # Store version-specific AI confidence score (convert 0-1 to 0-5 scale)
        version.ai_confidence_score = confidence * 5.0
        
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
                # Use version-level views for popularity
                self.popularity = getattr(version, 'views', 0)
                self.validator_confidence = confidence
                self.metadata = {'ai_confidence': confidence}
                self.ingredients = version.ingredients
                self.steps = version.steps
            def avg_rating(self):
                return 0.0
        
        mock_recipe = MockRecipe(recipe, version)
        ai_scores = {
            'ingredient_authenticity_score': scorer.ingredient_authenticity_score(mock_recipe),
            'serving_scalability_score': scorer.serving_scalability_score(mock_recipe),
            'ai_confidence_score': confidence * 5.0  # Convert 0-1 to 0-5 scale
        }
        popularity_score = scorer.popularity_score(mock_recipe)
        # Update score for this specific version
        update_recipe_score(self.db, recipe.recipe_id, ai_scores=ai_scores, popularity=popularity_score, version_id=version_id)
        
        return ValidationResponse(
            validation_id=validation.validation_id,
            version_id=validation.version_id,
            validated_at=format_dt(validation.validated_at),
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
        
        # Use version.views (integer count) not score.popularity_score (float 0-5)
        views = version.views if version.views is not None else 0
        
        return RecipeResponse(
            recipe_id=recipe.recipe_id,
            version_id=version.version_id,
            title=recipe.dish_name,
            servings=version.base_servings if hasattr(version, 'base_servings') and version.base_servings else getattr(recipe, 'servings', 1),
            approved=approved,
            views=views,
            ingredients=[{"name": ing.name, "quantity": ing.quantity, "unit": ing.unit} for ing in getattr(version, 'ingredients', [])],
            steps=[step.instruction for step in sorted(getattr(version, 'steps', []), key=lambda x: x.step_order)]
        )
    
    def rate_recipe(self, version_id: str, user_id: str, rating: float, comment: str = None) -> dict:
        """Rate a recipe version. Prevent trainers from rating their own recipes."""
        version = self.db.query(RecipeVersion).filter(RecipeVersion.version_id == version_id).first()
        if not version:
            raise ValueError("No recipe version found with the provided ID")

        recipe_id = version.recipe_id

        user = self.db.query(User).filter(User.user_id == user_id).first()
        if not user:
            raise ValueError("No user found with the provided user ID")

        # Prevent trainers from rating their own recipes
        recipe = self.db.query(DBRecipe).filter(DBRecipe.recipe_id == recipe_id).first()
        if recipe and recipe.created_by == user_id:
            raise PermissionError("You cannot rate your own recipe.")

        feedback = self.repo.add_rating(version_id, user_id, rating, comment)

        from Module.database import update_recipe_score, update_trainer_rating_score
        update_recipe_score(self.db, recipe_id, version_id=version_id)

        # After updating recipe score, also update the trainer's rating_score
        # Find the creator of the recipe
        trainer_id = None
        if recipe:
            trainer_id = recipe.created_by
        if trainer_id:
            try:
                update_trainer_rating_score(self.db, trainer_id)
            except Exception as e:
                print(f"[WARN] Could not update trainer rating_score: {e}")

        score = self.db.query(RecipeScore).filter(
            RecipeScore.recipe_id == recipe_id,
            RecipeScore.version_id == version_id
        ).first()
        # Both feedback and rating are on 0-5 scale
        avg_rating = (score.rating or 0.0) if score else 0.0

        return {
            "recipe_id": recipe_id,
            "version_id": version_id,
            "user_id": user_id,
            "title": version.recipe.dish_name if version.recipe else None,
            "servings": version.base_servings if hasattr(version, 'base_servings') and version.base_servings else None,
            "rating": round(rating, 2),
            "avg_rating": round(avg_rating, 2),
            "comment": feedback.comment or "",
            "created_at": format_dt(feedback.created_at) if feedback.created_at else None
        }
    def increment_views(self, version_id: str) -> int:
        """Increment recipe version views by 1 and recalculate score."""
        version = self.db.query(RecipeVersion).filter(RecipeVersion.version_id == version_id).first()
        if not version:
            raise ValueError(f"Recipe version with ID {version_id} not found")
        
        version.views = (version.views or 0) + 1
        self.db.commit()
        
        # Recalculate popularity score for this version
        from Module.database import update_recipe_score
        
        # Calculate new popularity_score based on updated views count
        from Module.scoring import ScoringEngine
        scorer = ScoringEngine()
        from Module.models import Recipe as RecipeModel
        mock_recipe = RecipeModel(
            id=version.recipe_id,
            title=version.recipe.dish_name,
            ingredients=[],
            steps=[],
            servings=version.base_servings,
            popularity=version.views
        )
        popularity_score = scorer.popularity_score(mock_recipe)
        # Ensure correct parameter mapping using keywords
        update_recipe_score(self.db, version.recipe_id, popularity=popularity_score, version_id=version_id)
        
        return version.views