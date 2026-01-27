"""
PostgreSQL-based repository implementation for KitchenMind.
"""
from typing import List, Optional
from sqlalchemy.orm import Session
import uuid
import datetime
from .database import Recipe as DBRecipe, Ingredient as DBIngredient, Step as DBStep
from Module.models import Recipe as RecipeModel, Ingredient
from Module.utils_time import get_india_time


class PostgresRecipeRepository:
    def find_draft(self, title: str, servings: int, created_by: str):
        """Find a draft (unpublished) recipe by title, servings, and created_by."""
        db_recipe = self.db.query(self.model).filter(
            self.model.dish_name == title,
            self.model.servings == servings,
            self.model.created_by == created_by,
            self.model.is_published == False
        ).first()
        if db_recipe:
            return self._to_model(db_recipe)
        return None

    @staticmethod
    def extract_minutes(instruction):
        import re
        match = re.search(r"(\d+)\s*(minute|min)s?", instruction.lower())
        if match:
            return int(match.group(1))
        return None
    def list(self) -> list:
        """Return all recipes in the database as RecipeModel objects."""
        db_recipes = self.db.query(self.model).all()
        print(f"[DEBUG] list() found {len(db_recipes)} recipes in DB")
        models = [self._to_model(r) for r in db_recipes]
        print(f"[DEBUG] list() returning models: {[m.id for m in models]}")
        return models

    def add_rating(self, version_id: str, user_id: str, rating: float, comment: str = None):
        """Add or update a user's rating and comment for a recipe version in the Feedback table."""
        from Module.database import Feedback
        from datetime import datetime
        # Enforce rating must be between 0 and 5
        if not (0 <= rating <= 5):
            raise ValueError("Rating must be between 0 and 5.")
        feedback = self.db.query(Feedback).filter(Feedback.version_id == version_id, Feedback.user_id == user_id).first()
        if feedback:
            feedback.rating = rating
            if comment is not None:
                feedback.comment = comment
            feedback.is_revised = True
            feedback.revised_at = get_india_time()
            print(f"[DEBUG] add_rating: updated feedback_id={feedback.feedback_id} rating={rating} user_id={user_id} version_id={version_id}")
        else:
            feedback = Feedback(
                feedback_id=str(uuid.uuid4()),
                version_id=version_id,
                user_id=user_id,
                created_at=get_india_time(),
                rating=rating,
                comment=comment
            )
            self.db.add(feedback)
            self.db.flush()  # Ensure feedback is persisted before update_recipe_score queries it
            print(f"[DEBUG] add_rating: created feedback_id={feedback.feedback_id} rating={rating} user_id={user_id} version_id={version_id}")
        self.db.commit()
        self.db.refresh(feedback)
        print(f"[DEBUG] add_rating: committed feedback_id={feedback.feedback_id} rating={feedback.rating}")
        return feedback

    def get_ratings(self, version_id: str):
        """Get all ratings for a recipe version from the Feedback table."""
        from Module.database import Feedback
        feedbacks = self.db.query(Feedback).filter(Feedback.version_id == version_id).all()
        return [min(max(fb.rating, 0), 5) for fb in feedbacks if fb.rating is not None]

    def create_recipe(self, title, ingredients, steps, servings, submitted_by=None, approved=False):
        """Create and persist a new recipe, returning the Recipe model with id. Prevent duplicate drafts."""
        print(f"[DEBUG] create_recipe called with title={title}, servings={servings}, submitted_by={submitted_by}, approved={approved}")
        # If this is a draft (not approved/published), check for existing draft and update it
        if not approved:
            existing_draft = self.find_draft(title, servings, submitted_by)
            if existing_draft:
                print(f"[DEBUG] Existing draft found, updating it: id={getattr(existing_draft, 'id', None)}")
                db_recipe = self.db.query(DBRecipe).filter(DBRecipe.recipe_id == existing_draft.id).first()
                if db_recipe:
                    # Get the latest version
                    version = db_recipe.versions[-1] if db_recipe.versions else None

                    if version:
                        # Clear existing ingredients and steps for this version
                        self.db.query(DBIngredient).filter(DBIngredient.version_id == version.version_id).delete()
                        self.db.query(DBStep).filter(DBStep.version_id == version.version_id).delete()

                        # Normalize ingredient input
                        from Module.models import Ingredient as IngredientModel
                        safe_ingredients = []
                        for ing in ingredients:
                            if isinstance(ing, dict):
                                safe_ingredients.append(IngredientModel(**ing))
                            else:
                                safe_ingredients.append(ing)

                        version.ingredients = [
                            DBIngredient(
                                ingredient_id=str(uuid.uuid4()),
                                version_id=version.version_id,
                                name=ing.name,
                                quantity=ing.quantity,
                                unit=ing.unit
                            ) for ing in safe_ingredients
                        ]

                        version.steps = []
                        for idx, step_text in enumerate(steps):
                            minutes = self.extract_minutes(step_text)
                            version.steps.append(DBStep(
                                step_id=str(uuid.uuid4()),
                                version_id=version.version_id,
                                step_order=idx,
                                instruction=step_text,
                                minutes=minutes
                            ))

                        version.base_servings = servings if servings is not None else db_recipe.servings
                        # Recipe.servings stays immutable (original submission)
                        # version.base_servings holds this version's serving size
                        db_recipe.dish_name = title

                        self.db.commit()
                        self.db.refresh(db_recipe)
                        print(f"[DEBUG] Draft updated and returned: id={getattr(db_recipe, 'recipe_id', None)}")
                        return self._to_model(db_recipe)

                print(f"[DEBUG] Existing draft could not be updated; returning original model")
                return existing_draft
        import datetime
        recipe_id = str(uuid.uuid4())
        version_id = str(uuid.uuid4())
        db_recipe = DBRecipe(
            recipe_id=recipe_id,
            version_id=None,  # Set after version is persisted
            dish_name=title,
            servings=servings if servings is not None else 1,
            created_by=submitted_by,
            is_published=approved,
            created_at=get_india_time()
        )
        print(f"[DEBUG] db_recipe created: {db_recipe}")
        db_version = self._create_version(version_id, recipe_id, submitted_by, servings, ingredients, steps)
        db_recipe.versions.append(db_version)
        # Add version to session FIRST and flush to ensure it's persisted
        self.db.add(db_version)
        self.db.flush()  # Force immediate insertion of RecipeVersion
        # Now set the version_id on the recipe after version is persisted
        db_recipe.version_id = version_id
        self.db.add(db_recipe)
        print(f"[DEBUG] db_version and db_recipe added to session")
        self.db.commit()
        print(f"[DEBUG] db_recipe committed")
        self.db.refresh(db_recipe)
        print(f"[DEBUG] After commit: db_recipe.recipe_id={getattr(db_recipe, 'recipe_id', None)}")
        # Return as Recipe model
        model = self._to_model(db_recipe)
        print(f"[DEBUG] _to_model returned: id={getattr(model, 'id', None)}, model={model}")
        return model

    def _create_version(self, version_id, recipe_id, submitted_by, servings, ingredients, steps):
        from .database import RecipeVersion, Ingredient as DBIngredient, Step as DBStep
        db_version = RecipeVersion(
            version_id=version_id,
            recipe_id=recipe_id,
            submitted_by=submitted_by,
            submitted_at=get_india_time(),
            status="submitted",
            ai_confidence_score=0.0,
            base_servings=servings,
            # avg_rating removed from RecipeVersion (field deleted)
        )
        # Ensure all ingredients are Ingredient objects
        from Module.models import Ingredient as IngredientModel
        safe_ingredients = []
        for ing in ingredients:
            if isinstance(ing, dict):
                safe_ingredients.append(IngredientModel(**ing))
            else:
                safe_ingredients.append(ing)
        db_version.ingredients = [
            DBIngredient(
                ingredient_id=str(uuid.uuid4()),
                version_id=version_id,
                name=ing.name,
                quantity=ing.quantity,
                unit=ing.unit
            ) for ing in safe_ingredients
        ]
        db_version.steps = []
        for idx, step_text in enumerate(steps):
            minutes = self.extract_minutes(step_text)
            db_version.steps.append(DBStep(
                step_id=str(uuid.uuid4()),
                version_id=version_id,
                step_order=idx,
                instruction=step_text,
                minutes=minutes
            ))
        return db_version

    def add_version_to_recipe(self, recipe_id: str, ingredients, steps, servings, submitted_by=None):
        """Add a new version to an existing recipe. Returns the updated Recipe model."""
        import datetime
        db_recipe = self.db.query(DBRecipe).filter(DBRecipe.recipe_id == recipe_id).first()
        if not db_recipe:
            raise ValueError(f"Recipe {recipe_id} not found")
        
        version_id = str(uuid.uuid4())
        db_version = self._create_version(version_id, recipe_id, submitted_by, servings, ingredients, steps)
        db_recipe.versions.append(db_version)
        # Recipe.servings stays immutable (original value)
        # New version's servings stored in version.base_servings
        
        self.db.add(db_version)
        self.db.commit()
        self.db.refresh(db_recipe)
        print(f"[DEBUG] Added version {version_id} to recipe {recipe_id}, new servings: {servings}")
        return self._to_model(db_recipe)

    """Repository using PostgreSQL for persistent storage."""

    def __init__(self, db: Session):
        self.db = db
        self.model = DBRecipe  # Set self.model to the Recipe SQLAlchemy model
    
    def add(self, recipe: RecipeModel):
        """Add a new recipe to the database, with a version, ingredients, and steps."""
        print(f"[DEBUG] PostgresRecipeRepository.add: Adding recipe with id={getattr(recipe, 'id', None)}, title={getattr(recipe, 'title', None)}, approved={getattr(recipe, 'approved', None)}")
        import datetime
        recipe_id = recipe.id if hasattr(recipe, 'id') else str(uuid.uuid4())
        version_id = str(uuid.uuid4())
        created_by = getattr(recipe, 'created_by', None)
        if not created_by:
            created_by = recipe.metadata.get('submitted_by_id') if hasattr(recipe, 'metadata') else None

        # Duplicate check before insert
        existing = self.db.query(DBRecipe).filter(
            DBRecipe.dish_name == recipe.title,
            DBRecipe.servings == (recipe.servings if recipe.servings is not None else 1),
            DBRecipe.created_by == created_by,
            DBRecipe.is_published == recipe.approved
        ).first()
        if existing:
            print(f"[DEBUG] Duplicate recipe detected: id={existing.recipe_id}, title={existing.dish_name}, servings={existing.servings}, created_by={existing.created_by}, is_published={existing.is_published}")
            return self._to_model(existing)

        db_recipe = DBRecipe(
            recipe_id=recipe_id,
            version_id=None,  # Set after version is persisted
            dish_name=recipe.title,
            servings=recipe.servings if recipe.servings is not None else 1,
            created_by=created_by,
            is_published=recipe.approved,
            created_at=get_india_time()
        )
        db_version = self._create_version(
            version_id=version_id,
            recipe_id=recipe_id,
            submitted_by=created_by,
            servings=recipe.servings,
            ingredients=recipe.ingredients,
            steps=recipe.steps
        )
        db_recipe.versions.append(db_version)
        print(f"[DEBUG] PostgresRecipeRepository.add: DBRecipe before add: recipe_id={db_recipe.recipe_id}, is_published={db_recipe.is_published}, created_by={db_recipe.created_by}")
        # Add version to session FIRST and flush to ensure it's persisted
        self.db.add(db_version)
        self.db.flush()  # Force immediate insertion of RecipeVersion
        # Now set the version_id on the recipe after version is persisted
        db_recipe.version_id = version_id
        self.db.add(db_recipe)
        self.db.commit()
        self.db.refresh(db_recipe)
        print(f"[DEBUG] PostgresRecipeRepository.add: DBRecipe after add: recipe_id={db_recipe.recipe_id}, is_published={db_recipe.is_published}, created_by={db_recipe.created_by}")

    def get(self, recipe_id: str) -> Optional[RecipeModel]:
        """Get a recipe by ID."""
        print(f"[DEBUG] get() called with recipe_id: {recipe_id}")
        db_recipe = self.db.query(DBRecipe).filter(DBRecipe.recipe_id == recipe_id).first()
        print(f"[DEBUG] get() db_recipe: {db_recipe}")
        if not db_recipe:
            print(f"[DEBUG] get() did not find recipe with id: {recipe_id}")
            return None
        model = self._to_model(db_recipe)
        print(f"[DEBUG] get() returning model: {model}")
        return model
    
    def find_by_title(self, title: str) -> List[RecipeModel]:
        """Find recipes by title (case-insensitive)."""
        db_recipes = self.db.query(DBRecipe).filter(
            DBRecipe.dish_name.ilike(f"%{title}%")
        ).all()
        return [self._to_model(r) for r in db_recipes]
    
    def pending(self) -> List[RecipeModel]:
        """Get all pending (unapproved) recipes."""
        db_recipes = self.db.query(DBRecipe).filter(DBRecipe.is_published == False).all()
        return [self._to_model(r) for r in db_recipes]
    
    def approved(self) -> List[RecipeModel]:
        """Get all approved recipes with ingredients and steps populated."""
        db_recipes = self.db.query(DBRecipe).filter(DBRecipe.is_published == True).all()
        models = []
        for r in db_recipes:
            model = self._to_model(r)
            # Ensure ingredients and steps are not None
            if model.ingredients is None:
                model.ingredients = []
            if model.steps is None:
                model.steps = []
            models.append(model)
        return models
    
    def update(self, recipe: RecipeModel):
        print(f"[DEBUG] PostgresRecipeRepository.update: Updating recipe with id={getattr(recipe, 'id', None)}, approved={getattr(recipe, 'approved', None)}")
        db_recipe = self.db.query(DBRecipe).filter(DBRecipe.recipe_id == recipe.id).first()
        print(f"[DEBUG] PostgresRecipeRepository.update: DBRecipe before update: {db_recipe}")
        if not db_recipe:
            print(f"[DEBUG] PostgresRecipeRepository.update: Recipe {recipe.id} not found in DB!")
            raise ValueError(f"Recipe {recipe.id} not found")

        db_recipe.dish_name = recipe.title
        # Recipe.servings is immutable - do not update
        # Only set is_published to True if recipe.approved is True
        if recipe.approved:
            db_recipe.is_published = True
        print(f"[DEBUG] PostgresRecipeRepository.update: DBRecipe after field update: recipe_id={db_recipe.recipe_id}, is_published={db_recipe.is_published}, created_by={db_recipe.created_by}")

        # Persist ratings using Feedback table
        from Module.database import Feedback
        # Only update if ratings are present in the RecipeModel
        if hasattr(recipe, 'ratings') and recipe.ratings:
            for rating_obj in recipe.ratings:
                user_id = getattr(rating_obj, 'user_id', None)
                rating_value = getattr(rating_obj, 'rating', None)
                if user_id and rating_value is not None:
                    # Ensure rating is within 0-5
                    rating_value = min(max(rating_value, 0), 5)
                    # Use latest version for feedback
                    latest_version = db_recipe.versions[-1] if db_recipe.versions else None
                    if latest_version:
                        feedback = self.db.query(Feedback).filter(Feedback.version_id == latest_version.version_id, Feedback.user_id == user_id).first()
                        if feedback:
                            feedback.rating = rating_value
                        else:
                            feedback = Feedback(
                                feedback_id=str(uuid.uuid4()),
                                version_id=latest_version.version_id,
                                user_id=user_id,
                                created_at=get_india_time(),
                                rating=rating_value,
                                comment=None
                            )
                            self.db.add(feedback)
        # Recalculate avg_rating logic removed (field deleted)

        self.db.commit()
        self.db.refresh(db_recipe)
        print(f"[DEBUG] PostgresRecipeRepository.update: DBRecipe after commit: recipe_id={db_recipe.recipe_id}, is_published={db_recipe.is_published}, created_by={db_recipe.created_by}")
    
    def delete(self, recipe_id: str):
        """Delete a recipe by ID."""
        db_recipe = self.db.query(DBRecipe).filter(DBRecipe.recipe_id == recipe_id).first()
        if db_recipe:
            self.db.delete(db_recipe)
            self.db.commit()
    
    def _to_model(self, db_recipe: DBRecipe) -> RecipeModel:
        """Convert database model to Recipe model."""
        print(f"[DEBUG] _to_model called with db_recipe: {db_recipe}")
        # Get the latest version (always use latest, never track mutable current_version_id)
        current_version = db_recipe.versions[-1] if db_recipe.versions else None
        if current_version:
            ingredients = [
                Ingredient(name=ing.name, quantity=ing.quantity, unit=ing.unit)
                for ing in current_version.ingredients
            ]
            steps = [s.instruction for s in sorted(current_version.steps, key=lambda x: x.step_order)]
            servings = current_version.base_servings if hasattr(current_version, 'base_servings') and current_version.base_servings else getattr(db_recipe, 'servings', 1)
        else:
            ingredients = []
            steps = []
            servings = getattr(db_recipe, 'servings', 1)
        # Ensure ingredients and steps are always lists
        if ingredients is None:
            ingredients = []
        if steps is None:
            steps = []
        print(f"[DEBUG] _to_model ingredients: {ingredients}")
        print(f"[DEBUG] _to_model steps: {steps}")
        if servings is None:
            servings = 1
        # Fetch ratings from Feedback table and ensure 0-5 limit
        from Module.database import Feedback
        latest_version = db_recipe.versions[-1] if db_recipe.versions else None
        feedbacks = self.db.query(Feedback).filter(Feedback.version_id == (latest_version.version_id if latest_version else None), Feedback.rating != None).all()
        safe_ratings = [min(max(f.rating, 0), 5) for f in feedbacks]
        avg_rating = round(sum(safe_ratings) / len(safe_ratings), 2) if safe_ratings else 0.0
        model = RecipeModel(
            id=getattr(db_recipe, 'recipe_id', None),
            title=db_recipe.dish_name,
            ingredients=ingredients,
            steps=steps,
            servings=servings,
            metadata={},
            ratings=safe_ratings,
            ai_confidence_score=0.0,
            popularity=0,
            approved=db_recipe.is_published
        )
        print(f"[DEBUG] _to_model: db_recipe.recipe_id={getattr(db_recipe, 'recipe_id', None)}, model.id={getattr(model, 'id', None)}, model={model}")
        return model
