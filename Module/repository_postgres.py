"""
PostgreSQL-based repository implementation for KitchenMind.
"""
from typing import List, Optional
from sqlalchemy.orm import Session
import uuid
from .database import Recipe as DBRecipe, Ingredient as DBIngredient, Step as DBStep
from Module.models import Recipe as RecipeModel, Ingredient


class PostgresRecipeRepository:
    def list(self) -> list:
        """Return all recipes in the database as RecipeModel objects."""
        db_recipes = self.db.query(self.model).all()
        print(f"[DEBUG] list() found {len(db_recipes)} recipes in DB")
        models = [self._to_model(r) for r in db_recipes]
        print(f"[DEBUG] list() returning models: {[m.id for m in models]}")
        return models

    def add_rating(self, recipe_id: str, user_id: str, rating: float):
        """Add or update a user's rating for a recipe in the Feedback table."""
        from Module.database import Feedback
        from datetime import datetime
        feedback = self.db.query(Feedback).filter(Feedback.recipe_id == recipe_id, Feedback.user_id == user_id).first()
        if feedback:
            feedback.rating = rating
            feedback.created_at = datetime.utcnow()
        else:
            feedback = Feedback(
                feedback_id=str(uuid.uuid4()),
                recipe_id=recipe_id,
                user_id=user_id,
                created_at=datetime.utcnow(),
                rating=rating,
                comment=None
            )
            self.db.add(feedback)
        self.db.commit()
        self.db.refresh(feedback)
        return feedback

    def get_ratings(self, recipe_id: str):
        """Get all ratings for a recipe from the Feedback table."""
        from Module.database import Feedback
        feedbacks = self.db.query(Feedback).filter(Feedback.recipe_id == recipe_id).all()
        return [fb.rating for fb in feedbacks if fb.rating is not None]

    def create_recipe(self, title, ingredients, steps, servings, submitted_by=None):
        """Create and persist a new recipe, returning the Recipe model with id."""
        print(f"[DEBUG] create_recipe called with title={title}, servings={servings}, submitted_by={submitted_by}")
        import datetime
        recipe_id = str(uuid.uuid4())
        version_id = str(uuid.uuid4())
        db_recipe = DBRecipe(
            recipe_id=recipe_id,
            dish_name=title,
            servings=servings if servings is not None else 1,
            created_by=submitted_by,
            is_published=False,
            created_at=datetime.datetime.utcnow(),
            current_version_id=version_id
        )
        print(f"[DEBUG] db_recipe created: {db_recipe}")
        db_version = self._create_version(version_id, recipe_id, submitted_by, servings, ingredients, steps)
        db_recipe.versions.append(db_version)
        self.db.add(db_recipe)
        print(f"[DEBUG] db_recipe and db_version added to session")
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
        import datetime
        db_version = RecipeVersion(
            version_id=version_id,
            recipe_id=recipe_id,
            submitted_by=submitted_by,
            submitted_at=datetime.datetime.utcnow(),
            status="submitted",
            validator_confidence=0.0,
            base_servings=servings,
            avg_rating=0.0
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
                unit=ing.unit,
                notes=None
            ) for ing in safe_ingredients
        ]
        db_version.steps = [
            DBStep(
                step_id=str(uuid.uuid4()),
                version_id=version_id,
                step_order=idx,
                instruction=step_text,
                minutes=None
            ) for idx, step_text in enumerate(steps)
        ]
        return db_version

    """Repository using PostgreSQL for persistent storage."""

    def __init__(self, db: Session):
        self.db = db
        self.model = DBRecipe  # Set self.model to the Recipe SQLAlchemy model
    
    def add(self, recipe: RecipeModel):
        """Add a new recipe to the database."""
        db_recipe = DBRecipe(
            recipe_id=recipe.id,
            dish_name=recipe.title,
            servings=recipe.servings if recipe.servings is not None else 1,
            created_by=None,  # Set appropriately if available
            is_published=recipe.approved,
            created_at=None  # Set appropriately if available
        )
        # Add ingredients
        for ing in recipe.ingredients:
            db_ing = DBIngredient(
                ingredient_id=str(uuid.uuid4()),
                version_id=None,  # Set appropriately if available
                name=ing.name,
                quantity=ing.quantity,
                unit=ing.unit,
                notes=None
            )
            db_recipe.ingredients.append(db_ing)
        # Add steps
        for idx, step_text in enumerate(recipe.steps):
            db_step = DBStep(
                step_id=str(uuid.uuid4()),
                version_id=None,  # Set appropriately if available
                step_order=idx,
                instruction=step_text,
                minutes=None
            )
            db_recipe.steps.append(db_step)
        self.db.add(db_recipe)
        self.db.commit()
        self.db.refresh(db_recipe)
    
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
        """Get all approved recipes."""
        db_recipes = self.db.query(DBRecipe).filter(DBRecipe.is_published == True).all()
        return [self._to_model(r) for r in db_recipes]
    
    def update(self, recipe: RecipeModel):
        """Update an existing recipe."""
        db_recipe = self.db.query(DBRecipe).filter(DBRecipe.recipe_id == recipe.id).first()
        if not db_recipe:
            raise ValueError(f"Recipe {recipe.id} not found")

        db_recipe.dish_name = recipe.title
        db_recipe.servings = recipe.servings
        db_recipe.is_published = recipe.approved

        # Persist ratings using Feedback table
        from Module.database import Feedback
        # Only update if ratings are present in the RecipeModel
        if hasattr(recipe, 'ratings') and recipe.ratings:
            for rating_obj in recipe.ratings:
                user_id = getattr(rating_obj, 'user_id', None)
                rating_value = getattr(rating_obj, 'rating', None)
                if user_id and rating_value is not None:
                    feedback = self.db.query(Feedback).filter(Feedback.recipe_id == recipe.id, Feedback.user_id == user_id).first()
                    if feedback:
                        feedback.rating = rating_value
                    else:
                        from datetime import datetime
                        feedback = Feedback(
                            feedback_id=str(uuid.uuid4()),
                            recipe_id=recipe.id,
                            user_id=user_id,
                            created_at=datetime.utcnow(),
                            rating=rating_value,
                            comment=None
                        )
                        self.db.add(feedback)
        # Recalculate avg_rating
        feedbacks = self.db.query(Feedback).filter(Feedback.recipe_id == recipe.id, Feedback.rating != None).all()
        if feedbacks:
            avg_rating = sum(f.rating for f in feedbacks) / len(feedbacks)
        else:
            avg_rating = 0.0
        # If RecipeVersion exists, update avg_rating
        from Module.database import RecipeVersion
        version = self.db.query(RecipeVersion).filter(RecipeVersion.recipe_id == recipe.id).first()
        if version:
            version.avg_rating = avg_rating

        self.db.commit()
        self.db.refresh(db_recipe)
    
    def delete(self, recipe_id: str):
        """Delete a recipe by ID."""
        db_recipe = self.db.query(DBRecipe).filter(DBRecipe.recipe_id == recipe_id).first()
        if db_recipe:
            self.db.delete(db_recipe)
            self.db.commit()
    
    def _to_model(self, db_recipe: DBRecipe) -> RecipeModel:
        """Convert database model to Recipe model."""
        print(f"[DEBUG] _to_model called with db_recipe: {db_recipe}")
        # Find the current version
        current_version = None
        if db_recipe.current_version_id:
            for v in db_recipe.versions:
                if v.version_id == db_recipe.current_version_id:
                    current_version = v
                    break
        if not current_version and db_recipe.versions:
            # fallback: use latest version
            current_version = db_recipe.versions[-1]
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
        print(f"[DEBUG] _to_model ingredients: {ingredients}")
        print(f"[DEBUG] _to_model steps: {steps}")
        if servings is None:
            servings = 1
        model = RecipeModel(
            id=getattr(db_recipe, 'recipe_id', None),
            title=db_recipe.dish_name,
            ingredients=ingredients,
            steps=steps,
            servings=servings,
            metadata={},
            ratings=[],
            validator_confidence=0.0,
            popularity=0,
            approved=db_recipe.is_published
        )
        print(f"[DEBUG] _to_model: db_recipe.recipe_id={getattr(db_recipe, 'recipe_id', None)}, model.id={getattr(model, 'id', None)}, model={model}")
        return model
