"""
Main controller for the KitchenMind system.
Orchestrates all components: recipes, users, synthesis, validation, etc.
"""

import uuid
from typing import Dict, List, Optional
from .models import User, Recipe, Ingredient
from .repository_postgres import PostgresRecipeRepository
from .database import SessionLocal
from .vector_store import MockVectorStore
from .scoring import ScoringEngine
from .synthesizer import Synthesizer
from .token_economy import TokenEconomy
from .event_planner import EventPlanner


class KitchenMind:
    def __init__(self, recipe_repo=None, db_session=None):
        # Always use PostgresRecipeRepository unless another is provided
        if db_session is None:
            db_session = SessionLocal()
        self.db_session = db_session
        self.recipes = recipe_repo if recipe_repo is not None else PostgresRecipeRepository(db_session)
        self.vstore = MockVectorStore()
        self.scorer = ScoringEngine()
        self.synth = Synthesizer()
        self.tokens = TokenEconomy()
        self.users: Dict[str, User] = {}

    def create_user(self, username: str, role: str = 'user') -> User:
        """Create a new user with specified role (user, trainer, admin)."""
        if role not in ['user', 'trainer', 'admin']:
            raise ValueError(f"Invalid role: {role}. Must be one of: user, trainer, admin")
        user = User(id=str(uuid.uuid4()), username=username, role=role)
        self.users[getattr(user, 'user_id', user.id)] = user
        return user

    def submit_recipe(self, trainer: User, title: str, ingredients: List[Dict], steps: List[str], servings: int) -> Recipe:
        """Submit a recipe for validation (trainer/admin only)."""
        print(f"[DEBUG] submit_recipe called with trainer={trainer}, title={title}, ingredients={ingredients}, steps={steps}, servings={servings}")
        if trainer.role not in ('trainer', 'admin'):
            raise PermissionError('Only trainers or admins can submit recipes.')
        if not title or title.strip() == "":
            raise ValueError("Recipe title cannot be empty")
        if not ingredients or len(ingredients) < 2:
            raise ValueError("Recipe must have at least 2 ingredients")
        if not steps or len(steps) < 1:
            raise ValueError("Recipe must have at least 1 step")
        if servings <= 0:
            raise ValueError("Servings must be a positive number")
        recipe = Recipe(
            id=str(uuid.uuid4()),
            title=title,
            ingredients=[Ingredient(**ing) for ing in ingredients],
            steps=steps,
            servings=servings,
            metadata={'submitted_by': trainer.username, 'submitted_by_id': trainer.id}
        )
        print(f"[DEBUG] Created recipe object: {recipe}")
        print(f"[DEBUG] Recipe type: {type(recipe)}")
        self.recipes.add(recipe)
        print(f"[DEBUG] Added recipe to self.recipes")
        self.vstore.index(recipe)
        print(f"[DEBUG] Indexed recipe in vstore")
        self.tokens.reward_trainer_submission(trainer, amount=1.0)
        print(f"[DEBUG] Rewarded trainer for submission")
        return recipe

    def validate_recipe(self, admin: User, recipe_id: str, approved: bool, feedback: Optional[str] = None, confidence: float = 0.8):
        """Validate a recipe with confidence scoring and auto-approval at 90%+ confidence. Only admins can validate."""
        print(f"[DEBUG] validate_recipe called with admin={admin}, recipe_id={recipe_id}, approved={approved}, feedback={feedback}, confidence={confidence}")
        if admin.role != 'admin':
            raise PermissionError('Only admins can validate recipes.')
        r = self.recipes.get(recipe_id)
        print(f"[DEBUG] Retrieved recipe: {r}")
        print(f"[DEBUG] Recipe type: {type(r)}")
        if r is None:
            raise KeyError(f'Recipe "{recipe_id}" not found')
        # Ensure r is a dataclass Recipe (not ORM)
        from .models import Recipe as RecipeModel
        if not isinstance(r, RecipeModel):
            # Convert ORM or dict to dataclass Recipe
            r = RecipeModel(
                id=getattr(r, 'id', getattr(r, 'recipe_id', None)),
                title=getattr(r, 'title', getattr(r, 'dish_name', '')),
                ingredients=getattr(r, 'ingredients', []),
                steps=getattr(r, 'steps', []),
                servings=getattr(r, 'servings', 1),
                metadata=getattr(r, 'metadata', {}),
                ratings=getattr(r, 'ratings', []),
                validator_confidence=getattr(r, 'validator_confidence', 0.0),
                popularity=getattr(r, 'popularity', 0),
                approved=getattr(r, 'approved', False),
                rejection_suggestions=getattr(r, 'rejection_suggestions', [])
            )
            print(f"[DEBUG] Converted recipe to dataclass: {r}")
        # Normalize leavening ingredients
        print(f"[DEBUG] Normalizing leavening ingredients for: {getattr(r, 'ingredients', None)}")
        r.ingredients = self.synth.normalize_leavening(r.ingredients)
        # Validate and normalize confidence score (0.0 to 1.0)
        r.validator_confidence = max(0.0, min(1.0, confidence))
        print(f"[DEBUG] Set validator_confidence: {r.validator_confidence}")
        # Add validator metadata
        r.metadata['validated_by'] = admin.username
        r.metadata['validated_by_id'] = admin.id
        r.metadata['confidence_score'] = r.validator_confidence
        # Auto-approve if confidence >= 0.9
        if r.validator_confidence >= 0.9:
            r.approved = True
            r.popularity += 1
            self.vstore.index(r)
            r.metadata['validation_feedback'] = feedback or 'Auto-approved with high confidence (≥90%)'
            r.metadata['auto_approved'] = True
            print(f"✓ Recipe '{r.title}' AUTO-APPROVED (confidence: {r.validator_confidence:.1%})")
        elif approved:
            # Manual approval (confidence < 90%)
            r.approved = True
            r.popularity += 1
            self.vstore.index(r)
            r.metadata['validation_feedback'] = feedback or 'Manually approved'
            r.metadata['auto_approved'] = False
            print(f"✓ Recipe '{r.title}' MANUALLY APPROVED (confidence: {r.validator_confidence:.1%})")
        else:
            # Rejected - generate AI suggestions for trainer
            r.approved = False
            r.metadata['validation_feedback'] = feedback
            r.metadata['auto_approved'] = False
            r.rejection_suggestions = self._generate_ai_suggestions(r, feedback, confidence)
            r.metadata['rejected'] = True
            r.metadata['rejection_reason'] = feedback or "Does not meet quality standards"
            print(f"✗ Recipe '{r.title}' REJECTED (confidence: {r.validator_confidence:.1%})")
            print(f"  Suggestions sent to trainer for improvement")
        # Reward admin for validation
        self.tokens.reward_validator(admin, amount=0.5)
        return r
    
    def _generate_ai_suggestions(self, recipe: Recipe, feedback: Optional[str], confidence: float) -> List[str]:
        """Generate AI suggestions for rejected recipes to help trainers improve them."""
        suggestions = []
        
        # 1. Confidence Analysis
        if confidence < 0.5:
            suggestions.append(f"🔴 CRITICAL: Very low confidence ({confidence:.1%}) - Recipe needs major comprehensive review")
        elif confidence < 0.7:
            suggestions.append(f"🟡 MODERATE: Confidence at {confidence:.1%} - Recipe needs significant refinement")
        else:
            suggestions.append(f"🟢 GOOD: Confidence at {confidence:.1%} - Minor adjustments needed for approval")
        
        # 2. Ingredient Analysis
        if len(recipe.ingredients) < 3:
            suggestions.append("❌ Incomplete ingredients: Add at least 3 ingredients (currently {})".format(len(recipe.ingredients)))
        elif len(recipe.ingredients) > 20:
            suggestions.append("⚠️  Too many ingredients ({}) - Consider simplifying the recipe".format(len(recipe.ingredients)))
        else:
            suggestions.append(f"✓ Ingredient count ({len(recipe.ingredients)}) is appropriate")
        
        # 3. Unit and Quantity Validation
        missing_units = []
        invalid_quantities = []
        for ing in recipe.ingredients:
            if not ing.unit or ing.unit.strip() == "":
                missing_units.append(ing.name)
            if ing.quantity is None or ing.quantity <= 0:
                invalid_quantities.append(f"{ing.name} (quantity: {ing.quantity})")
        
        if missing_units:
            suggestions.append(f"❌ Missing units: Specify measurement units for: {', '.join(missing_units)}")
        if invalid_quantities:
            suggestions.append(f"❌ Invalid quantities: Fix quantities for: {', '.join(invalid_quantities)}")
        
        # 4. Cooking Steps Analysis
        if len(recipe.steps) < 2:
            suggestions.append(f"❌ Insufficient steps ({len(recipe.steps)}): Add at least 2-3 detailed cooking steps")
        elif len(recipe.steps) > 50:
            suggestions.append(f"⚠️  Too many steps ({len(recipe.steps)}) - Consider consolidating similar steps")
        else:
            suggestions.append(f"✓ Step count ({len(recipe.steps)}) is reasonable")
        
        # 5. Step Quality Analysis
        short_steps = [s for s in recipe.steps if len(s.strip()) < 15]
        if short_steps:
            suggestions.append(f"⚠️  IMPROVE: {len(short_steps)} step(s) are too brief - Add more cooking details and timing")
        
        # 6. Servings Validation
        if recipe.servings <= 0 or recipe.servings > 100:
            suggestions.append(f"⚠️  ADJUST: Servings value ({recipe.servings}) seems unusual - Typically 1-50")
        
        # 7. Include Validator Feedback
        if feedback and feedback.strip():
            suggestions.append(f"\n📋 Validator Comment: {feedback}")
        
        # 8. General Improvement Tips
        if confidence < 0.9:
            suggestions.append("\n💡 GENERAL IMPROVEMENTS:")
            suggestions.append("  • Review ingredient proportions - ensure they're realistic for the servings")
            suggestions.append("  • Add cooking time and temperature estimates where applicable")
            suggestions.append("  • Ensure steps are in logical cooking order")
            suggestions.append("  • Use clear, specific cooking terms (e.g., 'medium heat', 'until golden brown')")
            suggestions.append("  • Consider adding prep time and difficulty level")
        
        # 9. Resubmission Instructions
        suggestions.append("\n📝 NEXT STEPS: Address the feedback above and resubmit the recipe for re-validation")
        
        return suggestions

    def request_recipe(self, user: User, dish_name: str, servings: int = 2, top_k: int = 10, reorder: bool = True) -> Recipe:
        """Request a synthesized recipe for a specific dish and serving size."""
        if not user:
            raise ValueError("User cannot be None")
        
        if servings <= 0:
            raise ValueError("Servings must be positive")
        
        # Try direct title match first (preferred)
        direct = [r for r in self.recipes.find_by_title(dish_name) if hasattr(r, 'approved') and r.approved]
        candidates = []

        if direct:
            candidates = direct
        else:
            # Fallback to semantic search
            search_text = f"{dish_name} for {servings} servings"
            results = self.vstore.query(search_text, top_k=top_k)
            candidate_ids = [rid for rid, _ in results]
            candidates = [
                self.recipes.get(rid)
                for rid in candidate_ids
                if self.recipes.get(rid) and hasattr(self.recipes.get(rid), 'approved') and self.recipes.get(rid).approved
            ]

        if not candidates:
            raise LookupError(f'No approved recipes found for "{dish_name}"')

        # Prefer recipes with dish name in title
        named = [r for r in candidates if hasattr(r, 'title') and dish_name.lower() in r.title.lower()]
        if named:
            candidates = named

        # Ensure all candidates are Recipe dataclass instances
        from .models import Recipe as RecipeModel
        def ensure_recipe_dataclass(obj):
            if isinstance(obj, RecipeModel):
                return obj
            # Try to convert if possible
            ingredients = getattr(obj, 'ingredients', None)
            if ingredients is None:
                print(f"[DEBUG] WARNING: Recipe object {obj} missing 'ingredients' attribute. Setting to empty list.")
                ingredients = []
            elif not isinstance(ingredients, list):
                print(f"[DEBUG] WARNING: Recipe object {obj} has non-list 'ingredients'. Setting to empty list.")
                ingredients = []
            return RecipeModel(
                id=getattr(obj, 'id', getattr(obj, 'recipe_id', None)),
                title=getattr(obj, 'title', getattr(obj, 'dish_name', '')),
                ingredients=ingredients,
                steps=getattr(obj, 'steps', []),
                servings=getattr(obj, 'servings', 1),
                metadata=getattr(obj, 'metadata', {}),
                ratings=getattr(obj, 'ratings', []),
                validator_confidence=getattr(obj, 'validator_confidence', 0.0),
                popularity=getattr(obj, 'popularity', 0),
                approved=getattr(obj, 'approved', False),
                rejection_suggestions=getattr(obj, 'rejection_suggestions', [])
            )
        top_candidates = [ensure_recipe_dataclass(r) for r in candidates]
        # Debug: Check all top_candidates for valid ingredients
        for idx, r in enumerate(top_candidates):
            if not hasattr(r, 'ingredients'):
                print(f"[DEBUG] ERROR: Candidate recipe at index {idx} missing 'ingredients' attribute: {r}")
                raise AttributeError(f"Candidate recipe at index {idx} missing 'ingredients' attribute")
            if not isinstance(r.ingredients, list):
                print(f"[DEBUG] ERROR: Candidate recipe at index {idx} has non-list 'ingredients': {r.ingredients}")
                raise TypeError(f"Candidate recipe at index {idx} has non-list 'ingredients'")
            if not r.ingredients:
                print(f"[DEBUG] WARNING: Candidate recipe at index {idx} has empty 'ingredients' list: {r}")

        # Score and synthesize
        scored = [(r, self.scorer.score(r)) for r in top_candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        top_n = [r for r, _ in scored[:2]]

        synthesized = self.synth.synthesize(top_n, servings, reorder=reorder)
        self.recipes.add(synthesized)
        self.vstore.index(synthesized)

        # Reward user
        self.tokens.reward_user_request(user, amount=0.25)
        return synthesized

    def rate_recipe(self, user: User, recipe_id: str, rating: float) -> Recipe:
        """Rate a recipe (1.0 to 5.0 stars)."""
        if not user:
            raise ValueError("User cannot be None")
        
        if rating < 1.0 or rating > 5.0:
            raise ValueError("Rating must be between 1.0 and 5.0")
        
        r = self.recipes.get(recipe_id)
        if not r:
            raise KeyError(f'Recipe "{recipe_id}" not found')
        
        r.ratings.append(max(1.0, min(5.0, rating)))
        r.popularity += 1
        
        # Reward user for rating
        self.tokens.reward_user_rating(user, amount=0.1)
        return r

    def list_pending(self) -> List[Recipe]:
        """List all recipes pending validation."""
        return self.recipes.pending()
    
    def list_approved(self) -> List[Recipe]:
        """List all approved recipes."""
        return self.recipes.approved()
    
    def get_recipe(self, recipe_id: str) -> Optional[Recipe]:
        """Get a recipe by ID."""
        return self.recipes.get(recipe_id)
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get a user by ID."""
        return self.users.get(user_id)

    def event_plan(self, event_name: str, guest_count: int, budget_per_person: float, dietary: Optional[str] = None):
        """Plan an event menu."""
        if guest_count <= 0:
            raise ValueError("Guest count must be positive")
        if budget_per_person <= 0:
            raise ValueError("Budget per person must be positive")
        
        planner = EventPlanner(self.recipes)
        return planner.plan_event(event_name, guest_count, budget_per_person, dietary)
