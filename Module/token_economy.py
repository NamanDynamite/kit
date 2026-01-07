"""
Token economy for rewarding contributors.
Implements RMDT token rewards for trainers and users.
"""

from typing import Dict
from .models import User


class TokenEconomy:
    """Manages RMDT token rewards and ledger"""
    
    def __init__(self):
        self.ledger: Dict[str, float] = {}

    def reward_trainer_submission(self, trainer: User, amount: float = 1.0):
        """Reward a trainer for submitting a recipe"""
        print(f"[DEBUG] reward_trainer_submission: trainer={trainer}, type={type(trainer)}")
        print(f"[DEBUG] reward_trainer_submission: hasattr(trainer, 'user_id')={hasattr(trainer, 'user_id')}, hasattr(trainer, 'id')={hasattr(trainer, 'id')}")
        print(f"[DEBUG] reward_trainer_submission: trainer.user_id={getattr(trainer, 'user_id', None)}, trainer.id={getattr(trainer, 'id', None)}")
        trainer.credit += amount
        key = getattr(trainer, 'user_id', trainer.id)
        print(f"[DEBUG] reward_trainer_submission: ledger key={key}")
        self.ledger.setdefault(key, 0.0)
        self.ledger[key] += amount

    # Validator role removed. No reward_validator method needed.

    def reward_user_request(self, user: User, amount: float = 0.25):
        """Reward a user for requesting a recipe synthesis.
        
        Args:
            user (User): User requesting the recipe
            amount (float): Token amount to reward
        """
        if not user:
            raise ValueError("User cannot be None")
        if amount <= 0:
            raise ValueError("Reward amount must be positive")
        
        print(f"[DEBUG] reward_user_request: user={user}, type={type(user)}")
        print(f"[DEBUG] reward_user_request: hasattr(user, 'user_id')={hasattr(user, 'user_id')}, hasattr(user, 'id')={hasattr(user, 'id')}")
        print(f"[DEBUG] reward_user_request: user.user_id={getattr(user, 'user_id', None)}")
        user.credit += amount
        key = getattr(user, 'user_id', None)
        print(f"[DEBUG] reward_user_request: ledger key={key}")
        self.ledger.setdefault(key, 0.0)
        self.ledger[key] += amount

    def reward_user_rating(self, user: User, amount: float = 0.1):
        """Reward a user for rating a recipe.
        
        Args:
            user (User): User rating the recipe
            amount (float): Token amount to reward
        """
        print(f"[DEBUG] reward_user_rating: user={user}, type={type(user)}")
        print(f"[DEBUG] reward_user_rating: hasattr(user, 'user_id')={hasattr(user, 'user_id')}, hasattr(user, 'id')={hasattr(user, 'id')}")
        print(f"[DEBUG] reward_user_rating: user.user_id={getattr(user, 'user_id', None)}")
        if not user:
            raise ValueError("User cannot be None")
        if amount <= 0:
            raise ValueError("Reward amount must be positive")
        
        user.credit += amount
        key = getattr(user, 'user_id', None)
        print(f"[DEBUG] reward_user_rating: ledger key={key}")
        self.ledger.setdefault(key, 0.0)
        self.ledger[key] += amount
