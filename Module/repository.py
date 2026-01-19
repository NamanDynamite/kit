"""
Repository interface for KitchenMind.
"""

from abc import ABC, abstractmethod
from typing import List, Any

class RecipeRepository(ABC):
    @abstractmethod
    def list(self) -> List[Any]:
        pass
    # Add other abstract methods as needed for your app
