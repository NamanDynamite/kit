
# KitchenMind - Modular AI Recipe Platform

KitchenMind is a scalable, community-driven AI recipe platform supporting 100k+ members and millions of recipes, with a FastAPI REST API and PostgreSQL backend.


## Quick Start

1. **Install Python 3.8+ and PostgreSQL 12+**
2. **Clone this repo and create a virtual environment**
3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Configure your database in `.env`**
5. **Initialize the database:**
    ```bash
    python setup_db.py
    ```
6. **Run the API server:**
    ```bash
    run_api.bat  # Windows
    # or
    ./run_api.sh # Linux/Mac
    ```
7. **Access the API docs:**
    - Swagger UI: http://localhost:8000/docs
    - ReDoc: http://localhost:8000/redoc

---

## Project Structure

```
Kitchen Mind/
├── Module/                    # Main package directory
│   ├── __init__.py           # Package initialization and exports
│   ├── models.py             # Data models (Ingredient, Recipe, User)
│   ├── repository.py         # Recipe storage and retrieval
│   ├── vector_store.py       # Semantic search functionality
│   ├── scoring.py            # Recipe ranking and scoring
│   ├── synthesizer.py        # Recipe synthesis and merging
│   ├── token_economy.py      # RMDT token rewards system
│   ├── event_planner.py      # Event planning functionality
│   └── controller.py         # Main KitchenMind controller
├── main.py                   # Entry point with example usage
├── one_file.py               # Original single-file version (preserved)
└── README.md                 # This file
```


## API Overview

KitchenMind exposes a REST API for user management, recipe submission, validation, synthesis, and event planning. See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for full details.

**Key Endpoints:**
- `POST /user` - Create user
- `POST /recipes` - Submit recipe
- `POST /recipes/{id}/validate` - Validate recipe
- `POST /recipes/synthesize` - Synthesize recipes
- `POST /event/plan` - Plan event

---

## Module Descriptions

### `models.py`
Core data classes:
- **Ingredient**: Represents a recipe ingredient with name, quantity, and unit
- **Recipe**: Complete recipe with ingredients, steps, metadata, and ratings
- **User**: User account with role-based permissions and RMDT balance

### `repository.py`
- **RecipeRepository**: In-memory storage for recipes
- Methods for adding, retrieving, and filtering recipes

### `vector_store.py`
- **MockVectorStore**: Simple semantic search implementation
- Uses pseudo-random vectors for similarity matching
- In production, replace with actual embeddings + vector DB

### `scoring.py`
- **ScoringEngine**: Multi-factor recipe ranking system
- Considers user ratings, validator confidence, authenticity, scalability, and popularity

### `synthesizer.py`
- **Synthesizer**: Combines multiple recipes into one optimized recipe
- Merges ingredients intelligently
- Normalizes cooking steps and phases
- Handles ingredient conflicts (e.g., leavening agents)

### `token_economy.py`
- **TokenEconomy**: Manages RMDT token rewards
- Rewards trainers for recipe submissions
- Rewards validators for recipe validation

### `event_planner.py`
- **EventPlanner**: Creates event menus from approved recipes
- Filters by dietary requirements
- Estimates costs and guest counts

### `controller.py`
- **KitchenMind**: Main orchestrator class
- Integrates all modules
- Provides high-level API for:
  - User management
  - Recipe submission and validation
  - Recipe requests and synthesis
  - Rating and feedback
  - Event planning


## Usage (Python API)

### Basic Example

```python
from Module import KitchenMind

# Initialize system
km = KitchenMind()

# Create users
trainer = km.create_user('alice', role='trainer')
validator = km.create_user('bob', role='validator')
user = km.create_user('charlie', role='user')

# Submit a recipe
recipe = km.submit_recipe(
    trainer,
    title='Idli – Traditional',
    ingredients=[
        {'name': 'Rice', 'quantity': 300, 'unit': 'g'},
        {'name': 'Urad Dal', 'quantity': 100, 'unit': 'g'},
        {'name': 'Water', 'quantity': 350, 'unit': 'ml'},
        {'name': 'Salt', 'quantity': 5, 'unit': 'g'},
    ],
    steps=[
        'Soak rice and urad dal separately for 4 hours.',
        'Grind both into a smooth batter.',
        'Let the batter ferment overnight.',
        'Add salt and steam for 12 minutes.'
    ],
    servings=4
)

# Validate recipe
km.validate_recipe(validator, recipe.id, approved=True, confidence=0.85)

# Request synthesized recipe
result = km.request_recipe(user, 'Idli', servings=5)
```


### Running the Example (Python)

```bash
python main.py
```

This will run a demonstration of the core logic (user creation, recipe submission, validation, synthesis, event planning, token rewards).

For REST API usage, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md) and use the interactive docs at http://localhost:8000/docs.


## Key Features

- Role-based user system (user, trainer, validator, admin)
- Recipe submission and validation workflow
- Multi-source recipe synthesis
- Semantic search for recipes
- Weighted scoring system
- Token-based reward economy
- Event planning with menu suggestions
- Ingredient normalization and conflict resolution

## Benefits of Modular Structure

1. **Maintainability**: Each module has a single, clear responsibility
2. **Testability**: Modules can be tested independently
3. **Reusability**: Components can be imported and used separately
4. **Scalability**: Easy to extend individual modules without affecting others
5. **Readability**: Smaller files are easier to understand and navigate
6. **Collaboration**: Multiple developers can work on different modules simultaneously


## Migration Notes

- The original `one_file.py` is preserved for reference
- All imports use relative imports (`.models`, `.repository`, etc.)
- The package can be imported as: `from Module import KitchenMind`
- Debug logging is preserved
- The synthesizer maintains API compatibility


## Future Enhancements

- Replace MockVectorStore with production vector DB
- Add more persistent storage options
- Expand LLM integration
- Add web interface
- Add more unit tests
- Implement caching and advanced logging


## License

MIT License - See LICENSE file


## Contributing

Contributions are welcome! Please:
- Follow code style
- Keep modules single-responsibility
- Add tests for new features
- Update documentation as needed
