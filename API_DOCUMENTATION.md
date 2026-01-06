# KitchenMind API Documentation

## Overview

KitchenMind API is a FastAPI-based REST API for recipe management, synthesis, and event planning with PostgreSQL backend.

## Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL 12+
- pip (Python package manager)

### Installation

1. **Clone or setup the project**
```bash
cd "Kitchen Mind"
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup PostgreSQL database**
```bash
python setup_db.py
```

5. **Configure environment variables**
Edit `.env` file with your PostgreSQL credentials:
```env
DATABASE_URL=postgresql://kitchenmind:password@localhost:5432/kitchenmind
```

6. **Run the API server**
```bash
# Windows
run_api.bat

# Linux/Mac
chmod +x run_api.sh
./run_api.sh

# Or directly
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

7. **Access the API**
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health Check

#### GET `/`
Health check and API info
```json
{
  "message": "KitchenMind API",
  "version": "1.0.0",
  "status": "online",
  "docs": "/docs"
}
```

#### GET `/health`
Detailed health check
```json
{
  "status": "healthy",
  "database": "connected",
  "api": "running"
}
```

---

### User Management


#### POST `/user`
Create a new user
```json
{
  "name": "Alice Trainer",
  "email": "alice@example.com",
  "login_identifier": "alice_trainer",
  "password_hash": "hashed_password",
  "auth_type": "local",
  "role_id": "trainer",
  "dietary_preference": "vegetarian"
}
```

Response:
```json
{
  "user_id": "uuid-string",
  "name": "Alice Trainer",
  "email": "alice@example.com",
  "login_identifier": "alice_trainer",
  "role_id": "trainer",
  "dietary_preference": "vegetarian",
  "rating_score": 0.0,
  "total_points": 0,
  "created_at": "2026-01-02T12:00:00Z",
  "last_login_at": "2026-01-02T12:00:00Z"
}
```

#### GET `/user/{user_id}`
Get user details
```
GET /user/uuid-string
```

Response:
```json
{
  "user_id": "uuid-string",
  "name": "Alice Trainer",
  "email": "alice@example.com",
  "login_identifier": "alice_trainer",
  "role_id": "trainer",
  "dietary_preference": "vegetarian",
  "rating_score": 1.5,
  "total_points": 10,
  "created_at": "2026-01-02T12:00:00Z",
  "last_login_at": "2026-01-02T12:00:00Z"
}
```

---

### Recipe Management

#### POST `/recipes`
Submit a new recipe (trainer only)

Query parameter: `trainer_id`

Request:
```json
{
  "title": "Idli - South Indian Rice Cakes",
  "servings": 4,
  "ingredients": [
    {
      "name": "Rice",
      "quantity": 300,
      "unit": "g"
    },
    {
      "name": "Urad Dal",
      "quantity": 100,
      "unit": "g"
    }
  ],
  "steps": [
    "Soak rice and urad dal for 4 hours",
    "Grind into smooth batter",
    "Ferment overnight",
    "Steam for 12 minutes"
  ]
}
```

Response:
```json
{
  "id": "recipe-id",
  "title": "Idli - South Indian Rice Cakes",
  "servings": 4,
  "approved": false,
  "popularity": 0,
  "avg_rating": 0.0
}
```

#### GET `/recipes`
List recipes

Query parameters:
- `approved_only` (bool): Filter approved recipes (default: true)

Response:
```json
[
  {
    "id": "recipe-id",
    "title": "Idli",
    "servings": 4,
    "approved": true,
    "popularity": 5,
    "avg_rating": 4.5
  }
]
```

#### GET `/recipes/{recipe_id}`
Get specific recipe details

Response:
```json
{
  "id": "recipe-id",
  "title": "Idli",
  "servings": 4,
  "approved": true,
  "popularity": 5,
  "avg_rating": 4.5
}
```

#### GET `/recipes/pending`
List pending (unapproved) recipes

Response:
```json
[
  {
    "id": "recipe-id",
    "title": "New Recipe",
    "servings": 3,
    "submitted_by": "trainer_username"
  }
]
```

---

### Recipe Validation

#### POST `/recipes/{recipe_id}/validate`
Validate a recipe (validator only)

Query parameters:
- `validator_id`: UUID of the validator

Request:
```json
{
  "approved": true,
  "feedback": "Well structured recipe",
  "confidence": 0.85
}
```

Response:
```json
{
  "message": "Recipe validated successfully",
  "recipe_id": "recipe-id",
  "approved": true
}
```

#### POST `/recipes/{recipe_id}/rate`
Rate a recipe

Query parameters:
- `user_id`: UUID of the user
- `rating`: Float 0-5

Response:
```json
{
  "message": "Recipe rated successfully",
  "recipe_id": "recipe-id",
  "rating": 4.5,
  "avg_rating": 4.5
}
```

---

### Recipe Synthesis

#### POST `/recipes/synthesize`
Synthesize multiple recipes into one

Query parameter: `user_id`

Request:
```json
{
  "dish_name": "Idli",
  "servings": 5,
  "top_k": 10,
  "reorder": true
}
```

Response:
```json
{
  "id": "synthesized-recipe-id",
  "title": "Synthesized - Idli (for 5 servings)",
  "servings": 5,
  "steps": [
    "Soak rice and urad dal...",
    "Grind into batter...",
    "Ferment overnight...",
    "Steam for 12 minutes"
  ],
  "ingredients": [
    {
      "name": "Rice",
      "quantity": 375.0,
      "unit": "g"
    }
  ],
  "metadata": {
    "sources": ["recipe-id-1", "recipe-id-2"],
    "ai_confidence": 0.85,
    "synthesis_method": "llm:google/flan-t5-base"
  }
}
```

---

### Event Planning


#### POST `/event/plan`
Plan an event with recipes

Request:
```json
{
  "event_name": "Birthday Party",
  "guest_count": 20,
  "budget_per_person": 5.0,
  "dietary": "vegetarian"
}
```

Response:
```json
{
  "event": "Birthday Party",
  "guests": 20,
  "budget": 100.0,
  "menu": [
    {
      "title": "Idli",
      "serves": 4
    },
    {
      "title": "Dosa",
      "serves": 6
    }
  ],
  "notes": "This is a sample plan..."
}
```

---

## Authentication

Currently, the API uses simple query parameters for user identification. For production, implement:
- JWT tokens
- API keys
- OAuth2

## Error Handling

All errors return appropriate HTTP status codes:

- `400 Bad Request`: Invalid input
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

Example error response:
```json
{
  "detail": "User not found"
}
```

## Database Schema

### Tables

**recipes**
- id (String, PK)
- title (String)
- servings (Integer)
- metadata (JSON)
- ratings (JSON array)
- validator_confidence (Float)
- popularity (Integer)
- approved (Boolean)

**ingredients**
- id (Integer, PK)
- recipe_id (String, FK)
- name (String)
- quantity (Float)
- unit (String)

**steps**
- id (Integer, PK)
- recipe_id (String, FK)
- order (Integer)
- text (String)

**users**
- id (String, PK)
- username (String, unique)
- role (String)
- rmdt_balance (Float)

## Usage Examples


### Create User
```bash
curl -X POST "http://localhost:8000/user" \
  -H "Content-Type: application/json" \
  -d '{"name": "Alice Trainer", "email": "alice@example.com", "login_identifier": "alice_trainer", "password_hash": "hashed_password", "auth_type": "local", "role_id": "trainer", "dietary_preference": "vegetarian"}'
```

### Submit Recipe
```bash
curl -X POST "http://localhost:8000/recipes?trainer_id=USER_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Idli",
    "servings": 4,
    "ingredients": [{"name": "Rice", "quantity": 300, "unit": "g"}],
    "steps": ["Step 1", "Step 2"]
  }'
```

### Synthesize Recipe
```bash
curl -X POST "http://localhost:8000/recipes/synthesize?user_id=USER_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "dish_name": "Idli",
    "servings": 5,
    "reorder": true
  }'
```

## Performance Tips

1. Use connection pooling for database
2. Cache frequently accessed recipes
3. Use pagination for large result sets
4. Index frequently queried fields

## Troubleshooting

### Database Connection Error
- Verify PostgreSQL is running
- Check DATABASE_URL in .env
- Ensure database user has proper permissions

### Module Not Found
- Activate virtual environment
- Run `pip install -r requirements.txt`

### Port Already in Use
- Change port in run_api.bat/.sh or .env
- Or: `lsof -i :8000` (Linux/Mac) to find process

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
```

### Type Checking
```bash
mypy .
```

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

## License

MIT License - See LICENSE file

## Support

For issues and questions:
- Check API documentation: http://localhost:8000/docs
- Review code examples in this file
- Check GitHub issues
