
# KitchenMind API Documentation

## Overview

KitchenMind API is a FastAPI-based REST API for recipe management, synthesis, and event planning with a PostgreSQL backend. It supports user management, recipe submission/validation, synthesis, event planning, and more.


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

7. **Access the API**
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc


---

## API Endpoints


### Health Check

- `GET /` — API info
- `GET /health` — Health check


### User Management

- `POST /user` — Create user
- `GET /user/{user_id}` — Get user details

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

- `POST /recipes` — Submit recipe (trainer only)
- `GET /recipes` — List recipes
- `GET /recipes/{recipe_id}` — Get recipe details
- `GET /recipes/pending` — List pending recipes

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


### Recipe Operations

- `POST /recipes/{recipe_id}/validate` — Validate recipe (validator only)
- `POST /recipes/{recipe_id}/rate` — Rate recipe

---


### Recipe Synthesis

- `POST /recipes/synthesize` — Synthesize recipes

---


### Event Planning

- `POST /event/plan` — Plan event with recipes

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

See [FASTAPI_SETUP.md](FASTAPI_SETUP.md) for schema summary. Main tables:
- recipes
- ingredients
- steps
- users


## Usage Examples

See [FASTAPI_SETUP.md](FASTAPI_SETUP.md) for curl and Python usage examples.

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
