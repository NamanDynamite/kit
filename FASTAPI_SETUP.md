# KitchenMind - FastAPI + PostgreSQL Integration

Complete integration of KitchenMind recipe synthesis system with FastAPI REST API and PostgreSQL database.

## Quick Start

### 1. Prerequisites
- Python 3.8+
- PostgreSQL 12+

### 2. Setup Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Database
Edit `.env` file with your PostgreSQL credentials:
```env
DATABASE_URL=postgresql://kitchenmind:password@localhost:5432/kitchenmind
```

### 5. Initialize Database
```bash
python setup_db.py
```

### 6. Run API Server
```bash
run_api.bat  # Windows
# or
chmod +x run_api.sh && ./run_api.sh  # Linux/Mac
```

### 7. Access the API
- API Base URL: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs (Swagger UI)
- ReDoc: http://localhost:8000/redoc
- Health Check: http://localhost:8000/health

---

## Database Schema
See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for full schema and endpoint details.

---

## Main Endpoints

- `POST /user` - Create user
- `GET /user/{user_id}` - Get user details
- `POST /recipes` - Submit recipe (trainer only)
- `GET /recipes` - List recipes
- `GET /recipes/{recipe_id}` - Get recipe details
- `GET /recipes/pending` - List pending recipes
- `POST /recipes/{recipe_id}/validate` - Validate recipe (validator only)
- `POST /recipes/{recipe_id}/rate` - Rate recipe
- `POST /recipes/synthesize` - Synthesize recipes
- `POST /event/plan` - Plan event with recipes
- `GET /health` - Health check

See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for request/response examples and more endpoints.

---

## Testing
Run the test suite:
```bash
python test_api.py
```
This will test all major endpoints.

---

## Usage Examples
See [API_DOCUMENTATION.md](API_DOCUMENTATION.md) for curl and Python usage examples.
# KitchenMind - FastAPI + PostgreSQL Integration

Complete integration of KitchenMind recipe synthesis system with FastAPI REST API and PostgreSQL database.

## 📋 What's Included

### New Files Created

**API & Database:**
- `api.py` - FastAPI application with all endpoints
- `Module/database.py` - SQLAlchemy ORM models and database setup
- `Module/repository_postgres.py` - PostgreSQL repository implementation
- `setup_db.py` - Database initialization script
- `.env` - Environment configuration

**Scripts:**
- `run_api.bat` - Windows startup script
- `run_api.sh` - Linux/Mac startup script
- `test_api.py` - API testing suite

**Documentation:**
- `API_DOCUMENTATION.md` - Complete API reference
- `requirements.txt` - Python dependencies (updated)

## 🚀 Quick Start

### Step 1: Prerequisites

**Install PostgreSQL:**
- Windows: https://www.postgresql.org/download/windows/
- Mac: https://www.postgresql.org/download/macosx/
- Linux: `sudo apt-get install postgresql postgresql-contrib`

**Install Python 3.8+:**
- Download from https://www.python.org/downloads/

### Step 2: Setup Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Database

Edit `.env` file with your PostgreSQL credentials:
```env
DATABASE_URL=postgresql://kitchenmind:kitchenmind_password@localhost:5432/kitchenmind
```

### Step 5: Initialize Database

```bash
python setup_db.py
```

This will:
- Create PostgreSQL user `kitchenmind`
- Create database `kitchenmind`
- Create all necessary tables

### Step 6: Run API Server

**Windows:**
```bash
run_api.bat
```

**Linux/Mac:**
```bash
chmod +x run_api.sh
./run_api.sh
```

**Or manually:**
```bash
python -m uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

### Step 7: Access the API

- **API Base URL:** http://localhost:8000
- **Interactive Docs:** http://localhost:8000/docs (Swagger UI)
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

## 📊 Database Schema

### Tables

1. **recipes** - Recipe information
   - id (UUID, primary key)
   - title, servings, metadata
   - ratings, validator_confidence
   - popularity, approved

2. **ingredients** - Recipe ingredients
   - id (auto-increment)
   - recipe_id (foreign key)
   - name, quantity, unit

3. **steps** - Cooking instructions
   - id (auto-increment)
   - recipe_id (foreign key)
   - order (step number)
   - text (instruction)

4. **users** - User accounts
   - id (UUID, primary key)
   - username (unique)
   - role (user/trainer/validator/admin)
   - rmdt_balance (token balance)

## 🔌 API Endpoints

### User Management
- `POST /users` - Create user
- `GET /users/{user_id}` - Get user details

### Recipe Management
- `POST /recipes` - Submit recipe (trainer only)
- `GET /recipes` - List recipes
- `GET /recipes/{recipe_id}` - Get recipe details
- `GET /recipes/pending` - List pending recipes

### Recipe Operations
- `POST /recipes/{recipe_id}/validate` - Validate recipe (validator only)
- `POST /recipes/{recipe_id}/rate` - Rate recipe
- `POST /recipes/synthesize` - Synthesize recipes

### Event Planning
- `POST /events/plan` - Plan event with recipes

### Health
- `GET /` - API info
- `GET /health` - Health check

**Full API documentation:** See `API_DOCUMENTATION.md`

## 🧪 Testing

Run the test suite:

```bash
python test_api.py
```

This will test all major endpoints:
- Health check
- User creation
- Recipe submission
- Recipe validation
- Recipe synthesis
- Event planning

## 📝 Usage Examples

### Create a User
```bash
curl -X POST "http://localhost:8000/users" \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "role": "trainer"}'
```

### Submit a Recipe
```bash
curl -X POST "http://localhost:8000/recipes?trainer_id=YOUR_USER_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Idli",
    "servings": 4,
    "ingredients": [
      {"name": "Rice", "quantity": 300, "unit": "g"},
      {"name": "Urad Dal", "quantity": 100, "unit": "g"}
    ],
    "steps": ["Soak rice and dal", "Grind to batter", "Ferment", "Steam"]
  }'
```

### Synthesize Recipes
```bash
curl -X POST "http://localhost:8000/recipes/synthesize?user_id=YOUR_USER_ID" \
  -H "Content-Type: application/json" \
  -d '{
    "dish_name": "Idli",
    "servings": 5,
    "reorder": true
  }'
```

## 🔧 Configuration

### Environment Variables (.env)

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/db_name

# API
SECRET_KEY=your-secret-key
DEBUG=True
ENVIRONMENT=development

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4
```

### Database Connection

The API automatically:
1. Creates connection pool to PostgreSQL
2. Initializes all tables on startup
3. Provides session dependency injection
4. Handles connection cleanup

## 📚 Project Structure

```
Kitchen Mind/
├── api.py                          # FastAPI application
├── setup_db.py                     # Database setup script
├── test_api.py                     # API tests
├── run_api.bat                     # Windows startup
├── run_api.sh                      # Linux/Mac startup
├── .env                            # Environment config
├── requirements.txt                # Dependencies
├── API_DOCUMENTATION.md            # API docs
├── correct_all_5-12.py             # Original implementation
│
├── Module/                         # Database layer
│   ├── __init__.py
│   ├── database.py                 # SQLAlchemy models
│   ├── repository_postgres.py      # PostgreSQL repository
│   └── controller.py               # Main controller
│
└── kitchenmind_module/             # Modular system
    ├── models/                     # Data models
    ├── core/                       # Core components
    └── services/                   # Business logic
```

## 🔐 Security Notes

### For Production:

1. **Change default credentials** in `.env`
2. **Use strong SECRET_KEY**
3. **Enable HTTPS/SSL**
4. **Implement proper authentication** (JWT, OAuth2)
5. **Add rate limiting**
6. **Use environment-specific configs**
7. **Enable CORS restrictions**
8. **Add API key validation**
9. **Log and monitor requests**
10. **Regular database backups**

Example production .env:
```env
DATABASE_URL=postgresql://secure_user:VERY_STRONG_PASSWORD@db.example.com:5432/kitchenmind
SECRET_KEY=GENERATE_STRONG_SECRET_KEY_HERE
DEBUG=False
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000
WORKERS=8
```

## 🐛 Troubleshooting

### PostgreSQL Connection Error
```
psycopg2.OperationalError: could not connect to server
```
**Solution:** 
- Ensure PostgreSQL is running
- Verify DATABASE_URL in .env
- Check user credentials and database existence

### Port Already in Use
```
Address already in use
```
**Solution:**
- Change PORT in .env or run script
- Kill process: `lsof -i :8000` (Linux/Mac)

### Import Errors
```
ModuleNotFoundError: No module named 'fastapi'
```
**Solution:**
- Activate virtual environment
- Run `pip install -r requirements.txt`

### Database Table Errors
```
ProgrammingError: relation "recipes" does not exist
```
**Solution:**
- Run `python setup_db.py` to create tables

## 📈 Performance Optimization

1. **Connection Pooling:** SQLAlchemy manages connection pool
2. **Indexes:** Database has indexes on frequently queried fields
3. **Pagination:** Use limit/offset for large result sets
4. **Caching:** Implement Redis for hot recipes
5. **Batch Operations:** Group database operations

## 🔄 Workflow

### Basic Recipe Flow

1. **Trainer** submits recipes
2. **Validator** reviews and approves recipes
3. **Users** can rate approved recipes
4. **System** synthesizes multiple recipes into one
5. **Organizers** plan events using recipes

### Role Permissions

| Operation | User | Trainer | Validator | Admin |
|-----------|------|---------|-----------|-------|
| Submit Recipe | - | ✓ | - | ✓ |
| Validate Recipe | - | - | ✓ | ✓ |
| Rate Recipe | ✓ | ✓ | ✓ | ✓ |
| Synthesize | ✓ | ✓ | ✓ | ✓ |
| Plan Events | ✓ | ✓ | ✓ | ✓ |

## 📞 Support

For issues and questions:
1. Check `API_DOCUMENTATION.md`
2. Review example requests in this README
3. Check test_api.py for working examples
4. Review error messages in API response
5. Check FastAPI docs at `/docs`

## 🎯 Next Steps

1. ✅ Setup and run the API
2. ✅ Test endpoints using test_api.py
3. ✅ Explore interactive docs at /docs
4. ✅ Build frontend to consume the API
5. ✅ Deploy to production

## 📄 File Changes Summary

### New Files:
- `api.py` (600+ lines) - FastAPI application
- `Module/database.py` (80+ lines) - SQLAlchemy models
- `Module/repository_postgres.py` (120+ lines) - PostgreSQL repo
- `setup_db.py` (80+ lines) - Setup script
- `test_api.py` (400+ lines) - Test suite
- `API_DOCUMENTATION.md` (300+ lines) - API docs
- `requirements.txt` (updated)
- `.env` - Configuration

### Modified Files:
- `requirements.txt` - Added FastAPI, SQLAlchemy, PostgreSQL drivers

## ✨ Features

✅ FastAPI REST API
✅ PostgreSQL Database
✅ SQLAlchemy ORM
✅ Recipe Management
✅ Recipe Synthesis
✅ User Management
✅ Event Planning
✅ Automatic API Documentation
✅ Database Initialization
✅ Environment Configuration
✅ Comprehensive Testing
✅ Error Handling
✅ CORS Support

---

**Status:** ✅ Ready to use
**Version:** 1.0.0
**Last Updated:** December 2025
