print("[DEBUG] test_api.py loaded")
"""
Test script for KitchenMind API
Tests all endpoints of KitchenMind API (new schema)
"""
import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000/api"

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_test(name: str):
    """Print test name."""
    print(f"\n{BLUE}→ Testing: {name}{RESET}")


def print_success(msg: str):
    """Print success message."""
    print(f"  {GREEN}✓ {msg}{RESET}")


def print_error(msg: str):
    """Print error message."""
    print(f"  {RED}✗ {msg}{RESET}")


def test_health_check():
    """Test health check endpoint."""
    print_test("Health Check")
    try:
        # Health check is at root, not under /api
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print_success("API is healthy")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Connection error: {e}")
        return False



def _unwrap(resp_json: Dict[str, Any]) -> Any:
    """Helper to unwrap `{status,message,data}` payloads."""
    if isinstance(resp_json, dict) and "data" in resp_json:
        return resp_json["data"]
    return resp_json


def test_create_role() -> Dict[str, Any]:
    """Test role creation (wrapped response)."""
    print_test("Create Role")
    role_data = {"role_id": "trainer", "role_name": "TRAINER", "description": "Trainer role"}
    try:
        response = requests.post(f"{BASE_URL}/roles", json=role_data)
        if response.status_code in (200, 201):
            payload = _unwrap(response.json())
            print_success(f"Role created: {payload.get('role_name')}")
            return payload
        if response.status_code == 409:
            get_resp = requests.get(f"{BASE_URL}/roles/{role_data['role_id']}")
            if get_resp.status_code == 200:
                payload = _unwrap(get_resp.json())
                print_success(f"Role exists: {payload.get('role_name')}")
                return payload
        print_error(f"Status code: {response.status_code}")
        print_error(f"Response: {response.text}")
        return None
    except Exception as e:
        print_error(f"Error: {e}")
        return None

def test_create_admin_role() -> Dict[str, Any]:
    """Test admin role creation."""
    print_test("Create Admin Role")
    try:
        role_data = {
            "role_id": "admin",
            "role_name": "ADMIN",
            "description": "Admin role"
        }
        response = requests.post(f"{BASE_URL}/roles", json=role_data)
        if response.status_code in (200, 201):
            payload = _unwrap(response.json())
            print_success(f"Admin role created: {payload.get('role_name')}")
            return payload
        elif response.status_code == 409:
            # Fetch existing role if duplicate
            get_resp = requests.get(f"{BASE_URL}/roles/{role_data['role_id']}")
            if get_resp.status_code == 200:
                payload = _unwrap(get_resp.json())
                print_success(f"Admin role exists: {payload.get('role_name')}")
                return payload
            print_error(f"Status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return None
        else:
            print_error(f"Status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Error: {e}")
        return None

def test_create_user(role_id: str = "trainer") -> Dict[str, Any]:
    """Test user creation via register endpoint."""
    print_test("Create User")
    user_data = {
        "first_name": "Test",
        "last_name": "Trainer" if role_id != "admin" else "Admin",
        "email": "test_trainer@example.com" if role_id != "admin" else "test_admin@example.com",
        "phone_number": "+919999999999",
        "password": "TestPass123!@",
        "role": role_id,
    }
    try:
        response = requests.post(f"{BASE_URL}/register", json=user_data)
        if response.status_code in (200, 201):
            payload = response.json()
            if payload.get("status"):
                print_success(f"User registered: {user_data['email']}")
                return user_data
        if response.status_code == 409:
            get_resp = requests.get(f"{BASE_URL}/user/email/{user_data['email']}")
            if get_resp.status_code == 200:
                payload = _unwrap(get_resp.json())
                print_success(f"User exists: {payload.get('email', user_data['email'])}")
                return payload
        print_error(f"Status code: {response.status_code}")
        print_error(f"Response: {response.text}")
        return None
    except Exception as e:
        print_error(f"Error: {e}")
        return None
def test_create_admin_profile(user: Dict[str, Any]) -> Dict[str, Any]:
    """Test admin profile creation."""
    print_test("Create Admin Profile")
    try:
        profile_data = {
            "name": user.get("name", "Test Admin"),
            "email": user.get("email", "test_admin@example.com")
        }
        response = requests.post(f"{BASE_URL}/admin_profiles", json=profile_data)
        if response.status_code in (200, 201):
            admin = _unwrap(response.json())
            print_success(f"Admin profile created: {admin.get('name')}")
            return admin
        elif response.status_code == 409:
            # Fetch existing admin by searching for user with this email, then get admin profile
            email = user.get("email")
            user_resp = requests.get(f"{BASE_URL}/user/email/{email}")
            if user_resp.status_code == 200:
                user_obj = _unwrap(user_resp.json())
                admin_id = user_obj.get("user_id") or user_obj.get("id")
                get_resp = requests.get(f"{BASE_URL}/admin_profiles/{admin_id}")
                if get_resp.status_code == 200:
                    admin = _unwrap(get_resp.json())
                    print_success(f"Admin profile exists: {admin.get('name')}")
                    return admin
                print_error(f"Status code: {get_resp.status_code}")
                print_error(f"Response: {get_resp.text}")
                return None
            print_error(f"Status code: {user_resp.status_code}")
            print_error(f"Response: {user_resp.text}")
            return None
        else:
            print_error(f"Status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Error: {e}")
        return None
def test_create_session(user: Dict[str, Any]) -> Dict[str, Any]:
    """Test session creation."""
    print_test("Create Session")
    if not user:
        print_error("No user provided")
        return None
    try:
        session_data = {
            "user_id": user.get("user_id") or user.get("id")
        }
        response = requests.post(f"{BASE_URL}/session", json=session_data)
        if response.status_code in (200, 201):
            session = _unwrap(response.json())
            print_success(f"Session created: {session.get('session_id')}")
            return session
        else:
            print_error(f"Status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Error: {e}")
        return None
def test_create_admin_action(admin: Dict[str, Any]) -> Dict[str, Any]:
    """Test admin action log creation."""
    print_test("Update Admin Action Fields (User Table)")
    if not admin:
        print_error("No admin provided")
        return None
    try:
        from datetime import datetime
        # Patch the admin user with new admin action fields
        user_id = admin.get("admin_id") or admin.get("user_id") or admin.get("id")
        if not user_id:
            print_error("No user_id found in admin object")
            return None
        patch_data = {
            "admin_action_type": "MANUAL_ADJUSTMENT",
            "admin_action_target_type": "test_target_type",
            "admin_action_target_id": "test_target_id",
            "admin_action_description": "Manual adjustment for test",
            "admin_action_created_at": datetime.utcnow().isoformat()
        }
        response = requests.patch(f"{BASE_URL}/user/{user_id}", json=patch_data)
        if response.status_code in (200, 201, 200):
            updated = response.json()
            print_success(f"Admin action fields updated for user: {user_id}")
            return updated
        else:
            print_error(f"Status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Error: {e}")
        return None


def test_submit_recipe(trainer: Dict[str, Any]) -> Dict[str, Any]:
    """Test recipe submission."""
    print("[DEBUG] test_submit_recipe called")
    print_test("Submit Recipe")
    if not trainer:
        print_error("No trainer provided")
        return None

    try:
        recipe_data = {
            "title": "Idli",  # Use exact title for synthesis to succeed
            "servings": 4,
            "ingredients": [
                {"name": "Rice", "quantity": 300, "unit": "g"},
                {"name": "Urad Dal", "quantity": 100, "unit": "g"},
                {"name": "Water", "quantity": 350, "unit": "ml"},
                {"name": "Salt", "quantity": 5, "unit": "g"}
            ],
            "steps": [
                "Soak rice and urad dal for 4 hours",
                "Grind into smooth batter",
                "Ferment overnight",
                "Steam for 12 minutes"
            ]
        }
        trainer_id = trainer.get("user_id") or trainer.get("id")
        print(f"[DEBUG TEST] Sending POST /recipe with trainer_id={trainer_id}, data={recipe_data}")
        response = requests.post(
            f"{BASE_URL}/recipe",
            json=recipe_data,
            params={"trainer_id": trainer_id}
        )
        print(f"[DEBUG TEST] Response status: {response.status_code}")
        print(f"[DEBUG TEST] Response text: {response.text}")
        if response.status_code == 200:
            recipe = _unwrap(response.json())
            print_success(f"Recipe created: {recipe.get('title')}")
            return recipe
        else:
            print_error(f"Status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Error: {e}")
        return None


def test_get_recipes() -> list:
    """Test recipe listing."""
    print_test("Get Recipes")
    try:
        response = requests.get(f"{BASE_URL}/recipes", params={"approved_only": False})
        if response.status_code == 200:
            recipes = _unwrap(response.json()) or []
            print_success(f"Retrieved {len(recipes)} recipes")
            return recipes
        else:
            print_error(f"Status code: {response.status_code}")
            return []
    except Exception as e:
        print_error(f"Error: {e}")
        return []


def test_get_recipe(recipe: Dict[str, Any]):
    """Test get single recipe."""
    print_test("Get Single Recipe")
    if not recipe:
        print_error("No recipe provided")
        return False
    
    try:
        # Use version_id for single recipe GET
        version_id = recipe.get('version_id')
        if not version_id:
            print_error("No version_id in recipe object")
            return False
        response = requests.get(f"{BASE_URL}/recipe/version/{version_id}")
        if response.status_code == 200:
            retrieved = _unwrap(response.json())
            print_success(f"Retrieved recipe: {retrieved.get('title')}")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False






def test_rate_recipe(user: Dict[str, Any], recipe: Dict[str, Any]):
    """Test recipe rating."""
    print_test("Rate Recipe")
    if not user or not recipe:
        print_error("Missing user or recipe")
        return False
    
    try:
        version_id = recipe.get('version_id')
        if not version_id:
            print_error("No version_id in recipe object for rating")
            return False
        response = requests.post(
            f"{BASE_URL}/recipe/version/{version_id}/rate",
            params={"user_id": user["id"], "rating": 4.5},
            json={"comment": "Great recipe!"}
        )
        print(f"[DEBUG TEST] rate_recipe status: {response.status_code}")
        print(f"[DEBUG TEST] rate_recipe text: {response.text}")
        if response.status_code == 200:
            result = _unwrap(response.json())
            if isinstance(result, dict) and 'id' not in result and 'recipe_id' in result:
                result['id'] = result['recipe_id']
            print_success(f"Recipe rated: {result}")
            print(f"[DEBUG TEST] rate_recipe response: {result}")
            print(f"[DEBUG TEST] rate_recipe type: {type(result)}, keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            if isinstance(result, dict) and 'id' not in result:
                print_error("'id' key missing in rate_recipe response!")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
    except Exception as e:
        print_error(f"Exception in rate_recipe: {e}")
        return False


def test_synthesize_recipe(user: Dict[str, Any]):
    """Test recipe synthesis."""
    print_test("Synthesize Recipe")
    if not user:
        print_error("No user provided")
        return False
    
    try:
        synthesis_data = {
            "dish_name": "Idli",
            "servings": 5,
            "top_k": 5,
            "reorder": True
        }
        response = requests.post(
            f"{BASE_URL}/recipe/synthesize",
            json=synthesis_data,
            params={"user_id": user["id"]}
        )
        print(f"[DEBUG TEST] synthesize_recipe status: {response.status_code}")
        print(f"[DEBUG TEST] synthesize_recipe text: {response.text}")
        if response.status_code == 200:
            result = _unwrap(response.json())
            print_success(f"Recipe synthesized: {result}")
            print(f"[DEBUG TEST] synthesize_recipe response: {result}")
            print(f"[DEBUG TEST] synthesize_recipe type: {type(result)}, keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            if isinstance(result, dict) and 'recipe_id' not in result and 'version_id' not in result:
                print_error("'recipe_id' or 'version_id' key missing in synthesize_recipe response!")
            # Print steps with explicit minutes
            steps = result.get('steps', []) if isinstance(result, dict) else []
            print("\n[TEST] Steps with explicit minutes:")
            for step in steps:
                if any(word in step.lower() for word in ['minute', 'minutes', 'min']):
                    print(f"  - {step}")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
    except Exception as e:
        print_error(f"Exception in synthesize_recipe: {e}")
        return False


def test_plan_event():
    """Test event planning."""
    print_test("Plan Event")
    try:
        # Create or fetch a trainer user for event planning
        user_data = {
            "first_name": "Test",
            "last_name": "Trainer",
            "email": "test_trainer@example.com",
            "phone_number": "+919999999999",
            "password": "TestPass123!@",
            "role": "trainer",
        }
        # Try to create; if exists, fetch it
        reg_resp = requests.post(f"{BASE_URL}/register", json=user_data)
        user_id = None
        if reg_resp.status_code in (200, 201):
            user_id = user_data.get("user_id")
        if not user_id or reg_resp.status_code == 409:
            # User may already exist; fetch it
            user_resp = requests.get(f"{BASE_URL}/user/email/{user_data['email']}")
            if user_resp.status_code == 200:
                user = _unwrap(user_resp.json())
                user_id = user.get("user_id") or user.get("id")
        if not user_id:
            print_error("Could not create or find test user for event planning")
            return False
        
        event_data = {
            "user_id": user_id,
            "event_name": "Test Party",
            "guest_count": 20,
            "budget_per_person": 5.0,
            "dietary": None
        }
        response = requests.post(f"{BASE_URL}/event/plan", json=event_data)
        if response.status_code == 200:
            result = _unwrap(response.json())
            print_success(f"Event planned: {result.get('event') if isinstance(result, dict) else result}")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False




def test_public_recipe_search():
    """Test public recipe search endpoint."""
    print_test("Public Recipe Search")
    try:
        response = requests.get(f"{BASE_URL}/public/recipes")
        if response.status_code == 200:
            recipes = _unwrap(response.json()) or []
            print_success(f"Retrieved {len(recipes)} public recipes")
            return recipes
        else:
            print_error(f"Status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return None
    except Exception as e:
        print_error(f"Error: {e}")
        return None

def test_ai_review_recipe(recipe: Dict[str, Any]) -> bool:
    """Test AI review/approval of a recipe."""
    print_test("AI Review Recipe (auto-approve)")
    if not recipe:
        print_error("No recipe provided for AI review")
        return False
    try:
        # Use version_id for AI review endpoint
        version_id = recipe.get('version_id')
        if not version_id:
            print_error("No version_id in recipe object for AI review")
            return False
        response = requests.post(f"{BASE_URL}/recipe/version/{version_id}/validate")
        print(f"[DEBUG TEST] ai_review_recipe status: {response.status_code}")
        print(f"[DEBUG TEST] ai_review_recipe text: {response.text}")
        if response.status_code == 200:
            result = _unwrap(response.json())
            print_success(f"Recipe AI-reviewed: {result}")
            if isinstance(result, dict) and not result.get('approved', False):
                print_error("AI review did not approve the recipe!")
                return False
            return True
        elif response.status_code == 500:
            # OpenAI API error (e.g., invalid key)
            print_error(f"AI validation service unavailable (OpenAI 401/config issue): {response.status_code}")
            print_error("Skipping AI review (recipe will remain unapproved)")
            return False
        else:
            print_error(f"Status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
    except Exception as e:
        print_error(f"Exception in ai_review_recipe: {e}")
        return False


def test_register_user():
    """Test user registration with OTP."""
    print_test("Register User (with validation)")
    reg_data = {
        "first_name": "Test",
        "last_name": "User",
        "email": "testuser@example.com",
        "phone_number": "+919876543210",
        "password": "TestPass123!@",
        "role": "user"
    }
    resp = requests.post(f"{BASE_URL}/register", json=reg_data)
    if resp.status_code == 201:
        result = resp.json()
        if result.get("status"):
            print_success(f"Registration complete: {result.get('data', {}).get('email', 'N/A')}")
            return reg_data
        else:
            print_error(f"Registration failed: {result.get('message')}")
            return None
    else:
        print_error(f"Status code: {resp.status_code}")
        print_error(f"Response: {resp.text}")
        return None


def test_login_user():
    """Test user login with OTP."""
    print_test("Login User (with OTP)")
    login_data = {
        "email": "testuser@example.com",
        "password": "TestPass123!@"
    }
    resp = requests.post(f"{BASE_URL}/login", json=login_data)
    if resp.status_code == 200:
        result = resp.json()
        if result.get("status"):
            print_success(f"Login initiated: {result.get('data', {}).get('email', 'N/A')}")
            return result.get("data")
        else:
            print_error(f"Login failed: {result.get('message')}")
            return None
    else:
        print_error(f"Status code: {resp.status_code}")
        print_error(f"Response: {resp.text}")
        return None


def test_refresh_token(refresh_token):
    """Test refresh token endpoint."""
    print_test("Refresh Token")
    data = {"refresh_token": refresh_token}
    resp = requests.post(f"{BASE_URL}/refresh-token", json=data)
    if resp.status_code == 200:
        result = resp.json()
        if result.get("status") and result.get("data", {}).get("access_token"):
            print_success("Access token refreshed successfully")
            return result["data"]["access_token"]
        else:
            print_error(f"Refresh failed: {result.get('message')}")
            return None
    else:
        print_error(f"Status code: {resp.status_code}")
        print_error(f"Response: {resp.text}")
        return None

def test_protected_route(access_token):
    """Test protected route with access token."""
    print_test("Protected Route (JWT)")
    headers = {"Authorization": f"Bearer {access_token}"}
    resp = requests.get(f"{BASE_URL}/protected", headers=headers)
    if resp.status_code == 200:
        print_success("Accessed protected route successfully")
        return True
    else:
        print_error(f"Status code: {resp.status_code}")
        print_error(f"Response: {resp.text}")
        return False


def test_verify_otp(email, otp):
    """Test OTP verification."""
    print_test("Verify OTP")
    verify_data = {"email": email, "otp": otp}
    resp = requests.post(f"{BASE_URL}/verify-otp", json=verify_data)
    if resp.status_code == 200:
        result = resp.json()
        if result.get("status"):
            print_success("OTP verified successfully")
            return result.get("data")
        else:
            print_error(f"OTP verification failed: {result.get('message')}")
            return None
    else:
        print_error(f"Status code: {resp.status_code}")
        print_error(f"Response: {resp.text}")
        return None


def run_all_tests():
    """Run all tests for all endpoints."""
    print(f"\n{BLUE}{'='*50}")
    print(f"KitchenMind API Test Suite (Full Endpoints)")
    print(f"{'='*50}{RESET}\n")
    results = {}

    # Health check
    results["health_check"] = test_health_check()

    # Create roles
    role = test_create_role()
    print(f"[DEBUG TEST] role object after creation: {role}")
    results["create_role"] = role is not None

    # Create admin role before admin profile
    admin_role = test_create_admin_role()
    print(f"[DEBUG TEST] admin_role object after creation: {admin_role}")
    results["create_admin_role"] = admin_role is not None

    # Create admin user and always fetch from DB to ensure commit
    user = test_create_user(role_id="admin")
    print(f"[DEBUG TEST] user object after creation: {user}")
    if user and 'email' in user:
        get_resp = requests.get(f"{BASE_URL}/user/email/{user['email']}")
        if get_resp.status_code == 200:
            fetched_user = _unwrap(get_resp.json())
            if fetched_user:
                user = fetched_user
            print(f"[DEBUG TEST] user object fetched from DB: {user}")
    if user and 'id' not in user and 'user_id' in user:
        user['id'] = user['user_id']
        print(f"[DEBUG TEST] user object patched with id: {user}")
    results["create_user"] = user is not None

    # Patch user to have role_id 'admin' for admin profile creation
    if user:
        user['role_id'] = 'admin'

    # Create admin profile
    admin_profile = test_create_admin_profile(user)
    print(f"[DEBUG TEST] admin_profile object after creation: {admin_profile}")
    results["create_admin_profile"] = admin_profile is not None

    # Create session
    session = test_create_session(user)
    print(f"[DEBUG TEST] session object after creation: {session}")
    results["create_session"] = session is not None

    # Create admin action log
    admin_action = test_create_admin_action(admin_profile)
    print(f"[DEBUG TEST] admin_action object after creation: {admin_action}")
    results["create_admin_action"] = admin_action is not None


    # Submit recipe (reuse test_submit_recipe)
    recipe = test_submit_recipe(user)
    print(f"[DEBUG TEST] recipe object after creation: {recipe}")
    if recipe and 'id' not in recipe and 'recipe_id' in recipe:
        recipe['id'] = recipe['recipe_id']
        print(f"[DEBUG TEST] recipe object patched with id: {recipe}")
    results["submit_recipe"] = recipe is not None

    # Approve recipe so synthesis will succeed
    # If validator approval is only via OpenAI API, always use test_ai_review_recipe.
    if recipe:
        print(f"[DEBUG TEST] recipe object before AI review: {recipe}")
        approved = test_ai_review_recipe(recipe)
        print(f"[DEBUG TEST] approve_recipe result: {approved}")
        # Fetch the recipe again to check approval status
        version_id = recipe.get('version_id')
        if version_id:
            resp = requests.get(f"{BASE_URL}/recipe/version/{version_id}")
            if resp.status_code == 200:
                fetched_recipe = _unwrap(resp.json())
                print(f"[DEBUG TEST] Recipe approval status after AI review: approved={fetched_recipe.get('approved')}, version_id={version_id}")
            else:
                print_error(f"[DEBUG] Could not fetch recipe after AI review, status: {resp.status_code}")
        else:
            print_error("[DEBUG] No version_id in recipe to fetch after AI review!")
        if not approved:
            print_error("[DEBUG] Recipe was not approved before synthesis! Check AI review step above.")
            print_error(f"[DEBUG] Recipe object at failure: {recipe}")
        results["approve_recipe"] = approved
    else:
        print_error("[DEBUG] No recipe object to approve!")
        results["approve_recipe"] = False

    # Get recipes
    recipes = test_get_recipes()
    print(f"[DEBUG TEST] recipes list after fetch: {recipes}")
    results["get_recipes"] = len(recipes) > 0

    # Get single recipe
    if recipe:
        print(f"[DEBUG TEST] single recipe object before get: {recipe}")
        results["get_single_recipe"] = test_get_recipe(recipe)

    # Validator-related tests are skipped because 'validator' is not defined.
    # results["create_validator"] = validator is not None
    # print(f"[DEBUG TEST] validator object: {validator}")
    # print(f"[DEBUG TEST] recipe object: {recipe}")
    # if recipe and validator:
    #     results["validate_recipe"] = test_validate_recipe(validator, recipe)


    # Synthesize recipe
    print(f"[DEBUG TEST] user object: {user}")
    if user:
        print(f"[DEBUG TEST] user object before synthesis: {user}")
        print(f"[DEBUG TEST] recipe object before synthesis: {recipe}")
        if not results.get("approve_recipe", False):
            print_error("[DEBUG] Synthesis will likely fail because recipe is not approved!")
        results["synthesize_recipe"] = test_synthesize_recipe(user)

    # Rate recipe
    print(f"[DEBUG TEST] user object: {user}")
    print(f"[DEBUG TEST] recipe object: {recipe}")
    if recipe and user:
        results["rate_recipe"] = test_rate_recipe(user, recipe)

    # Plan event
    results["plan_event"] = test_plan_event()

    # Summary
    print(f"\n{BLUE}{'='*50}")
    print("Test Summary")
    print(f"{'='*50}{RESET}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test_name, result in results.items():
        status = f"{GREEN}✓ PASS{RESET}" if result else f"{RED}✗ FAIL{RESET}"
        print(f"  {test_name}: {status}")

    print(f"\n{BLUE}Total: {passed}/{total} tests passed{RESET}\n")


if __name__ == "__main__":
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print(f"\n{RED}Tests interrupted{RESET}")
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}")
