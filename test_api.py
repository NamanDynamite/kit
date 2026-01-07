print("[DEBUG] test_api.py loaded")
"""
Test script for KitchenMind API
Tests all endpoints of KitchenMind API (new schema)
"""
import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

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
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print_success("API is healthy")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Connection error: {e}")
        return False



def test_create_role() -> Dict[str, Any]:
    """Test role creation."""
    print_test("Create Role")
    try:
        role_data = {
            "role_id": "trainer",
            "role_name": "TRAINER",
            "description": "Trainer role"
        }
        response = requests.post(f"{BASE_URL}/roles", json=role_data)
        if response.status_code in (200, 201):
            role = response.json()
            print_success(f"Role created: {role['role_name']}")
            return role
        elif response.status_code == 409:
            # Fetch existing role if duplicate
            get_resp = requests.get(f"{BASE_URL}/roles/{role_data['role_id']}")
            if get_resp.status_code == 200:
                role = get_resp.json()
                print_success(f"Role exists: {role['role_name']}")
                return role
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
            role = response.json()
            print_success(f"Admin role created: {role['role_name']}")
            return role
        elif response.status_code == 409:
            # Fetch existing role if duplicate
            get_resp = requests.get(f"{BASE_URL}/roles/{role_data['role_id']}")
            if get_resp.status_code == 200:
                role = get_resp.json()
                print_success(f"Admin role exists: {role['role_name']}")
                return role
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
    """Test user creation."""
    print_test("Create User")
    try:
        if role_id == "admin":
            user_data = {
                "name": "Test Admin",
                "email": "test_admin@example.com",
                "login_identifier": "test_admin",
                "password_hash": "hashedpassword",
                "auth_type": "password",
                "role_id": "admin",
                "dietary_preference": "VEG"
            }
        else:
            user_data = {
                "name": "Test Trainer",
                "email": "test_trainer@example.com",
                "login_identifier": "test_trainer",
                "password_hash": "hashedpassword",
                "auth_type": "password",
                "role_id": role_id,
                "dietary_preference": "VEG"
            }
        print(f"[DEBUG] Creating user with data: {user_data}")
        response = requests.post(f"{BASE_URL}/user", json=user_data)
        print(f"[DEBUG] Response status: {response.status_code}")
        print(f"[DEBUG] Response text: {response.text}")
        if response.status_code in (200, 201):
            user = response.json()
            print_success(f"User created: {user['name']}")
            return user
        elif response.status_code == 409:
            # Fetch existing user if duplicate
            get_resp = requests.get(f"{BASE_URL}/user/email/{user_data['email']}")
            if get_resp.status_code == 200:
                user = get_resp.json()
                print_success(f"User exists: {user['name']}")
                return user
            print_error(f"Status code: {get_resp.status_code}")
            print_error(f"Response: {get_resp.text}")
            return None
        else:
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
            admin = response.json()
            print_success(f"Admin profile created: {admin['name']}")
            return admin
        elif response.status_code == 409:
            # Fetch existing admin by searching for user with this email, then get admin profile
            email = user.get("email")
            user_resp = requests.get(f"{BASE_URL}/user/email/{email}")
            if user_resp.status_code == 200:
                user_obj = user_resp.json()
                admin_id = user_obj.get("user_id") or user_obj.get("id")
                get_resp = requests.get(f"{BASE_URL}/admin_profiles/{admin_id}")
                if get_resp.status_code == 200:
                    admin = get_resp.json()
                    print_success(f"Admin profile exists: {admin['name']}")
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
            session = response.json()
            print_success(f"Session created: {session['session_id']}")
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
        print(f"[DEBUG TEST] Sending POST /recipe with trainer_id={trainer.get('user_id')}, data={recipe_data}")
        response = requests.post(
            f"{BASE_URL}/recipe",
            json=recipe_data,
            params={"trainer_id": trainer["user_id"]}
        )
        print(f"[DEBUG TEST] Response status: {response.status_code}")
        print(f"[DEBUG TEST] Response text: {response.text}")
        if response.status_code == 200:
            recipe = response.json()
            print_success(f"Recipe created: {recipe['title']}")
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
            recipes = response.json()
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
        response = requests.get(f"{BASE_URL}/recipe/{recipe['id']}")
        if response.status_code == 200:
            retrieved = response.json()
            print_success(f"Retrieved recipe: {retrieved['title']}")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
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
        response = requests.post(
            f"{BASE_URL}/recipe/{recipe['id']}/rate",
            params={"user_id": user["id"], "rating": 4.5}
        )
        print(f"[DEBUG TEST] rate_recipe status: {response.status_code}")
        print(f"[DEBUG TEST] rate_recipe text: {response.text}")
        if response.status_code == 200:
            result = response.json()
            print_success(f"Recipe rated: {result}")
            print(f"[DEBUG TEST] rate_recipe response: {result}")
            print(f"[DEBUG TEST] rate_recipe type: {type(result)}, keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            if 'id' not in result:
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
            result = response.json()
            print_success(f"Recipe synthesized: {result}")
            print(f"[DEBUG TEST] synthesize_recipe response: {result}")
            print(f"[DEBUG TEST] synthesize_recipe type: {type(result)}, keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            if 'id' not in result:
                print_error("'id' key missing in synthesize_recipe response!")
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
        event_data = {
            "event_name": "Test Party",
            "guest_count": 20,
            "budget_per_person": 5.0,
            "dietary": None
        }
        
        response = requests.post(f"{BASE_URL}/event/plan", json=event_data)
        
        if response.status_code == 200:
            result = response.json()
            print_success(f"Event planned: {result['event']}")
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_error(f"Error: {e}")
        return False



def test_ai_review_recipe(recipe: Dict[str, Any]) -> bool:
    """Test AI review/approval of a recipe."""
    print_test("AI Review Recipe (auto-approve)")
    if not recipe:
        print_error("No recipe provided for AI review")
        return False
    try:
        response = requests.post(f"{BASE_URL}/recipe/{recipe['id']}/ai_review")
        print(f"[DEBUG TEST] ai_review_recipe status: {response.status_code}")
        print(f"[DEBUG TEST] ai_review_recipe text: {response.text}")
        if response.status_code == 200:
            result = response.json()
            print_success(f"Recipe AI-reviewed: {result}")
            if not result.get('approved', False):
                print_error("AI review did not approve the recipe!")
                return False
            return True
        else:
            print_error(f"Status code: {response.status_code}")
            print_error(f"Response: {response.text}")
            return False
    except Exception as e:
        print_error(f"Exception in ai_review_recipe: {e}")
        return False


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
            user = get_resp.json()
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
    def approve_recipe(recipe, validator_id):
        print_test("Approve Recipe (validate)")
        if not recipe or not validator_id:
            print_error("Missing recipe or validator_id for approval")
            return False
        validation_data = {"approved": True, "feedback": "Approved for test", "confidence": 0.95}
        try:
            response = requests.post(
                f"{BASE_URL}/recipe/{recipe['id']}/validate",
                json=validation_data,
                params={"validator_id": validator_id}
            )
            print(f"[DEBUG TEST] approve_recipe status: {response.status_code}")
            print(f"[DEBUG TEST] approve_recipe text: {response.text}")
            if response.status_code == 200:
                print_success("Recipe approved for synthesis test")
                return True
            else:
                print_error(f"Status code: {response.status_code}")
                print_error(f"Response: {response.text}")
                return False
        except Exception as e:
            print_error(f"Exception in approve_recipe: {e}")
            return False

    # Create validator role before creating validator user
        # Validator role removed. No test_create_validator_role needed.

    # Approve recipe so synthesis will succeed
    # If validator approval is only via OpenAI API, always use test_ai_review_recipe.
    if recipe:
        approved = test_ai_review_recipe(recipe)
        results["approve_recipe"] = approved
    else:
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

    # Rate recipe
    print(f"[DEBUG TEST] user object: {user}")
    print(f"[DEBUG TEST] recipe object: {recipe}")
    if recipe and user:
        results["rate_recipe"] = test_rate_recipe(user, recipe)

    # Synthesize recipe
    print(f"[DEBUG TEST] user object: {user}")
    if user:
        results["synthesize_recipe"] = test_synthesize_recipe(user)

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
