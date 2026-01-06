"""
PostgreSQL database setup script.
Run this before starting the application for the first time.
"""
import os
from dotenv import load_dotenv
import subprocess
import sys

load_dotenv()

# Database credentials
DB_USER = "kitchenmind"
DB_PASSWORD = "kitchenmind_password"
DB_NAME = "kitchenmind"
DB_HOST = "localhost"
DB_PORT = "5432"

def setup_database():
    """Setup PostgreSQL database and user."""
    
    print("🔧 KitchenMind PostgreSQL Setup")
    print("=" * 50)
    
    # Check if PostgreSQL is installed
    print("\n1. Checking PostgreSQL installation...")
    try:
        result = subprocess.run(['psql', '--version'], capture_output=True, text=True)
        print(f"   ✓ {result.stdout.strip()}")
    except FileNotFoundError:
        print("   ✗ PostgreSQL not found. Please install PostgreSQL first.")
        print("   Download from: https://www.postgresql.org/download/")
        sys.exit(1)
    
    # Create database user
    print("\n2. Creating database user...")
    try:
        subprocess.run([
            'psql', '-U', 'postgres', '-c',
            f"CREATE USER {DB_USER} WITH PASSWORD '{DB_PASSWORD}';"
        ], check=False, capture_output=True)
        print(f"   ✓ User '{DB_USER}' created (or already exists)")
    except Exception as e:
        print(f"   ⚠ Error creating user: {e}")
    
    # Create database
    print("\n3. Creating database...")
    try:
        subprocess.run([
            'psql', '-U', 'postgres', '-c',
            f"CREATE DATABASE {DB_NAME} OWNER {DB_USER};"
        ], check=False, capture_output=True)
        print(f"   ✓ Database '{DB_NAME}' created (or already exists)")
    except Exception as e:
        print(f"   ⚠ Error creating database: {e}")
    
    # Grant privileges
    print("\n4. Granting privileges...")
    try:
        subprocess.run([
            'psql', '-U', 'postgres', '-d', DB_NAME, '-c',
            f"GRANT ALL PRIVILEGES ON DATABASE {DB_NAME} TO {DB_USER};"
        ], check=False, capture_output=True)
        print(f"   ✓ Privileges granted to '{DB_USER}'")
    except Exception as e:
        print(f"   ⚠ Error granting privileges: {e}")
    
    # Create tables
    print("\n5. Creating tables...")
    try:
        from Module.database import init_db
        init_db()
        print("   ✓ Tables created successfully")
    except Exception as e:
        print(f"   ⚠ Error creating tables: {e}")
    
    print("\n" + "=" * 50)
    print("✓ Setup complete!")
    print(f"\nDatabase: {DB_NAME}")
    print(f"User: {DB_USER}")
    print(f"Host: {DB_HOST}:{DB_PORT}")
    print("\nYou can now run: python -m uvicorn api:app --reload")


if __name__ == "__main__":
    setup_database()
