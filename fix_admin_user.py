import os
from sqlalchemy import create_engine, text

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://kitchenmind:password@localhost:5432/kitchenmind"
)

engine = create_engine(DATABASE_URL)

with engine.begin() as conn:
    result = conn.execute(
        text("UPDATE \"user\" SET dietary_preference = 'VEG' WHERE email = 'test_admin@example.com';")
    )
    print(f"Rows updated: {result.rowcount}")

    # Optional: verify
    verify = conn.execute(
        text("SELECT user_id, email, dietary_preference FROM \"user\" WHERE email = 'test_admin@example.com';")
    )
    for row in verify:
        print(dict(row._mapping))
