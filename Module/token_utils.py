import jwt
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, Any
from zoneinfo import ZoneInfo

SECRET_KEY = os.getenv("SECRET_KEY", "dev-fallback-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7

def format_expiration_time(unix_timestamp) -> str:
    """Convert Unix timestamp to human-readable format in IST."""
    ist = ZoneInfo("Asia/Kolkata")
    dt = datetime.fromtimestamp(unix_timestamp, tz=ist)
    return dt.strftime('%d-%b-%Y %I:%M %p IST')

def create_access_token(data: dict, expires_delta: timedelta = None) -> Dict[str, Any]:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    # Extract Unix timestamp from datetime object
    exp_timestamp = int(expire.timestamp())
    return {
        "token": token,
        "expires_at": format_expiration_time(exp_timestamp)
    }

def create_refresh_token(data: dict, expires_delta: timedelta = None) -> Dict[str, Any]:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS))
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    # Extract Unix timestamp from datetime object
    exp_timestamp = int(expire.timestamp())
    return {
        "token": token,
        "expires_at": format_expiration_time(exp_timestamp)
    }

def decode_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise Exception("Token expired")
    except jwt.InvalidTokenError:
        raise Exception("Invalid token")
