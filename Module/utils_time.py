from datetime import datetime
from zoneinfo import ZoneInfo

def get_india_time():
    """Return current time in Asia/Kolkata (UTC+05:30) timezone as a timezone-aware datetime object."""
    return datetime.now(ZoneInfo("Asia/Kolkata"))

# Example usage:

def format_datetime_ampm(dt: datetime) -> str:
    """
    Format a timezone-aware datetime object to a string in 12-hour am/pm format with timezone in IST.
    Converts from any timezone (including UTC) to IST.
    Example: '12-Jan-2026 03:45 PM IST'
    """
    if dt is None:
        return None
    
    # If naive datetime, assume it's UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    
    # Convert to IST
    ist = ZoneInfo("Asia/Kolkata")
    dt_ist = dt.astimezone(ist)
    
    return dt_ist.strftime("%d-%b-%Y %I:%M %p") + " IST"

# Example usage:
# now = get_india_time()
# print(format_datetime_ampm(now))
