from pydantic import BaseModel, EmailStr, constr

class LoginRequest(BaseModel):
    """Schema for login request."""
    email: EmailStr
    password: constr(min_length=8)

class LoginResponse(BaseModel):
    """Schema for login response."""
    email: str
    message: str

class OTPVerifyRequest(BaseModel):
    """Schema for OTP verification request."""
    email: EmailStr
    otp: constr(strip_whitespace=True, min_length=6, max_length=6, pattern=r'^\d{6}$')

class RefreshRequest(BaseModel):
    """Schema for refresh token request."""
    refresh_token: str

class ChangePasswordRequest(BaseModel):
    """Schema for change password request."""
    old_password: constr(min_length=8)
    new_password: constr(min_length=8)
