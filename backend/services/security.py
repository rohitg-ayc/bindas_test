import jwt
from backend.core.config import *
from jwt import InvalidTokenError
from passlib.context import CryptContext
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
bearer_scheme = HTTPBearer()

def hash_pwd(password: str) -> str:
    return pwd_ctx.hash(password)

def verify_pwd(plain, hashed) -> bool:
    return pwd_ctx.verify(plain, hashed)

def create_token(data: dict, expires_delta: timedelta, key: str):
    payload = data.copy()
    payload["exp"] = datetime.now() + expires_delta
    return jwt.encode(payload, key, algorithm=ALGORITHM)

def create_access_token(data: dict):
    return create_token(data, timedelta(minutes=int(ACCESS_EXPIRE_MIN)), SECRET_KEY)

def create_refresh_token(data: dict):
    return create_token(data, timedelta(days=int(REFRESH_EXPIRE_DAYS)), REFRESH_SECRET_KEY)

def decode_token(token: str, key: str):
    try:
        return jwt.decode(token, key, algorithms=[ALGORITHM])
    except InvalidTokenError as e:
        raise ValueError(f"Token invalid or expired: {str(e)}")

# Update get_current_user to support both schemes
async def get_current_user(
    token: str = Depends(oauth2_scheme),
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
):
    print(f"Token received: {token or credentials.credentials}")
    token_to_decode = token or credentials.credentials
    payload = decode_token(token_to_decode, SECRET_KEY)
    return payload
