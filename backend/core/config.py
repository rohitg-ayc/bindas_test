import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"backend\.env")

# Goggle redirect midleware
SESSION_KEY = os.getenv("SESSION_SECRET_KEY")

# DB details
DB_URL = os.getenv("DATABASE_URL")

# JWT-related settings
SECRET_KEY = os.getenv("SECRET_KEY")
REFRESH_SECRET_KEY = os.getenv("REFRESH_SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_EXPIRE_MIN = os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")
REFRESH_EXPIRE_DAYS = os.getenv("REFRESH_TOKEN_EXPIRE_DAYS")

# Google OAuth credentials
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

# Fernet key for data encryption
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")