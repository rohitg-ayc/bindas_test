import random
from backend.models.schemas import *
from backend.services import security
from backend.models.database import get_db
from backend.services.security import *
from datetime import datetime
from sqlalchemy.orm import Session
from backend.services.google_oauth import oauth
from backend.utils.response_generator import standard_json_response
from sqlalchemy.exc import SQLAlchemyError
from backend.services.email_utils import send_otp_email
from fastapi import APIRouter, Depends, Request
from backend.models.models import User, EmailOTP, Organization
from fastapi.security import OAuth2PasswordRequestForm
from backend.utils.exceptions import ConfigurationError, UserNotFoundError

router = APIRouter()

# ==========================
# 1. Email login - Token Issuing
# ==========================
@router.post("/login", response_model=StandardResponse)
def login_with_email(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    try:
        # Check In database 
        user = db.query(User).filter(User.email == form_data.username).first()
        
        # If user is not present in db or password is wrong
        if not user or not verify_pwd(form_data.password, user.password_hash):
            raise ValueError("Invalid credentials")

        # If user is not active
        if not user.is_active:
            raise PermissionError("User inactive")

        payload = {
            "user_id": user.user_id,
            "user_type": user.user_type,
            "user_role": user.user_role,
            "organization_id": user.organization_id
        }
        
        access_token=create_access_token(payload)
        refresh_token=create_refresh_token(payload)                                

        return standard_json_response(success=True,
                                      data={"access_token":access_token,
                                            "refresh_token":refresh_token,
                                            "token_type":"bearer"})
        
        # return {
        #     "access_token":access_token,
        #     "refresh_token":refresh_token,
        #     "token_type":"bearer"
        # }
        
    except SQLAlchemyError as db_error:
        return standard_json_response(status_code=500,
                                      success=False,
                                      message=f"Database error: {str(db_error)}")
        
    except Exception as e:
        return standard_json_response(status_code=500,
                                      success=False,
                                      message=f"Internal server error: {str(e)}")

# ==========================
# 2. Send OTP
# ==========================
@router.post("/send-otp", response_model=StandardResponse)
async def send_otp(email: EmailStr, db: Session = Depends(get_db)):
    try:
        # Generate otp
        otp = str(random.randint(100000, 999999))

        # Update or insert in db
        existing = db.query(EmailOTP).filter(EmailOTP.email == email).first()
        if existing:
            existing.otp = otp
            existing.created_on = datetime.now()
        else:
            db.add(EmailOTP(email=email, otp=otp))
        
        # Send OTP email
        await send_otp_email(email, otp)
        
        db.commit()
        
        return standard_json_response(success=True,
                                      message="OTP sent successfully.")
    
    except SQLAlchemyError as db_error:
        db.rollback()
        return standard_json_response(status_code=500,
                                      success=False,
                                      message=f"Database error: {str(db_error)}")
        
    except Exception as e:
        return standard_json_response(status_code=500,
                                      success=False,
                                      message=f"Internal server error: {str(e)}")


# ==========================
# 3. Verify OTP
# ==========================
@router.post("/verify-otp", response_model=StandardResponse)
def verify_otp(data: OTPVerify, db: Session = Depends(get_db)):
    try:
        # Check in db
        record = db.query(EmailOTP).filter(EmailOTP.email == data.email).first()
        
        if not record:
            raise LookupError("OTP record not found")
        
        # Check time limit
        now = datetime.now()
        if now - record.created_on > timedelta(minutes=5):
            raise TimeoutError("OTP expired. Please request a new one.")
        
        if record.otp != data.otp:
            raise ValueError("Invalid OTP")
        
        # Mark email as verified table
        record.email_verified = True
        db.commit()

        return standard_json_response(success=True,
                                      message="OTP verified, Please complete your registration In 1 hour.")
    
    except SQLAlchemyError as db_error:
        db.rollback()
        return standard_json_response(status_code=500,
                                      success=False,
                                      message=f"Database error: {str(db_error)}")
        
    except Exception as e:
        return standard_json_response(status_code=500,
                                      success=False,
                                      message=f"Internal server error: {str(e)}")


# ==========================
# 4. Register Corporate/Individual (post-verification)
# ==========================
@router.post("/register", response_model=StandardResponse)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    try: 
        # Check if already registered
        if db.query(User).filter(User.email == user.email).first():
            raise FileExistsError("User already exists")
        
        # Check if OTP was verified and not expired (59 minutes validity)
        otp_record = db.query(EmailOTP).filter(EmailOTP.email == user.email).first()
        
        if not otp_record or not otp_record.email_verified:
            raise PermissionError("Please verify your email with OTP first.")

        # Optional: enforce time limit after OTP verification
        now = datetime.now()
        if now - otp_record.created_on > timedelta(minutes=59):
            raise TimeoutError("Verification expired. Please resend OTP.")
        
        # Insert org
        org = Organization(
            name=user.org_name,
            industry=user.org_industry,
            contact_email=user.org_contact_email,
            phone=user.org_phone,
            created_on=now,
            is_active=True
        )
        db.add(org)
        # Get org_id without commit
        db.flush()  
    
        new_user = User(
            auth_provider="email",
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            name=user.name,
            email_verified=True,
            user_type=user.user_type,
            user_role=user.user_role,
            organization_id=org.organization_id,
            password_hash=hash_pwd(user.password),
            created_on=datetime.now(),
            is_active=True
        )
        
        db.add(new_user)
        
        # delete OTP record after successful registration
        db.delete(otp_record)
        
        db.commit()
        # db.refresh(new_user)
        
        return standard_json_response(success=True,
                                      message="Registration successful. Please log in.")
    
    except SQLAlchemyError as db_error:
        db.rollback()
        return standard_json_response(status_code=500,
                                      success=False,
                                      message=f"Database error: {str(db_error)}")
        
    except Exception as e:
        return standard_json_response(status_code=500,
                                      success=False,
                                      message=f"Internal server error: {str(e)}")


# ==========================
# 5. Google Login
# ==========================
@router.get("/google-login")
async def login_with_google(request: Request):
    try:
        if not GOOGLE_REDIRECT_URI:
            raise ConfigurationError("Redirect URI not configured")
        
        return await oauth.google.authorize_redirect(request, GOOGLE_REDIRECT_URI)
    
    except Exception as e:
        return standard_json_response(status_code=500,
                                      success=False,
                                      message=f"Google OAuth login failed: {str(e)}")

# ==========================
# 6. Google Redirect
# ==========================
@router.get("/google-callback", response_model=StandardResponse)
async def google_callback(request: Request, db: Session = Depends(get_db)):
    try:
        token = await oauth.google.authorize_access_token(request)
        userinfo = await oauth.google.userinfo(token=token)
    except Exception as e:
        raise RuntimeError(f"OAuth failed: {str(e)}")

    try:
        email = userinfo.get("email")
        if not email:
            raise ValueError("Google account has no email")

        user = db.query(User).filter(User.email == email).first()

        # access_token = token.get("access_token")
        # refresh_token = token.get("refresh_token")
        # expires_in = token.get("expires_in")
        # token_expiry = datetime.now() + timedelta(minutes=int(ACCESS_EXPIRE_MIN))

        if user:
            # Check if user already exist in database
            if user.auth_provider != "google":
                raise PermissionError("This email is already registered using manual authentication.\nPlease use email + password to login.")
        
            # If user is in DB Update existing user
            user.auth_provider = "google"
            user.google_user_id = userinfo.get("sub")
            user.name = userinfo.get("name")
            user.first_name = userinfo.get("given_name")
            user.last_name = userinfo.get("family_name")
            user.profile_picture_url = userinfo.get("picture")
            user.email_verified = userinfo.get("email_verified", True)
            user.user_type = "individual"
            user.user_role = "admin"
            
        else:
            # Create placeholder org
            placeholder_org = Organization(
                name="N/A",
                industry="N/A",
                contact_email="N/A",
                phone="N/A",
                created_on=datetime.now(),
                is_active=True
            )
            db.add(placeholder_org)
            db.flush()

            user = User(
                email=email,
                auth_provider="google",
                google_user_id=userinfo.get("sub"),
                name=userinfo.get("name"),
                first_name=userinfo.get("given_name"),
                last_name=userinfo.get("family_name"),
                profile_picture_url=userinfo.get("picture"),
                email_verified=userinfo.get("email_verified", True),
                user_type="individual",
                user_role="admin",
                organization_id=placeholder_org.organization_id,
                is_active=True,
                created_on=datetime.now(),
            )
            db.add(user)

        db.commit()
        db.refresh(user)

        # Final JWT payload
        jwt_payload = {
            "user_id": user.user_id,
            "user_type": user.user_type,
            "user_role": user.user_role,
            "organization_id": user.organization_id
        }

        return standard_json_response(success=True,
                                      data={"access_token": security.create_access_token(jwt_payload),
                                            "refresh_token": security.create_refresh_token(jwt_payload),
                                            "token_type":"bearer"})
    
    except SQLAlchemyError as db_error:
        db.rollback()
        return standard_json_response(status_code=500,
                                      success=False,
                                      message=f"Database error: {str(db_error)}")
        
    except Exception as e:
        return standard_json_response(status_code=500,
                                      success=False,
                                      message=f"Internal server error: {str(e)}")



# ==========================
# 7. Get Current User from Token
# ==========================
@router.get("/me", response_model=StandardResponse)
def get_me(current_user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        # Check in db    
        print("=========================> current_user",current_user)
        user = db.query(User).filter(User.user_id == current_user["user_id"]).first()
        
        if not user:
            raise UserNotFoundError("User not found")
        
        return standard_json_response(success=True,
                                      data={"user_id":user.user_id,
                                            "user_type":user.user_type,
                                            "user_role":user.user_role,
                                            "organization_id":user.organization_id})
    
    except SQLAlchemyError as db_error:
        return standard_json_response(status_code=500,
                                      success=False,
                                      message=f"Database error: {str(db_error)}")
        
    except Exception as e:
        return standard_json_response(status_code=500,
                                      success=False,
                                      message=f"Internal server error: {str(e)}")


# ==========================
# 8. Refresh Token
# ==========================
@router.post("/refresh", response_model=StandardResponse)
def refresh_token(body: TokenRefresh, db: Session = Depends(get_db)):
    try:
        payload = security.decode_token(body.refresh_token, secret_key=security.REFRESH_SECRET_KEY)
        user_id = payload.get("user_id")

        if not user_id:
            raise ValueError("Invalid token payload")

        user = db.query(User).filter(User.user_id == user_id, User.is_active == True).first()

        if not user:
            raise UserNotFoundError("User not found or inactive")

        jwt_payload = {
            "user_id": user.user_id,
            "user_type": user.user_type,
            "user_role": user.user_role,
            "organization_id": user.organization_id,
        }
        
        return standard_json_response(success=True,
                                      data={"access_token": security.create_access_token(jwt_payload),
                                            "refresh_token": security.create_refresh_token(jwt_payload),
                                            "token_type":"bearer"})
    
    except SQLAlchemyError as db_error:
        return standard_json_response(status_code=500,
                                      success=False,
                                      message=f"Database error: {str(db_error)}")
        
    except Exception as e:
        return standard_json_response(status_code=500,
                                      success=False,
                                      message=f"Internal server error: {str(e)}")

