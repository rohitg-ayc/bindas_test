import os
from pydantic import EmailStr
from dotenv import load_dotenv
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig

load_dotenv()

conf = ConnectionConfig(
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_FROM=os.getenv("MAIL_FROM"),
    MAIL_PORT=int(os.getenv("MAIL_PORT")),
    MAIL_SERVER=os.getenv("MAIL_SERVER"),
    MAIL_STARTTLS=True,         
    MAIL_SSL_TLS=False,    
    USE_CREDENTIALS=True,
)

async def send_otp_email(email: EmailStr, otp: str):
    message = MessageSchema(
        subject="BINDAS Email Verification OTP",
        recipients=[email],
        body=f"Your verification OTP is: {otp} \nValid for only 5 minutes.",
        subtype="plain"
    )
    fm = FastMail(conf)
    await fm.send_message(message)
