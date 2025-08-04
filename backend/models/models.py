import datetime
from backend.models.database import Base
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Boolean, DateTime, ForeignKey

class Organization(Base):
    __tablename__ = "organizations"

    organization_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)
    industry = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    phone = Column(String, nullable=True)
    created_on = Column(DateTime, default=datetime.now())
    is_active = Column(Boolean, default=True)
    
    users = relationship("User", back_populates="organization")
    subscriptions = relationship("SubscriptionDetail", back_populates="organization")
    audit_logs = relationship("OrgAuditLog", back_populates="organization")
    
class UserServerDetails(Base):
    __tablename__ = "userserverdetails"  
    id = Column(Integer, primary_key=True)  
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=False)
    encrypted_server = Column(String, nullable=True)
    encrypted_database = Column(String, nullable=True)
    encrypted_username = Column(String, nullable=True)
    encrypted_password = Column(String, nullable=True)
    # encrypted_driver = Column(String, nullable=True)
    encrypted_use_windows_auth = Column(String, nullable=True)
    encrypted_port = Column(String, nullable=True)
    encrypted_dsn = Column(String, nullable=True)
    encrypted_connection_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.now())
    session_expiry = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True)
    encrypted_duckdb_path = Column(String, nullable=True)
    
    user = relationship("User", back_populates="server_details")
    
class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    auth_provider = Column(String, nullable=True)
    google_user_id = Column(String, unique=True, nullable=True)
    email = Column(String, unique=True, index=True, nullable=False)
    email_verified = Column(Boolean, default=False)
    name = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    profile_picture_url = Column(String, nullable=True)
    password_hash = Column(String, nullable=True)
    user_type = Column(String, default="manual")
    user_role = Column(String, default="user")
    organization_id = Column(Integer, ForeignKey("organizations.organization_id"), nullable=True)
    created_on = Column(DateTime, default=datetime.now())
    is_active = Column(Boolean, default=True)

    organization = relationship("Organization", back_populates="users")
    subscriptions = relationship("SubscriptionDetail", back_populates="user")
    audit_logs = relationship("OrgAuditLog", back_populates="user")
    server_details = relationship("UserServerDetails", back_populates="user", cascade="all, delete-orphan")


class EmailOTP(Base):
    __tablename__ = "email_otps"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    otp = Column(String)
    created_on = Column(DateTime, default=datetime.now())
    email_verified = Column(Boolean, default=False)


class SubscriptionMaster(Base):
    __tablename__ = "subscription_master"
    plan_id = Column(Integer, primary_key=True, index=True)
    plan_type = Column(String)
    plan_description = Column(String)
    validitydays = Column(Integer)
    amount = Column(String)
    maintenance_charge = Column(String)
    created_on = Column(DateTime, default=datetime.now())


class SubscriptionDetail(Base):
    __tablename__ = "subscription_details"
    subscription_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    organization_id = Column(Integer, ForeignKey("organizations.organization_id"))
    plan_id = Column(Integer, ForeignKey("subscription_master.plan_id"))
    subscription_start = Column(DateTime)
    subscription_end = Column(DateTime)
    is_active = Column(Boolean, default=True)
    created_on = Column(DateTime, default=datetime.now())

    user = relationship("User", back_populates="subscriptions")
    organization = relationship("Organization", back_populates="subscriptions")


class OrgAuditLog(Base):
    __tablename__ = "org_audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    organization_id = Column(Integer, ForeignKey("organizations.organization_id"))
    action_by_user = Column(String)
    database_name = Column(String)
    user_role = Column(String)
    action_type = Column(String)
    created_on = Column(DateTime, default=datetime.now())

    user = relationship("User", back_populates="audit_logs")
    organization = relationship("Organization", back_populates="audit_logs")
