from pydantic import BaseModel, EmailStr, Field
from typing import Optional, Literal, Any, List

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    first_name: str
    last_name: str
    name: str
    user_type: Literal["indivisual", "corporate"]
    user_role: Literal["admin", "user"]
    org_name: str
    org_industry: str
    org_contact_email: EmailStr
    org_phone: str
    
class UserRead(BaseModel):
    user_id: int
    user_type: str
    # email: EmailStr
    # name: str
    user_role: Optional[str]
    # is_active: bool
    organization_id: int
    # file_path: Optional[str]
    class Config:
        from_attributes = True

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenRefresh(BaseModel):
    refresh_token: str

class OTPVerify(BaseModel):
    email: EmailStr
    otp: str

class StandardResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    data: Optional[Any] = None
    metadata: Optional[Any] = None
    error: Optional[Any] = None
    
class DBConnectRequest(BaseModel):
    server: Optional[str] = None
    database: Optional[str] = "master"
    username: Optional[str] = None
    password: Optional[str] = None
    driver: str = 'ODBC Driver 17 for SQL Server'
    use_windows_auth: Optional[str] = None
    port: Optional[str] = None
    dsn: Optional[str] = None
    connection_url: Optional[str] = None
    duckdb_path: Optional[str] = None

class SelectDBRequest(BaseModel):
    database: str

class RelationshipDisplay(BaseModel):
    parent: str
    primaryKey: str
    childTable: str
    foreignKey: str
    relationType: str
    displayText: str
    id: Optional[str] = None

class PrimaryKeyDisplay(BaseModel):
    table: str
    key: str
    displayText: str

class Option(BaseModel):
    label: str
    value: str
    id: Optional[str] = None

class UpdateRelationshipsRequest(BaseModel):
    selectedPrimaryKeys: List[str]
    selectedRelationships: List[str]

class RemoveRelationshipRequest(BaseModel):
    relationshipId: str
    
class RelationshipDisplay(BaseModel):
    parent: str
    primaryKey: str
    childTable: str
    foreignKey: str
    relationType: str
    displayText: str
    id: Optional[str] = None

class PrimaryKeyDisplay(BaseModel):
    table: str
    key: str
    displayText: str

class UpdateRelationshipsRequest(BaseModel):
    selectedPrimaryKeys: List[str]
    selectedRelationships: List[str]

class RemoveRelationshipRequest(BaseModel):
    relationshipId: str
