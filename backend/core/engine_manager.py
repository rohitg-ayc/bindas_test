import os
import threading
from ldap3 import Server, Connection, SIMPLE, NTLM, ALL
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.exc import OperationalError, ProgrammingError
from core.db_helper import DBHelper
from core.config import Config
from utils.exceptions import InvalidCredentialsError, MissingCredentialsError, EngineConnectionError

class EngineManager:
    _instance = None
    _engine_lock = threading.Lock()
    _lock = threading.Lock()
    _engine_pool = {}
    _session_pool = {}


    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(EngineManager, cls).__new__(cls)
                cls._instance._org_key = None  # Lazy init
        return cls._instance

    @property
    def org_key(self):
        if self._org_key is None:
            key = Config.get_org_key()
            if not key:
                raise ValueError("ORG_KEY is not set in the environment.")
            self._org_key = key
        return self._org_key
    
    def force_set_org_key(self, key):
        """Optional method to explicitly set the org_key manually."""
        self._org_key = key
    
    def _make_app_server_engine(self, username, password, host, port, db):
        url = URL.create(
            drivername="mssql+pyodbc",
            username=username,
            password=password,
            host=host, port=port,
            database=db,
            query={"driver": "ODBC Driver 17 for SQL Server"}
        )
        return create_engine(url, pool_size=3, max_overflow=5, pool_recycle=3600)
    
    def _make_client_server_engine(self, username, password, host, db):
        url = URL.create(
            drivername="mssql+pyodbc",
            username=username,
            password=password,
            host=host, #port=port,
            database=db,
            query={"driver": "ODBC Driver 17 for SQL Server"}
        )
        return create_engine(url, pool_size=3, max_overflow=5, pool_recycle=3600)
    
    def _make_AD_client_server_engine(self, username, password, host, db):
        url = URL.create(
            drivername="mssql+pyodbc",
            username=username,
            password=password,
            host=host, #port=port,
            database=db,
            query={"driver": "ODBC Driver 17 for SQL Server",
                    "trusted_connection":"yes"}
        )
        return create_engine(url, pool_size=3, max_overflow=5, pool_recycle=3600)
    

    def authenticate_ad_user(self, domain_controller, domain_name, username, password):
        """Authenticate an AD user against a domain controller."""
        user_dn = f"{username}@{domain_name}"
        server = Server(domain_controller, use_ssl=True, get_info=ALL)


        try:
            conn = Connection(server, user=user_dn, password=password, authentication=SIMPLE,  auto_bind=True) #NTLM
            conn.unbind()
            return True
        except Exception as e:
            return False

    def app_server(self, db="SUVI_AdminHub"):
        if db=='client':
            client_data = self._fetch_client_db_name()
            db=client_data['OrgDBName']
            
        key = f"app::{db}"
        with self._engine_lock:
            if key not in self._engine_pool:
                engine = self._make_app_server_engine(
                    Config.SUVI_USER, Config.SUVI_PASSWORD, 
                    Config.SUVI_HOST, Config.SUVI_PORT, db
                    )
                self._engine_pool[key] = engine
            return self._engine_pool[key]
    
    def client_server_engine(self, db="master", login_type="nativeuser", user_type="admin", username=None, password=None):
        """
        Creates engine for client server based on login and user type.
        - login_type: "aduser" or "nativeuser"
        - user_type: "admin" or "user"
        - username, password: only required if user_type == "user"
        """
        details = self._fetch_client_server_details(self.org_key)

        try:
            # ----------- AD user -----------
            if login_type in ["aduser", "onprem_ad"]:
                domain = details['DomainName']
                domain_controller = details['ServerName']
                if user_type == "admin":
                    username = details['AD_Admin_Username'].split('@')[0]
                    password = details['AD_Admin_Password']
                elif not username or not password:
                    raise MissingCredentialsError("Username and password must be provided for AD user login.")

                # Step 1: Authenticate with AD
                if not self.authenticate_ad_user(domain_controller, domain, username, password):
                    raise InvalidCredentialsError("Invalid Active Directory credentials.")

                # Step 2: Create engine with AD credentials
                ad_user = f"{domain}\\{username}"
                key = f"client::{db}::{login_type}::{user_type}::{username.lower()}"
                with self._engine_lock:
                    if key not in self._engine_pool:
                        engine = self._make_AD_client_server_engine(
                            username=ad_user,
                            password=password,
                            host=details['IPAddress'],
                            db=db
                        )
                        self._engine_pool[key] = engine
                    return self._engine_pool[key]

            # ----------- Native user -----------
            else:
                if user_type == "admin":
                    username = details['System_Admin_Username']
                    password = details['System_Admin_Password']
                elif not username or not password:
                    raise MissingCredentialsError("Username and password must be provided for native user login.")
            

                key = f"client::{db}::{login_type}::{user_type}::{username.lower()}"
                with self._engine_lock:
                    if key not in self._engine_pool:
                        engine = self._make_client_server_engine(
                            username=username,
                            password=password,
                            host=details['IPAddress'],
                            db=db
                        )
                        try:
                            _ = DBHelper.select(engine, "SELECT 1")
                        except Exception as e:
                            raise InvalidCredentialsError("Invalid native credentials.")
                        self._engine_pool[key] = engine
                    return self._engine_pool[key]
        except (InvalidCredentialsError, MissingCredentialsError) as e:
            raise e
        except Exception as e:
            raise EngineConnectionError(f"Unable to create engine: {str(e)}")
    
    def _server_detail_flag(self, org_id):
        engine = self.app_server()
        query = '''SELECT oi.OrgID, oi.OrgKey, oi.OrgDBName, osd.* 
                    FROM OrgInfo as oi
                    LEFT JOIN OrgServerDetails as osd
                    ON oi.OrgID = osd.OrgID 
                    WHERE oi.OrgID = :org_id;'''
        connection_state = DBHelper.select(engine, query, {"org_id": org_id}, data_format='records')[0]
        return connection_state
    
    def _fetch_client_db_name(self):
        engine = self.app_server()
        query = '''SELECT * FROM OrgInfo as oi
                    WHERE OrgKey = :org_key
                '''
        client_data = DBHelper.select(engine, query, {"org_key": self.org_key})

        if not client_data:
            raise ValueError("Invalid ORG_KEY or client setup not found")
        return client_data[0]  # Dict with db
    
    def _fetch_client_server_details(self, org_key):
        if not org_key:
            raise ValueError("ORG_KEY not provided")
        engine = self.app_server()
        query = '''SELECT oi.OrgID, oi.OrgKey, oi.OrgDBName, osd.* 
                    FROM OrgInfo as oi
                    LEFT JOIN OrgServerDetails as osd
                    ON oi.OrgID = osd.OrgID 
                    WHERE oi.OrgKey = :org_key'''
        client_server_data = DBHelper.select(engine, query, {"org_key": org_key})
        if not client_server_data:
            raise ValueError("Invalid ORG_KEY or client setup not found")
        return client_server_data[0]  # Dict with db credentials

    def get_scoped_session(self, engine_key):
        """
        Returns a scoped_session bound to the engine identified by engine_key.
        Caches sessions per engine key.
        """
        with self._engine_lock:
            if engine_key not in self._engine_pool:
                raise ValueError(f"No engine found for key: {engine_key}")

            if engine_key not in self._session_pool:
                session_factory = sessionmaker(bind=self._engine_pool[engine_key])
                self._session_pool[engine_key] = scoped_session(session_factory)
            
            return self._session_pool[engine_key]
        
    def clear(self):
        with self._engine_lock:
            self._engine_pool = {}
            self._session_pool = {}




