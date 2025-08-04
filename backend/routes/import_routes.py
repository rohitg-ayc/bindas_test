from fastapi import APIRouter, UploadFile
from backend.services import db_manager

router = APIRouter()

# @router.post("/import")
# def import_table(file: UploadFile):
#     # call db_manager to import into DuckDB
#     pass



import io
import pandas as pd
import duckdb
from fastapi import APIRouter, UploadFile, File

router = APIRouter()
con = duckdb.connect(r"backend\core\bindas_project.duckdb")

@router.post("/import")
async def import_tables(files: list[UploadFile] = File(...)):
    results = []

    for file in files:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        table_name = file.filename.split('.')[0]
        table_name = table_name.replace("-", "_")  

        con.execute(f'CREATE TABLE IF NOT EXISTS "{table_name}" AS SELECT * FROM df')
        results.append({"table_name": table_name, "rows": len(df)})

    return {"message": "All files imported successfully.", "details": results}

from backend.models.schemas import StandardResponse, DBConnectRequest, SelectDBRequest
from backend.models.schemas import UpdateRelationshipsRequest, RemoveRelationshipRequest
from sqlalchemy.orm import Session
from backend.models.database import get_db
from backend.services.security import get_current_user
from fastapi import Depends
import urllib
from backend.utils.exceptions import UserNotFoundError 
from backend.models.models import User, UserServerDetails
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from backend.core.config import ENCRYPTION_KEY, ACCESS_EXPIRE_MIN
from backend.core.duckdb_project_manager import DuckDBProjectManager
from backend.services.relationship_engine import RelationDetector
import graphviz
from os import path, remove
import logging

fernet = Fernet(ENCRYPTION_KEY.encode())

def encrypt_data(data: str) -> str:
    return fernet.encrypt(data.encode()).decode() if data else None

def decrypt_data(encrypted: str) -> str:
    return fernet.decrypt(encrypted.encode()).decode() if encrypted else None

@router.post("/db_explorer", response_model=StandardResponse)
async def db_explorer(data: DBConnectRequest,
                     current_user: dict = Depends(get_current_user),
                     db: Session = Depends(get_db)):
    
    try:
        # Validate user
        user = db.query(User).filter(User.user_id == current_user["user_id"]).first()
        if not user:
            raise UserNotFoundError("User not found")
        
        # Engine creation
        if data.connection_url:
            # Use full connection URL if provided
            engine = create_engine(data.connection_url)

        elif data.dsn:
            # DSN-based connection (ODBC)
            engine = create_engine(
                f"mssql+pyodbc://{data.dsn}?driver={urllib.parse.quote_plus(data.driver)}"
            )

        else:
            # Build connection string manually
            driver_enc = urllib.parse.quote_plus(data.driver)
            if data.port:
                    server = f"{data.server},{data.port}"

            if data.use_windows_auth:
                # Windows Authentication (Trusted_Connection)
                connection_string = (
                    f"mssql+pyodbc://@{server if data.port else data.server}/{data.database}"
                    f"?driver={driver_enc}&trusted_connection=yes&Encrypt=yes&TrustServerCertificate=yes"
                )
            else:
                # SQL Server Authentication
                encoded_password = urllib.parse.quote_plus(data.password)
                connection_string = (
                    f"mssql+pyodbc://{data.username}:{encoded_password}@{data.server}/{data.database}"
                    f"?driver={driver_enc}&Encrypt=yes&TrustServerCertificate=yes"
                )

            engine = create_engine(connection_string)

        # Fetch databases
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sys.databases WHERE name NOT IN ('master', 'model', 'msdb', 'tempdb', 'FlaskJavaAuthDB')")
            )
            databases = [row[0] for row in result.fetchall()]
        engine = None
        
        # Update existing record if found
        userserverdetails = db.query(UserServerDetails).filter(UserServerDetails.user_id == current_user["user_id"]).first()
        session_expiry = datetime.now() + timedelta(minutes=int(ACCESS_EXPIRE_MIN))
        
        if userserverdetails:
            userserverdetails.encrypted_server = encrypt_data(data.server)
            userserverdetails.encrypted_database = encrypt_data(data.database)
            userserverdetails.encrypted_username = encrypt_data(data.username)
            userserverdetails.encrypted_password = encrypt_data(data.password)
            userserverdetails.encrypted_use_windows_auth = encrypt_data(str(data.use_windows_auth))
            userserverdetails.encrypted_port = encrypt_data(data.port)
            # userserverdetails.encrypted_dsn = encrypt_data(data.dsn)
            userserverdetails.encrypted_connection_url = encrypt_data(data.connection_url)
            userserverdetails.created_at = datetime.now()
            userserverdetails.session_expiry = session_expiry
            userserverdetails.is_active = True
            userserverdetails.encrypted_duckdb_path = encrypt_data(data.duckdb_path)
            
        else:
            # Store new connection in UserConnection 
            user_connection = UserServerDetails(
                user_id=current_user["user_id"],
                encrypted_server=encrypt_data(data.server),
                encrypted_database=encrypt_data(data.database),
                encrypted_username=encrypt_data(data.username),
                encrypted_password=encrypt_data(data.password),
                encrypted_use_windows_auth=encrypt_data(str(data.use_windows_auth)),
                encrypted_port=encrypt_data(data.port),
                encrypted_dsn=encrypt_data(data.dsn),
                encrypted_connection_url=encrypt_data(data.connection_url),
                created_at=datetime.now(),
                session_expiry=session_expiry,
                is_active=True,
                encrypted_duckdb_path = encrypt_data(data.duckdb_path)
            )
            
            db.add(user_connection)
        db.commit()

        return StandardResponse(success=True,
                                data={"databases": databases},
                                message="Connection Successful"
                            )

    except SQLAlchemyError as db_error:
        db.rollback()
        return StandardResponse(success=False,
                                status_code=500,
                                message="Database error!",
                                data={},
                                error=str(db_error)
                            )
    except Exception as e:
        return StandardResponse(success=False,
                                status_code=500,
                                message="Connection Failed!",
                                data={},
                                error=str(e))
        
        
    
@router.post("/select_database", response_model=StandardResponse)
async def select_database(
    data: SelectDBRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
    ):
    try:
        userserverdetails = db.query(UserServerDetails).filter(UserServerDetails.user_id == current_user["user_id"],
                                                               UserServerDetails.is_active == True,
                                                               UserServerDetails.session_expiry > datetime.now()).first()

        if not userserverdetails:
            return StandardResponse(success=False,
                                    status_code=400,
                                    message="No database selected or connection lost",
                                    data={})
        
        # fernet = Fernet(ENCRYPTION_KEY.encode())
            
        # Create engine with available details
        if userserverdetails.encrypted_connection_url:
            decry_conn_url = decrypt_data(userserverdetails.encrypted_connection_url)
            engine = create_engine(decry_conn_url)

        # DSN-based connection (ODBC)
        elif userserverdetails.encrypted_dsn:
            decry_dsn = decrypt_data(userserverdetails.encrypted_dsn)
            engine = create_engine(
                f"mssql+pyodbc://{decry_dsn}?driver={urllib.parse.quote_plus("ODBC Driver 17 for SQL Server")}"
            )

        else:
            # Build connection string manually
            driver_enc = urllib.parse.quote_plus("ODBC Driver 17 for SQL Server")
            decry_server = decrypt_data(userserverdetails.encrypted_server)
            
            if userserverdetails.encrypted_port:
                decry_port = decrypt_data(userserverdetails.encrypted_port)
                server = f"{decry_server},{decry_port}"

            if userserverdetails.encrypted_use_windows_auth:
                # Windows Authentication (Trusted_Connection)
                # decry_database = fernet.decrypt(userserverdetails.database.encode()).decode()
                    
                connection_string = (
                    f"mssql+pyodbc://@{server if userserverdetails.encrypted_port else decry_server}/{data.database}"
                    f"?driver={driver_enc}&trusted_connection=yes&Encrypt=yes&TrustServerCertificate=yes"
                )
            else:
                # SQL Server Authentication
                decry_username = decrypt_data(userserverdetails.encrypted_username)
                decry_pass = decrypt_data(userserverdetails.encrypted_password)
                encoded_password = urllib.parse.quote_plus(decry_pass)
                
                connection_string = (
                    f"mssql+pyodbc://{decry_username}:{encoded_password}@{server if userserverdetails.encrypted_port else decry_server}/{data.database}"
                    f"?driver={driver_enc}&Encrypt=yes&TrustServerCertificate=yes"
                )

            engine = create_engine(connection_string)
        
        # Verify database access
        try:
            with engine.connect() as conn:
                conn.execute(text(f"USE [{data.database}]; SELECT TOP 1 name FROM sys.tables;"))
    
        except Exception as e:
            raise Exception(f"Database '{data.database}' is not accessible or permission denied.")
            
        
        # Fetch table data and store in DuckDB
        table_data_dict = RelationDetector.download_all_tables(engine, data.database)
        duckdb_path =decrypt_data(userserverdetails.encrypted_duckdb_path)
        # duckdb_path = f"duckdb_{current_user['user_id']}_{int(datetime.now().timestamp())}.duckdb"
        duckdb_manager = DuckDBProjectManager(duckdb_path, ENCRYPTION_KEY)
        duckdb_manager.create_project_db()
        for table_name, df in table_data_dict.items():
            duckdb_manager.create_table_from_dataframe(f"uploaded_{table_name}", df)
            duckdb_manager.create_table_from_dataframe(f"transformed_{table_name}", df)

        # Store initial metadata and relationships
        final_primary_keys = RelationDetector.fetch_final_primary_keys(engine, data.database, table_data_dict)
        final_foreign_keys = RelationDetector.fetch_final_foreign_keys(engine, data.database, table_data_dict)
        relationships = RelationDetector.detect_relationships(table_data_dict, final_primary_keys, final_foreign_keys)
        # duckdb_manager.store_metadata(table_data_dict, final_primary_keys, final_foreign_keys)#, relationships)
        duckdb_manager.insert_temp(current_user['user_id'], final_primary_keys, final_foreign_keys, relationships)
        duckdb_manager.close()
        
        # duckdb_manager.encrypt_project_file()
        
        engine = None
        
        # Update data
        userserverdetails.selected_db = encrypt_data(data.database)
        userserverdetails.encrypted_duckdb_path = encrypt_data(duckdb_path)
        db.commit()

        return StandardResponse(success=True,
                                data={"selected_db": data.database},
                                message=f"Database '{data.database}' selected successfully!")

    except SQLAlchemyError as db_error:
        db.rollback()
        return StandardResponse(success=False,
                                status_code=500,
                                message="Database error!",
                                data={},
                                error=str(db_error))
    except Exception as e:
        return StandardResponse(success=False,
                                status_code=500,
                                message=f"Internal server error!",
                                data={},
                                error=str(e))


@router.post("/fetch_relationship_from_db", response_model=StandardResponse)
async def fetch_relationship_from_db(current_user: dict = Depends(get_current_user),
                                    db: Session = Depends(get_db)):
    try:
        
        userserverdetails = db.query(UserServerDetails).filter(UserServerDetails.user_id == current_user["user_id"],
                                                               UserServerDetails.is_active == True,
                                                               UserServerDetails.session_expiry > datetime.now()).first()

        if not userserverdetails:
            return StandardResponse(success=False,
                                    status_code=400,
                                    message="No database selected or connection lost",
                                    data={})
        
        # Load DuckDB
        encrypted_duckdb_path = decrypt_data(userserverdetails.encrypted_duckdb_path)
        # duckdb_path = encrypted_duckdb_path.replace(".enc", "")
        
        print("============================> encrypted_duckdb_path", encrypted_duckdb_path)
        
        duckdb_manager = DuckDBProjectManager(encrypted_duckdb_path, ENCRYPTION_KEY)
        # duckdb_manager.decrypt_project_file(encrypted_duckdb_path)
        table_data_dict = duckdb_manager.fetch_transformed_tables(db_path=encrypted_duckdb_path)
            
        # Fetch metadata and relationships
        temp_data = duckdb_manager.fetch_temp_data(db_path=encrypted_duckdb_path)
        
        final_primary_keys = temp_data.get("final_primary_keys", {})
        final_foreign_keys = temp_data.get("final_foreign_keys", {})
        relationships = temp_data.get("relationships", {})
        
        # metadata_df = duckdb_manager.fetch_metadata()
        # relationships_df = duckdb_manager.fetch_relationships()
        # final_primary_keys = {}
        # for _, row in metadata_df[metadata_df['is_primary_key'] == True].iterrows():
        #     table = row['table_name']
        #     if table not in final_primary_keys:
        #         final_primary_keys[table] = []
        #     final_primary_keys[table].append(row['column_name'])
        # relationships = {}
        # for _, row in relationships_df.iterrows():
        #     parent, pk, child_table, fk_col, relation_type = row['source_table'], row['source_column'], row['target_table'], row['target_column'], row['relation_type']
        #     if parent not in relationships:
        #         relationships[parent] = {}
        #     if pk not in relationships[parent]:
        #         relationships[parent][pk] = []
        #     relationships[parent][pk].append((child_table, fk_col, relation_type))
            
        # Generate ER diagram
        graph = graphviz.Digraph(format="svg")
        graph.attr(rankdir="LR", size="12")
        for table, df in table_data_dict.items():
            label = f"<<TABLE BORDER='1' CELLBORDER='1' CELLSPACING='0'>"
            label += f"<TR><TD BGCOLOR='lightblue' COLSPAN='2'><B>{table}</B></TD></TR>"
            for col in df.columns:
                dtype = str(df[col].dtype)
                check_pk = final_primary_keys.get(table, [])
                if len(check_pk) > 0 and isinstance(check_pk[0], tuple):
                    check_pk = check_pk[0]
                if col in check_pk:
                    label += f"<TR><TD ALIGN='LEFT'><B>{col} (PK)</B></TD><TD ALIGN='LEFT'>{dtype}</TD></TR>"
                else:
                    label += f"<TR><TD ALIGN='LEFT'>{col}</TD><TD ALIGN='LEFT'>{dtype}</TD></TR>"
            label += "</TABLE>>"
            graph.node(table, shape="plaintext", label=label)
        for parent, pk_dict in relationships.items():
            for pk, child_data in pk_dict.items():
                for child_table, fk_col, relation_type in child_data:
                    graph.edge(parent, child_table, label=f"{relation_type}\n{fk_col}", color="blue")
        svg_data = graph.pipe(format="svg").decode('utf-8')
        duckdb_manager.close()

        relationship_display = []
        for parent, pk_dict in relationships.items():
            for pk, child_data in pk_dict.items():
                for child_table, fk_col, relation_type in child_data:
                    relationship_display.append({
                        "parent": parent,
                        "primaryKey": pk,
                        "childTable": child_table,
                        "foreignKey": fk_col,
                        "relationType": relation_type,
                        "displayText": f"{parent} ({pk}) → {child_table} ({fk_col}) as ({relation_type})"
                    })

        primary_keys_display = []
        for table, keys in final_primary_keys.items():
            if isinstance(keys, list):
                for key in keys:
                    primary_keys_display.append({
                        "table": table,
                        "key": key,
                        "displayText": f"{table} → {key}"
                    })
            else:
                primary_keys_display.append({
                    "table": table,
                    "key": keys,
                    "displayText": f"{table} → {keys}"
                })

        return StandardResponse(
            success=True,
            data={
                "svg": svg_data,
                "relationships": relationship_display,
                "primaryKeys": primary_keys_display,
                "tables": list(table_data_dict.keys())
            },
            message="Relationships fetched successfully"
        )

    except Exception as e:
        return StandardResponse(success=False, 
                                status_code=500, 
                                message=f"Failed to fetch relationships: {str(e)}", 
                                data={})
        
        
@router.post("/detect_relationships", response_model=StandardResponse)
async def detect_relationships(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        userserverdetails = db.query(UserServerDetails).filter(
            UserServerDetails.user_id == current_user["user_id"],
            UserServerDetails.is_active == True,
            UserServerDetails.session_expiry > datetime.utcnow()
        ).first()
        if not userserverdetails or not userserverdetails.selected_db:
            return StandardResponse(success=False, status_code=400, message="No database selected or connection lost", data={})

        # Load DuckDB
        encrypted_duckdb_path = decrypt_data(userserverdetails.encrypted_duckdb_path)
        duckdb_path = encrypted_duckdb_path.replace(".enc", "")
        duckdb_manager = DuckDBProjectManager(duckdb_path, ENCRYPTION_KEY)
        duckdb_manager.decrypt_project_file(encrypted_duckdb_path)
        table_data_dict = duckdb_manager.fetch_all_tables()

        # Detect relationships
        sorted_table_data_dict = RelationDetector.sort_tables_by_size(table_data_dict)
        primary_keys, composite_keys = RelationDetector.detect_candidate_primary_keys(sorted_table_data_dict, max_columns=2)
        foreign_keys, reference_count = RelationDetector.detect_foreign_keys(sorted_table_data_dict, primary_keys, composite_keys)
        final_primary_keys, final_foreign_keys = RelationDetector.finalize_primary_and_foreign_keys(
            primary_keys, composite_keys, reference_count, foreign_keys, sorted_table_data_dict
        )
        relationships = RelationDetector.detect_relationships(table_data_dict, final_primary_keys, final_foreign_keys)

        # Update DuckDB
        duckdb_manager.conn.execute("DELETE FROM metadata_tables")
        duckdb_manager.store_metadata(table_data_dict, final_primary_keys, final_foreign_keys)
        encrypted_duckdb_path = duckdb_manager.encrypt_project_file()
        duckdb_manager.close()

        # Update UserServerDetails
        userserverdetails.encrypted_duckdb_path = encrypt_data(encrypted_duckdb_path)
        db.commit()

        # Generate ER diagram
        graph = graphviz.Digraph(format="svg")
        graph.attr(rankdir="LR", size="12")
        for table, df in table_data_dict.items():
            label = f"<<TABLE BORDER='1' CELLBORDER='1' CELLSPACING='0'>"
            label += f"<TR><TD BGCOLOR='lightblue' COLSPAN='2'><B>{table}</B></TD></TR>"
            for col in df.columns:
                dtype = str(df[col].dtype)
                check_pk = final_primary_keys.get(table, [])
                if len(check_pk) > 0 and isinstance(check_pk[0], tuple):
                    check_pk = check_pk[0]
                if col in check_pk:
                    label += f"<TR><TD ALIGN='LEFT'><B>{col} (PK)</B></TD><TD ALIGN='LEFT'>{dtype}</TD></TR>"
                else:
                    label += f"<TR><TD ALIGN='LEFT'>{col}</TD><TD ALIGN='LEFT'>{dtype}</TD></TR>"
            label += "</TABLE>>"
            graph.node(table, shape="plaintext", label=label)
        for parent, pk_dict in relationships.items():
            for pk, child_data in pk_dict.items():
                for child_table, fk_col, relation_type in child_data:
                    graph.edge(parent, child_table, label=f"{relation_type}\n{fk_col}", color="blue")
        svg_data = graph.pipe(format="svg").decode('utf-8')

        relationship_display = []
        for parent, pk_dict in relationships.items():
            for pk, child_data in pk_dict.items():
                for child_table, fk_col, relation_type in child_data:
                    rel_id = f"{parent}|{pk}|{child_table}|{fk_col}|{relation_type}"
                    relationship_display.append({
                        "parent": parent,
                        "primaryKey": pk,
                        "childTable": child_table,
                        "foreignKey": fk_col,
                        "relationType": relation_type,
                        "displayText": f"{parent} ({pk}) → {child_table} ({fk_col}) as ({relation_type})",
                        "id": rel_id
                    })

        primary_keys_display = []
        for table, keys in final_primary_keys.items():
            if isinstance(keys, list):
                for key in keys:
                    primary_keys_display.append({
                        "table": table,
                        "key": key,
                        "displayText": f"{table} → {key}"
                    })
            else:
                primary_keys_display.append({
                    "table": table,
                    "key": keys,
                    "displayText": f"{table} → {keys}"
                })

        all_pk_options = []
        for parent_table, primary_key_list in (primary_keys | composite_keys).items():
            for primary_key in primary_key_list:
                all_pk_options.append({
                    "label": f"{parent_table} → {primary_key}",
                    "value": f"{parent_table} → {primary_key}"
                })

        all_rel_options = []
        for parent, pk_dict in relationships.items():
            for pk, child_data in pk_dict.items():
                for child_table, fk_col, rel_type in child_data:
                    rel_id = f"{parent}|{pk}|{child_table}|{fk_col}|{rel_type}"
                    all_rel_options.append({
                        "label": f"{parent} ({pk}) → {child_table} ({fk_col}) as ({rel_type})",
                        "value": f"{parent} ({pk}) → {child_table} ({fk_col}) as ({rel_type})",
                        "id": rel_id
                    })

        return StandardResponse(
            success=True,
            data={
                "svg": svg_data,
                "relationships": relationship_display,
                "primaryKeys": primary_keys_display,
                "allPrimaryKeyOptions": all_pk_options,
                "allRelationshipOptions": all_rel_options,
                "tables": list(table_data_dict.keys())
            },
            message="Relationships detected successfully"
        )

    except Exception as e:
        return StandardResponse(success=False, status_code=500, message=f"Failed to detect relationships: {str(e)}", data={})

@router.post("/update_detected_relationships", response_model=StandardResponse)
async def update_detected_relationships(
    data: UpdateRelationshipsRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        userserverdetails = db.query(UserServerDetails).filter(
            UserServerDetails.user_id == current_user["user_id"],
            UserServerDetails.is_active == True,
            UserServerDetails.session_expiry > datetime.utcnow()
        ).first()
        if not userserverdetails or not userserverdetails.selected_db:
            return StandardResponse(success=False, status_code=400, message="No database selected or connection lost", data={})

        # Load DuckDB
        encrypted_duckdb_path = decrypt_data(userserverdetails.encrypted_duckdb_path)
        duckdb_path = encrypted_duckdb_path.replace(".enc", "")
        duckdb_manager = DuckDBProjectManager(duckdb_path, ENCRYPTION_KEY)
        duckdb_manager.decrypt_project_file(encrypted_duckdb_path)
        table_data_dict = duckdb_manager.fetch_all_tables()

        # Update relationships
        selected_pk, selected_ck, selected_comkey = RelationDetector.convert_options_to_dict(data.selectedPrimaryKeys)
        selected_rel_dict = RelationDetector.convert_relationship_options_to_dict(data.selectedRelationships)

        # Generate ER diagram
        graph = graphviz.Digraph(format="svg")
        graph.attr(rankdir="LR", size="12")
        for table, df in table_data_dict.items():
            label = f"<<TABLE BORDER='1' CELLBORDER='1' CELLSPACING='0'>"
            label += f"<TR><TD BGCOLOR='lightblue' COLSPAN='2'><B>{table}</B></TD></TR>"
            for col in df.columns:
                dtype = str(df[col].dtype)
                check_pk = selected_comkey.get(table, [])
                if len(check_pk) > 0 and isinstance(check_pk[0], tuple):
                    check_pk = check_pk[0]
                if col in check_pk:
                    label += f"<TR><TD ALIGN='LEFT'><B>{col} (PK)</B></TD><TD ALIGN='LEFT'>{dtype}</TD></TR>"
                else:
                    label += f"<TR><TD ALIGN='LEFT'>{col}</TD><TD ALIGN='LEFT'>{dtype}</TD></TR>"
            label += "</TABLE>>"
            graph.node(table, shape="plaintext", label=label)
        for parent, pk_dict in selected_rel_dict.items():
            for pk, child_data in pk_dict.items():
                for child_table, fk_col, relation_type in child_data:
                    graph.edge(parent, child_table, label=f"{relation_type}\n{fk_col}", color="blue")
        svg_data = graph.pipe(format="svg").decode('utf-8')

        # Update DuckDB
        sorted_table_data_dict = RelationDetector.sort_tables_by_size(table_data_dict)
        foreign_keys, reference_count = RelationDetector.detect_foreign_keys(sorted_table_data_dict, selected_pk, selected_ck)
        final_primary_keys, final_foreign_keys = RelationDetector.finalize_primary_and_foreign_keys(
            selected_pk, selected_ck, reference_count, foreign_keys, sorted_table_data_dict
        )
        updated_relationships = RelationDetector.detect_relationships(table_data_dict, final_primary_keys, final_foreign_keys)
        duckdb_manager.conn.execute("DELETE FROM metadata_tables")
        duckdb_manager.store_metadata(table_data_dict, final_primary_keys, final_foreign_keys)
        encrypted_duckdb_path = duckdb_manager.encrypt_project_file()
        duckdb_manager.close()

        # Update UserServerDetails
        userserverdetails.encrypted_duckdb_path = encrypt_data(encrypted_duckdb_path)
        db.commit()

        relationship_display = []
        for parent, pk_dict in selected_rel_dict.items():
            for pk, child_data in pk_dict.items():
                for child_table, fk_col, relation_type in child_data:
                    rel_id = f"{parent}|{pk}|{child_table}|{fk_col}|{relation_type}"
                    relationship_display.append({
                        "parent": parent,
                        "primaryKey": pk,
                        "childTable": child_table,
                        "foreignKey": fk_col,
                        "relationType": relation_type,
                        "displayText": f"{parent} ({pk}) → {child_table} ({fk_col}) as ({relation_type})",
                        "id": rel_id
                    })

        all_rel_options = []
        for parent, pk_dict in updated_relationships.items():
            for pk, child_data in pk_dict.items():
                for child_table, fk_col, rel_type in child_data:
                    rel_id = f"{parent}|{pk}|{child_table}|{fk_col}|{rel_type}"
                    all_rel_options.append({
                        "label": f"{parent} ({pk}) → {child_table} ({fk_col}) as ({rel_type})",
                        "value": f"{parent} ({pk}) → {child_table} ({fk_col}) as ({rel_type})",
                        "id": rel_id
                    })

        return StandardResponse(
            success=True,
            data={
                "svg": svg_data,
                "relationships": relationship_display,
                "allRelationshipOptions": all_rel_options
            },
            message="Relationships updated successfully"
        )

    except Exception as e:
        return StandardResponse(success=False, status_code=500, message=f"Failed to update relationships: {str(e)}", data={})

@router.post("/remove_relationship", response_model=StandardResponse)
async def remove_relationship(
    data: RemoveRelationshipRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        userserverdetails = db.query(UserServerDetails).filter(
            UserServerDetails.user_id == current_user["user_id"],
            UserServerDetails.is_active == True,
            UserServerDetails.session_expiry > datetime.utcnow()
        ).first()
        if not userserverdetails or not userserverdetails.selected_db:
            return StandardResponse(success=False, status_code=400, message="No database selected or connection lost", data={})

        # Load DuckDB
        encrypted_duckdb_path = decrypt_data(userserverdetails.encrypted_duckdb_path)
        duckdb_path = encrypted_duckdb_path.replace(".enc", "")
        duckdb_manager = DuckDBProjectManager(duckdb_path, ENCRYPTION_KEY)
        duckdb_manager.decrypt_project_file(encrypted_duckdb_path)
        table_data_dict = duckdb_manager.fetch_all_tables()

        # Update relationships
        relationship_id = data.relationshipId
        parts = relationship_id.split('|')
        if len(parts) != 5:
            return StandardResponse(success=False, status_code=400, message="Invalid relationship ID format", data={})
        parent, pk, child_table, fk_col, relation_type = parts

        relationships_df = duckdb_manager.fetch_relationships()
        relationships = {}
        for _, row in relationships_df.iterrows():
            p, p_pk, c_table, c_fk, r_type = row['source_table'], row['source_column'], row['target_table'], row['target_column'], row['relation_type']
            if p not in relationships:
                relationships[p] = {}
            if p_pk not in relationships[p]:
                relationships[p][p_pk] = []
            relationships[p][p_pk].append((c_table, c_fk, r_type))

        if parent in relationships and pk in relationships[parent]:
            child_data = relationships[parent][pk]
            for i, (c_table, c_fk, c_rel) in enumerate(child_data):
                if c_table == child_table and c_fk == fk_col and c_rel == relation_type:
                    child_data.pop(i)
                    break
            if not child_data:
                relationships[parent].pop(pk)
            if not relationships[parent]:
                relationships.pop(parent)

        # Update DuckDB
        duckdb_manager.update_relationships(relationships)
        encrypted_duckdb_path = duckdb_manager.encrypt_project_file()
        duckdb_manager.close()

        # Update UserServerDetails
        userserverdetails.encrypted_duckdb_path = encrypt_data(encrypted_duckdb_path)
        db.commit()

        # Generate ER diagram
        metadata_df = duckdb_manager.fetch_metadata()
        final_primary_keys = {}
        for _, row in metadata_df[metadata_df['is_primary_key'] == True].iterrows():
            table = row['table_name']
            if table not in final_primary_keys:
                final_primary_keys[table] = []
            final_primary_keys[table].append(row['column_name'])

        graph = graphviz.Digraph(format="svg")
        graph.attr(rankdir="LR", size="12")
        for table, df in table_data_dict.items():
            label = f"<<TABLE BORDER='1' CELLBORDER='1' CELLSPACING='0'>"
            label += f"<TR><TD BGCOLOR='lightblue' COLSPAN='2'><B>{table}</B></TD></TR>"
            for col in df.columns:
                dtype = str(df[col].dtype)
                check_pk = final_primary_keys.get(table, [])
                if len(check_pk) > 0 and isinstance(check_pk[0], tuple):
                    check_pk = check_pk[0]
                if col in check_pk:
                    label += f"<TR><TD ALIGN='LEFT'><B>{col} (PK)</B></TD><TD ALIGN='LEFT'>{dtype}</TD></TR>"
                else:
                    label += f"<TR><TD ALIGN='LEFT'>{col}</TD><TD ALIGN='LEFT'>{dtype}</TD></TR>"
            label += "</TABLE>>"
            graph.node(table, shape="plaintext", label=label)
        for parent, pk_dict in relationships.items():
            for pk, child_data in pk_dict.items():
                for child_table, fk_col, relation_type in child_data:
                    graph.edge(parent, child_table, label=f"{relation_type}\n{fk_col}", color="blue")
        svg_data = graph.pipe(format="svg").decode('utf-8')

        relationship_display = []
        for parent, pk_dict in relationships.items():
            for pk, child_data in pk_dict.items():
                for child_table, fk_col, relation_type in child_data:
                    rel_id = f"{parent}|{pk}|{child_table}|{fk_col}|{relation_type}"
                    relationship_display.append({
                        "parent": parent,
                        "primaryKey": pk,
                        "childTable": child_table,
                        "foreignKey": fk_col,
                        "relationType": relation_type,
                        "displayText": f"{parent} ({pk}) → {child_table} ({fk_col}) as ({relation_type})",
                        "id": rel_id
                    })

        all_rel_options = []
        for parent, pk_dict in relationships.items():
            for pk, child_data in pk_dict.items():
                for child_table, fk_col, rel_type in child_data:
                    rel_id = f"{parent}|{pk}|{child_table}|{fk_col}|{rel_type}"
                    all_rel_options.append({
                        "label": f"{parent} ({pk}) → {child_table} ({fk_col}) as ({rel_type})",
                        "value": f"{parent} ({pk}) → {child_table} ({fk_col}) as ({rel_type})",
                        "id": rel_id
                    })

        return StandardResponse(
            success=True,
            data={
                "svg": svg_data,
                "relationships": relationship_display,
                "allRelationshipOptions": all_rel_options
            },
            message="Relationship removed successfully"
        )

    except Exception as e:
        return StandardResponse(success=False, status_code=500, message=f"Failed to remove relationship: {str(e)}", data={})

@router.post("/db_disconnect", response_model=StandardResponse)
async def db_disconnect(
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        userserverdetails = db.query(UserServerDetails).filter(
            UserServerDetails.user_id == current_user["user_id"],
            UserServerDetails.is_active == True
        ).first()

        if userserverdetails:
            # Delete DuckDB file
            if userserverdetails.encrypted_duckdb_path:
                encrypted_path = decrypt_data(userserverdetails.encrypted_duckdb_path)
                if path.exists(encrypted_path):
                    remove(encrypted_path)
                duckdb_path = encrypted_path.replace(".enc", "")
                if path.exists(duckdb_path):
                    remove(duckdb_path)
            userserverdetails.is_active = False
            db.commit()
            logging.info(f"Connection closed for user_id: {current_user['user_id']}")

        return StandardResponse(success=True, data={}, message="Disconnected successfully")

    except SQLAlchemyError as db_error:
        db.rollback()
        return StandardResponse(success=False, status_code=500, message=f"Database error: {str(db_error)}", data={})
    except Exception as e:
        return StandardResponse(success=False, status_code=500, message=f"Internal server error: {str(e)}", data={})