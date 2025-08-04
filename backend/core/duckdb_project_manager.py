# import duckdb
# import json
# # from cryptography.fernet import Fernet
# import os

# class DuckDBProjectManager:
#     def __init__(self, db_path):
#         self.db_path = db_path
#         self.conn = None

#     def create_project_db(self):
#         """Creates a new DuckDB file and initializes required tables."""
#         if os.path.exists(self.db_path):
#             print(f"Project file already exists: {self.db_path}")
#         self.conn = duckdb.connect(self.db_path)
#         self.create_system_tables()
#         print(f"DuckDB project file created at: {self.db_path}")

#     def create_system_tables(self):
#         """Creates system metadata/config tables inside the project file."""

#         queries = [
#             """
#             CREATE TABLE IF NOT EXISTS metadata_tables (
#                 id INT PRIMARY KEY,
#                 table_name TEXT,
#                 column_name TEXT,
#                 data_type TEXT,
#                 is_primary_key BOOLEAN,
#                 is_foreign_key BOOLEAN,
#                 category TEXT
#             )
#             """,
#             """
#             CREATE TABLE IF NOT EXISTS relationships (
#                 id INT PRIMARY KEY,
#                 source_table TEXT,
#                 source_column TEXT,
#                 target_table TEXT,
#                 target_column TEXT,
#                 relation_type TEXT
#             )
#             """,
#             """
#             CREATE TABLE IF NOT EXISTS applied_steps (
#                 id INT PRIMARY KEY,
#                 table_name TEXT,
#                 operation TEXT,
#                 parameters TEXT,
#                 applied_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                 sequence INTEGER
#             )
#             """,
#             """
#             CREATE TABLE IF NOT EXISTS ml_tags (
#                 id INT PRIMARY KEY,
#                 table_name TEXT,
#                 column_name TEXT,
#                 ml_tag TEXT,
#                 classification_confidence REAL,
#                 is_target BOOLEAN,
#                 target_confidence REAL,
#                 tagged_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             )
#             """,
#             """
#             CREATE TABLE IF NOT EXISTS dashboard_configs (
#                 id INT PRIMARY KEY,
#                 dashboard_name TEXT,
#                 config_json TEXT,
#                 created_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             )
#             """,
#             """
#             CREATE TABLE IF NOT EXISTS session_logs (
#                 id INT PRIMARY KEY,
#                 event TEXT,
#                 description TEXT,
#                 event_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#             )
#             """
#         ]

#         for query in queries:
#             self.conn.execute(query)
#         print("All system tables created successfully.")

#     # ---------- Utility Methods ---------- #

#     def insert_metadata(self, table_name, column_name, data_type, is_primary_key, is_foreign_key, category):
#         self.conn.execute("""
#             INSERT INTO metadata_tables (table_name, column_name, data_type, is_primary_key, is_foreign_key, category)
#             VALUES (?, ?, ?, ?, ?, ?)
#         """, (table_name, column_name, data_type, is_primary_key, is_foreign_key, category))

#     def add_relationship(self, source_table, source_column, target_table, target_column, relation_type):
#         self.conn.execute("""
#             INSERT INTO relationships (source_table, source_column, target_table, target_column, relation_type)
#             VALUES (?, ?, ?, ?, ?)
#         """, (source_table, source_column, target_table, target_column, relation_type))

#     def log_applied_step(self, table_name, operation, parameters, sequence):
#         self.conn.execute("""
#             INSERT INTO applied_steps (table_name, operation, parameters, sequence)
#             VALUES (?, ?, ?, ?)
#         """, (table_name, operation, json.dumps(parameters), sequence))

#     def log_ml_tag(self, table_name, column_name, ml_tag, confidence):
#         self.conn.execute("""
#             INSERT INTO ml_tags (table_name, column_name, ml_tag, confidence)
#             VALUES (?, ?, ?, ?)
#         """, (table_name, column_name, ml_tag, confidence))

#     def add_dashboard_config(self, dashboard_name, config_json):
#         self.conn.execute("""
#             INSERT INTO dashboard_configs (dashboard_name, config_json)
#             VALUES (?, ?)
#         """, (dashboard_name, json.dumps(config_json)))

#     def log_session_event(self, event, description):
#         self.conn.execute("""
#             INSERT INTO session_logs (event, description)
#             VALUES (?, ?)
#         """, (event, description))

#     def fetch_metadata(self):
#         return self.conn.execute("SELECT * FROM metadata_tables").fetchdf()

#     def fetch_relationships(self):
#         return self.conn.execute("SELECT * FROM relationships").fetchdf()

#     # ---------- Encryption Methods ---------- #

#     # @staticmethod
#     # def generate_encryption_key():
#     #     """Generates a new Fernet encryption key."""
#     #     return Fernet.generate_key()

#     def encrypt_project_file(self, output_path=None):
#         """Encrypts the DuckDB file and saves it."""
#         if not self.encryption_key:
#             raise ValueError("Encryption key not set.")
#         if not os.path.exists(self.db_path):
#             raise FileNotFoundError("Project file not found.")
        
#         with open(self.db_path, "rb") as f:
#             data = f.read()
#         encrypted_data = self.cipher.encrypt(data)
#         target_path = output_path if output_path else self.db_path + ".enc"
#         with open(target_path, "wb") as f:
#             f.write(encrypted_data)
#         print(f"Encrypted file saved at {target_path}")

#     def decrypt_project_file(self, encrypted_file_path, output_path):
#         """Decrypts an encrypted DuckDB file."""
#         if not self.encryption_key:
#             raise ValueError("Encryption key not set.")
#         if not os.path.exists(encrypted_file_path):
#             raise FileNotFoundError("Encrypted file not found.")
        
#         with open(encrypted_file_path, "rb") as f:
#             encrypted_data = f.read()
#         decrypted_data = self.cipher.decrypt(encrypted_data)
#         with open(output_path, "wb") as f:
#             f.write(decrypted_data)
#         print(f"Decrypted project file saved at {output_path}")

#     def close(self):
#         if self.conn:
#             self.conn.close()
#             print("Database connection closed.")


# # Generate and save key (should be securely stored per user/project)
# # key = DuckDBProjectManager.generate_encryption_key()
# # print(f"Project Encryption Key: {key.decode()}")

# # Create project with encryption
# # db_path = "./data/bindas_project.duckdb"
# # db = DuckDBProjectManager(db_path, encryption_key=key)
# # db.create_project_db()

# # Add metadata entry
# # db.insert_metadata("Sales", "SaleID", "INTEGER", True, False, "Identifier")

# # Encrypt project file
# # db.close()
# # db.encrypt_project_file()

# # # Later: Decrypt file for use
# # db.decrypt_project_file("./data/bindas_project.duckdb.enc", "./data/bindas_project_decrypted.duckdb")












# # # %%
# # from duckdb_project_manager import DuckDBProjectManager
# # db_path = "./bindas_project.duckdb"
# # db = DuckDBProjectManager(db_path)
# # db.create_project_db()
# # db.close()
# # # %%



import duckdb
import pandas as pd
import json
from cryptography.fernet import Fernet
import os
from datetime import datetime

class DuckDBProjectManager:
    def __init__(self, db_path, encryption_key):
        self.db_path = db_path
        self.encryption_key = encryption_key
        self.cipher = Fernet(encryption_key.encode())
        self.conn = None

    def create_project_db(self):
        """Creates a new DuckDB file and initializes required tables."""
        # if os.path.exists(self.db_path):
        #     print(f"Project file already exists: {self.db_path}")
        # self.conn = duckdb.connect(self.db_path)
        # self.create_system_tables()
        # print(f"DuckDB project file created at: {self.db_path}")
        
        try:
            if os.path.exists(self.db_path):
                print(f"Project file already exists: {self.db_path}")

            self.conn = duckdb.connect(self.db_path)
            self.create_system_tables()
            print(f"DuckDB project file created at: {self.db_path}")

        except Exception as e:
            raise Exception(f"Failed to create DuckDB project file at {self.db_path}: {e}")

    def create_system_tables(self):
        """Creates system metadata/config tables inside the project file."""
        queries = [
            """
            CREATE TABLE IF NOT EXISTS metadata_tables (
                id INTEGER PRIMARY KEY,
                table_name VARCHAR,
                column_name VARCHAR,
                data_type VARCHAR,
                is_primary_key BOOLEAN,
                is_foreign_key BOOLEAN,
                category VARCHAR
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS relationships (
                id INTEGER PRIMARY KEY,
                source_table VARCHAR,
                source_column VARCHAR,
                target_table VARCHAR,
                target_column VARCHAR,
                relation_type VARCHAR
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS applied_steps (
                id INTEGER PRIMARY KEY,
                table_name VARCHAR,
                operation VARCHAR,
                parameters VARCHAR,
                applied_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                sequence INTEGER
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS temp (
                id INTEGER PRIMARY KEY,
                final_primary_keys JSON,
                final_foreign_keys JSON,
                relationships JSON,
                applied_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        ]
        for query in queries:
            self.conn.execute(query)
        print("All system tables created successfully.")

    def map_sqlserver_to_duckdb_type(self, pandas_dtype):
        """Maps pandas (SQL Server) data types to DuckDB data types."""
        dtype_str = str(pandas_dtype).lower()
        mapping = {
            'int64': 'INTEGER',
            'int32': 'INTEGER',
            'float64': 'DOUBLE',
            'float32': 'FLOAT',
            'object': 'VARCHAR',
            'string': 'VARCHAR',
            'datetime64[ns]': 'TIMESTAMP',
            'boolean': 'BOOLEAN',
            'bool': 'BOOLEAN',
            # 'nvarchar': 'VARCHAR',
            # 'varchar': 'VARCHAR',
            # 'text': 'VARCHAR',
            # 'bigint': 'BIGINT',
            # 'smallint': 'SMALLINT',
            # 'tinyint': 'TINYINT',
            # 'decimal': 'DECIMAL',
            # 'numeric': 'DECIMAL',
            # 'date': 'DATE',
            # 'datetime': 'TIMESTAMP'
        }
        return mapping.get(dtype_str, 'VARCHAR')  # Default to VARCHAR for unknown types

    def create_table_from_dataframe(self, table_name, df):
        """Creates a DuckDB table from a pandas DataFrame with mapped data types."""
        try: 
            columns = []
            for col in df.columns:
                dtype = self.map_sqlserver_to_duckdb_type(df[col].dtype)
                # Sanitize column name to avoid SQL injection
                # col_safe = col.replace('"', '').replace(' ', '_')
                col_safe = col.replace(' ', '_')
                columns.append(f'"{col_safe}" {dtype}')
            columns_sql = ", ".join(columns)
            create_query = f'CREATE TABLE IF NOT EXISTS "{table_name}" ({columns_sql})'
            self.conn.execute(create_query)
            self.conn.register('temp_df', df)
            self.conn.execute(f'INSERT INTO "{table_name}" SELECT * FROM temp_df')
            self.conn.unregister('temp_df')
            
        except Exception as e:
            raise Exception(f"Failed to fetch the data!: {e}")

    # def store_metadata(self, table_data_dict, primary_keys, foreign_keys, relationships):
    #     """Stores table metadata and relationships in DuckDB."""
        # for table, df in table_data_dict.items():
        #     for col in df.columns:
        #         dtype = self.map_sqlserver_to_duckdb_type(df[col].dtype)
        #         is_pk = col in primary_keys.get(table, [])
        #         is_fk = any(col == fk_col for _, fk_col, _ in foreign_keys.get(table, []))
        # self.insert_metadata(table, col, dtype, is_pk, is_fk, None)

        # for parent, pk_dict in relationships.items():
        #     for pk, child_data in pk_dict.items():
        #         for child_table, fk_col, relation_type in child_data:
        #             self.add_relationship(parent, pk, child_table, fk_col, relation_type)

    def insert_metadata(self, table_name, column_name, data_type, is_primary_key, is_foreign_key, category):
        self.conn.execute("""
            INSERT INTO metadata_tables (table_name, column_name, data_type, is_primary_key, is_foreign_key, category)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (table_name, column_name, data_type, is_primary_key, is_foreign_key, category))

    def insert_temp(self, user_id, final_primary_keys, final_foreign_keys, relationships):
        self.conn.execute("""
            INSERT INTO temp (id, final_primary_keys, final_foreign_keys, relationships)
            VALUES (?, ?, ?, ?)
        """, (user_id, final_primary_keys, final_foreign_keys, relationships))
    
    def add_relationship(self, source_table, source_column, target_table, target_column, relation_type):
        self.conn.execute("""
            INSERT INTO relationships (source_table, source_column, target_table, target_column, relation_type)
            VALUES (?, ?, ?, ?, ?)
        """, (source_table, source_column, target_table, target_column, relation_type))

    def fetch_table_data(self, table_name):
        """Fetches a table as a pandas DataFrame from DuckDB."""
        return self.conn.execute(f'SELECT * FROM "{table_name}"').fetchdf()

    # def fetch_all_tables(self):
    #     """Fetches all tables as a dictionary of DataFrames."""
    #     tables = self.conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
    #     return {table[0]: self.fetch_table_data(table[0]) for table in tables if table[0] not in ['metadata_tables', 'relationships', 'applied_steps']}
    
    def fetch_transformed_tables(self, db_path):
        """Fetches all tables starting with 'transformed_' as a dictionary of DataFrames."""
        try:
            conn = duckdb.connect(db_path)
            
            if conn is None:
                raise ValueError(f"Failed to connect to DuckDB. Invalid or missing database path.: {db_path}")

            # Fetch all table names in the 'main' schema
            tables = conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
            """).fetchall()

            # Filter for tables starting with 'transformed_'
            transformed_tables = [
                table[0] for table in tables 
                if table[0].startswith('transformed_') and table[0] not in ['metadata_tables', 'relationships', 'applied_steps']
            ]

            # Raise exception if none found
            if not transformed_tables:
                raise ValueError("No tables found in the database.")

            # Return dictionary of DataFrames
            return {table: conn.execute(f'SELECT * FROM "{table}"').fetchdf() for table in transformed_tables}

        except Exception as e:
            raise Exception(f"Error fetching transformed tables: {e}")

    def fetch_metadata(self):
        return self.conn.execute("SELECT * FROM metadata_tables").fetchdf()

    def fetch_relationships(self):
        return self.conn.execute("SELECT * FROM relationships").fetchdf()

    def update_relationships(self, relationships):
        """Updates relationships in the relationships table."""
        self.conn.execute("DELETE FROM relationships")
        for parent, pk_dict in relationships.items():
            for pk, child_data in pk_dict.items():
                for child_table, fk_col, relation_type in child_data:
                    self.add_relationship(parent, pk, child_table, fk_col, relation_type)

    def fetch_temp_data(self, db_path):
        """Fetches final_primary_keys, final_foreign_keys, and relationships from the 'temp' table."""
        try:
            conn = duckdb.connect(db_path)
            
            if conn is None:
                raise ValueError(f"Failed to connect to DuckDB. Invalid or missing database path.: {db_path}")

            # Fetch the one row from 'temp'
            row = conn.execute("""
                SELECT final_primary_keys, final_foreign_keys, relationships 
                FROM temp 
                LIMIT 1
            """).fetchone()

            if not row:
                raise ValueError("Relation data is not avilable.")

            # Parse JSON columns
            final_primary_keys = json.loads(row[0])
            final_foreign_keys = json.loads(row[1])
            relationships = json.loads(row[2])

            return {
                "final_primary_keys": final_primary_keys,
                "final_foreign_keys": final_foreign_keys,
                "relationships": relationships
            }

        except Exception as e:
            raise Exception(f"Error fetching relation data: {e}")

    def encrypt_project_file(self):
        """Encrypts the DuckDB file and saves it."""
        if not os.path.exists(self.db_path):
            raise FileNotFoundError("Project file not found.")
        with open(self.db_path, "rb") as f:
            data = f.read()
        encrypted_data = self.cipher.encrypt(data)
        encrypted_path = self.db_path + ".enc"
        with open(encrypted_path, "wb") as f:
            f.write(encrypted_data)
        os.remove(self.db_path)  # Remove unencrypted file
        print(f"Encrypted file saved at {encrypted_path}")
        return encrypted_path

    def decrypt_project_file(self, encrypted_file_path):
        """Decrypts an encrypted DuckDB file."""
        if not os.path.exists(encrypted_file_path):
            raise FileNotFoundError("Encrypted file not found.")
        with open(encrypted_file_path, "rb") as f:
            encrypted_data = f.read()
        decrypted_data = self.cipher.decrypt(encrypted_data)
        with open(self.db_path, "wb") as f:
            f.write(decrypted_data)
        print(f"Decrypted project file saved at {self.db_path}")

    def close(self):
        if self.conn:
            self.conn.close()
            print("Database connection closed.")
