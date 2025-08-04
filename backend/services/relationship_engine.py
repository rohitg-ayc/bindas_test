from sqlalchemy import create_engine, text
from collections import defaultdict
import pandas as pd
import urllib.parse
from itertools import combinations
import ast


class RelationDetector:


    @staticmethod
    def map_sql_to_pandas(sql_dtype):
        """Maps SQL Server datatypes to Pandas datatypes."""
        dtype_mapping = {
            "int": "Int64", 
            "bigint": "Int64", 
            "smallint": "Int64", 
            "tinyint": "Int64",
            "bit": "boolean",
            "decimal": "float64",
            "numeric": "float64",
            "money": "float64",
            "smallmoney": "float64",
            "float": "float64",
            "real": "float64",
            "char": "string",
            "varchar": "string",
            "text": "string",
            "nchar": "string",
            "nvarchar": "string",
            "ntext": "string",
            "date": "datetime64[ns]",
            "datetime": "datetime64[ns]",
            "datetime2": "datetime64[ns]",
            "smalldatetime": "datetime64[ns]",
            "time": "string"
        }
        return dtype_mapping.get(sql_dtype.lower(), "object")


    @staticmethod
    def get_table_names(engine, database_name):
        """Fetches all table names from the database."""
        query = text(f"""
            DECLARE @DatabaseName NVARCHAR(128) = '{database_name}';
            DECLARE @TableNames  NVARCHAR(MAX)
            SET @TableNames = N'USE ' + @DatabaseName + N';
                                SELECT name as Table_Names FROM ' + @DatabaseName + N'.sys.tables;';
            EXEC sp_executesql @TableNames;
        """)
        with engine.connect() as conn:
            result = conn.execute(query)
            return result.scalars().all()


    @staticmethod
    def get_column_details(engine, database_name, table_name):
        """Fetches column details (names and datatypes) for a specific table."""
        query = text(f"""
            SET NOCOUNT ON;
            DECLARE @DatabaseName NVARCHAR(128) = '{database_name}';
            DECLARE @TableName NVARCHAR(128) = '{table_name}';
            DECLARE @SQL NVARCHAR(MAX);
            SET @SQL = N'SELECT c.name AS ColumnName, ty.name AS DataType
                        FROM ' + QUOTENAME(@DatabaseName) + '.sys.tables t
                        LEFT JOIN ' + QUOTENAME(@DatabaseName) + '.sys.columns c ON t.object_id = c.object_id
                        LEFT JOIN ' + QUOTENAME(@DatabaseName) + '.sys.types ty ON c.user_type_id = ty.user_type_id
                        WHERE t.name = @TableName;';
            EXEC sp_executesql @SQL, N'@TableName NVARCHAR(128)', @TableName;
        """)
        with engine.connect() as conn:
            result = conn.execute(query)
            return pd.DataFrame(result.fetchall(), columns=result.keys())

 
    @staticmethod
    def get_table_data(engine, database_name, table_name):
        """Fetches all data from a specific table."""
        query = text(f"""
            DECLARE @DatabaseName NVARCHAR(128) = '{database_name}';
            DECLARE @TableName NVARCHAR(128) = '{table_name}';
            DECLARE @SQLQuery NVARCHAR(MAX);
            SET @SQLQuery = 'USE ' + QUOTENAME(@DatabaseName) + '; SELECT * FROM ' + QUOTENAME(@TableName) + ';';
            EXEC sp_executesql @SQLQuery;
        """)
        with engine.connect() as conn:
            result = conn.execute(query)
            return pd.DataFrame(result.fetchall(), columns=result.keys())


    @staticmethod
    def convert_dtypes(df, column_details):
        """Applies datatype conversion based on SQL to Pandas mapping."""
        dtype_mapping = {row["ColumnName"]: RelationDetector.map_sql_to_pandas(row["DataType"]) for _, row in column_details.iterrows()}
        for col, dtype in dtype_mapping.items():
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                print(f"Warning: Could not convert column {col} to {dtype}. Error: {e}")
        return df


# Use this to get table_data_dict
    @staticmethod
    def download_all_tables(engine, database_name):
        """Fetches all tables from the database and stores them in a dictionary with correct datatypes."""
        table_names = RelationDetector.get_table_names(engine, database_name)
        table_data_dict = {}
        for table_name in table_names:
            column_details = RelationDetector.get_column_details(engine, database_name, table_name)
            table_df = RelationDetector.get_table_data(engine, database_name, table_name)
            table_df = RelationDetector.convert_dtypes(table_df, column_details)
            table_data_dict[table_name] = table_df

        # Check for- Keys are table names (strings), Values are valid DataFrames, DataFrames are not empty.
        # If an invalid key-value pair is found, it will be removed automatically.
        table_data_dict = {
        table: df for table, df in table_data_dict.items()
        if isinstance(table, str) and isinstance(df, pd.DataFrame) and not df.empty
        }
        return table_data_dict


# If user wants to use his previous connection then fetch from master database (2 steps)
    @staticmethod
    def fetch_final_primary_keys(engine, database_name, table_data_dict):
        query = f"""
        SELECT 
            tc.TABLE_NAME, 
            kcu.COLUMN_NAME, 
            tc.CONSTRAINT_NAME
        FROM {database_name}.INFORMATION_SCHEMA.TABLE_CONSTRAINTS AS tc
        JOIN {database_name}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS kcu
            ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
        WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
        ORDER BY tc.TABLE_NAME, kcu.ORDINAL_POSITION
        """
        with engine.connect() as conn:
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            # 1. Construct final_primary_keys dictionary in the desired format
            primary_keys_dict = defaultdict(list)
            for _, row in df.iterrows():
                table_name = row['TABLE_NAME']
                column_name = row['COLUMN_NAME']
                if table_name in table_data_dict:
                    primary_keys_dict[table_name].append(column_name)
            
            # Format primary keys: tuple for composite keys, string for single keys
            final_primary_keys = {
                table: tuple(columns) if len(columns) > 1 else columns[0]
                for table, columns in primary_keys_dict.items()
            }
            return final_primary_keys


    @staticmethod
    def fetch_final_foreign_keys(engine, database_name, table_data_dict):
        query = f"""
        SELECT 
            fk.TABLE_NAME AS FK_Table, 
            fk.COLUMN_NAME AS FK_Column, 
            pk.TABLE_NAME AS PK_Table, 
            pk.COLUMN_NAME AS PK_Column,
            rc.CONSTRAINT_NAME,
            fk.ORDINAL_POSITION AS FK_Ordinal,
            pk.ORDINAL_POSITION AS PK_Ordinal
        FROM {database_name}.INFORMATION_SCHEMA.REFERENTIAL_CONSTRAINTS AS rc
        JOIN {database_name}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS fk
            ON rc.CONSTRAINT_NAME = fk.CONSTRAINT_NAME
        JOIN {database_name}.INFORMATION_SCHEMA.KEY_COLUMN_USAGE AS pk
            ON rc.UNIQUE_CONSTRAINT_NAME = pk.CONSTRAINT_NAME
        ORDER BY rc.CONSTRAINT_NAME, fk.ORDINAL_POSITION
        """
        with engine.connect() as conn:
            result = conn.execute(text(query))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

            # 1. Construct final_foreign_keys dictionary in the desired format
            foreign_keys_dict = defaultdict(lambda: defaultdict(list))
            
            for _, row in df.iterrows():
                pk_table = row['PK_Table']
                pk_column = row['PK_Column']
                fk_table = row['FK_Table']
                fk_column = row['FK_Column']
                constraint_name = row['CONSTRAINT_NAME']
                fk_ordinal = row['FK_Ordinal']
                pk_ordinal = row['PK_Ordinal']
            
                if pk_table in table_data_dict and fk_table in table_data_dict:
                    foreign_keys_dict[constraint_name]['pk_table'] = pk_table
                    foreign_keys_dict[constraint_name]['fk_table'] = fk_table
                    
                    # Append columns in the correct ordinal position for composite keys
                    foreign_keys_dict[constraint_name].setdefault('pk_columns', []).append((pk_ordinal, pk_column))
                    foreign_keys_dict[constraint_name].setdefault('fk_columns', []).append((fk_ordinal, fk_column))
            
            # Transform into final structure with proper tuple handling
            final_foreign_keys = defaultdict(dict)
            
            for constraint_name, fk_data in foreign_keys_dict.items():
                pk_table = fk_data['pk_table']
                fk_table = fk_data['fk_table']
                
                # Sort columns by ordinal position
                pk_columns = [col for _, col in sorted(fk_data['pk_columns'])]
                fk_columns = [col for _, col in sorted(fk_data['fk_columns'])]
            
                # Use tuples for composite keys, strings for single keys
                pk_columns = tuple(pk_columns) if len(pk_columns) > 1 else pk_columns[0]
                fk_columns = tuple(fk_columns) if len(fk_columns) > 1 else fk_columns[0]
                
                if pk_columns:
                    if pk_columns not in final_foreign_keys[pk_table]:
                        final_foreign_keys[pk_table][pk_columns] = []
                    final_foreign_keys[pk_table][pk_columns].append((fk_table, fk_columns))
            
            # Remove empty dictionaries from final_foreign_keys
            final_foreign_keys = {k: v for k, v in final_foreign_keys.items() if v}

            return final_foreign_keys


# Function fetch the relationship details if not available
    @staticmethod
    def sort_tables_by_size(table_data_dict):
        """Sort tables based on number of rows, ascending order."""
        return dict(sorted(table_data_dict.items(), key=lambda x: len(x[1])))


    @staticmethod
    def detect_candidate_primary_keys(table_data_dict, max_columns=2):
        """Detect candidate primary keys (single & composite) for each table."""
        primary_keys = {}
        composite_keys = {}

        for table, df in table_data_dict.items():
            single_primary_keys = []
            # For primary key consider only 'string', 'object', 'int' data types
            for col in df.select_dtypes(include=['string', 'object', 'int']).columns:
                # Check primary key conditions
                if df[col].is_unique and df[col].notna().all():
                    single_primary_keys.append(col)
                    
            # If primary key is present then add to dictionary
            if single_primary_keys:
                primary_keys[table] = single_primary_keys

            # If primary key is not present then check for composite key
            if not single_primary_keys:
                # For composite key check for basic conditions like nulls and data types
                non_null_columns = [col for col in df.select_dtypes(include=['string', 'object', 'int']).columns if df[col].notna().all()]
                for r in range(2, min(len(non_null_columns), max_columns) + 1):
                    # Based on max_columns input check for all possible combinations of composite key
                    for combo in combinations(non_null_columns, r):
                        if df[list(combo)].drop_duplicates().shape[0] == df.shape[0]:
                            # If composite key found save it as a tuple
                            composite_keys.setdefault(table, []).append(tuple(combo))

        return primary_keys, composite_keys


    @staticmethod
    def detect_foreign_keys(table_data_dict, primary_keys, composite_keys):
        """
        Detect foreign keys based on detected primary keys.
        
        This function identifies foreign keys by checking whether a column (or a combination of columns)
        in one table (child) is a subset of a primary key in another table (parent). 

        Args:
            table_data_dict (dict): Dictionary where keys are table names and values are DataFrames.
            primary_keys (dict): Dictionary where keys are table names and values are lists of primary key column names.
            composite_keys (dict): Dictionary where keys are table names and values are lists of composite key tuples.

        Returns:
            foreign_keys (dict): Mapping of primary keys to their detected foreign key references.
            reference_count (dict): Count of how many times each primary key column is referenced as a foreign key.
        """

        foreign_keys = {}
        reference_count = {}  # Stores reference count with both table and column names

        # Step 1: Detect foreign keys for single-column primary keys
        for parent_table, primary_key_cols in primary_keys.items():
            for pk_col in primary_key_cols:
                for child_table, df in table_data_dict.items():
                    if parent_table != child_table:  # Avoid self-referencing
                        for fk_col in df.columns:
                            if df[fk_col].isna().all():  # Skip columns that are entirely null
                                continue

                            # Check for data type and subset
                            if df[fk_col].dtype == table_data_dict[parent_table][pk_col].dtype and \
                                    set(df[fk_col].dropna()).issubset(set(table_data_dict[parent_table][pk_col].dropna())):

                                # Store foreign key relationship
                                foreign_keys.setdefault(parent_table, {}).setdefault(pk_col, []).append((child_table, fk_col))

                                # Update reference count at column level
                                reference_count.setdefault(parent_table, {}).setdefault(pk_col, 0)
                                reference_count[parent_table][pk_col] += 1  

        # Step 2: Detect foreign keys for composite primary keys
        for parent_table, composite_keys_list in composite_keys.items():
            for composite_key in composite_keys_list:
                composite_key_tuple = tuple(composite_key)

                for child_table, df in table_data_dict.items():
                    if parent_table != child_table:  # Avoid self-referencing
                        possible_fk_combinations = [cols for cols in combinations(df.columns, len(composite_key))]
                        
                        for fk_combination in possible_fk_combinations:
                            if any(df[col].isna().all() for col in fk_combination):  # Skip if any column in FK pair is NULL
                                continue

                            # Check for combination of foreign keys based on merging
                            if tuple(sorted(fk_combination)) == tuple(sorted(composite_key)):
                                merged = df[list(fk_combination)].dropna().merge(
                                    table_data_dict[parent_table][list(composite_key)].dropna(),
                                    on=list(composite_key),
                                    how="left",
                                    indicator=True
                                )

                                # Check for all merged rows exist
                                if (merged['_merge'] == 'both').all():
                                    # Store composite foreign key relationship
                                    foreign_keys.setdefault(parent_table, {}).setdefault(composite_key_tuple, []).append((child_table, fk_combination))

                                    # Update reference count at column level
                                    reference_count.setdefault(parent_table, {}).setdefault(composite_key_tuple, 0)
                                    reference_count[parent_table][composite_key_tuple] += 1  

        return foreign_keys, reference_count


    @staticmethod
    def finalize_primary_and_foreign_keys(primary_keys, composite_keys, reference_count, foreign_keys, table_data_dict):
        """
        Finalize the primary key and the most valid foreign key relationships for each table.

        Args:
            primary_keys (dict): Candidate primary keys {table_name: [list of single PKs]}.
            composite_keys (dict): Candidate composite primary keys {table_name: [(tuple of composite PKs)]}.
            reference_count (dict): Count of how many times each primary key column is referenced.
            foreign_keys (dict): Mapping of primary keys to detected foreign key references.
            table_data_dict (dict): Dictionary of dataframes for data type comparison.

        Returns:
            final_primary_keys (dict): The finalized primary key for each table.
            final_foreign_keys (dict): The finalized foreign key relationships for each table.
        """

        final_primary_keys = {}  # Stores finalized primary keys for each table
        final_foreign_keys = {}  # Stores finalized foreign key relationships

        # Step 1: Determine Final Primary Keys
        for table in primary_keys.keys() | composite_keys.keys():  # Get all tables with PK candidates
            # Here we are merging the single and composite primary keys because table can have both although we are not considering it in primary key function 
            pk_candidates = primary_keys.get(table, []) + composite_keys.get(table, [])  # Merge single & composite PKs

            if len(pk_candidates) == 1:
                # If only one candidate primary key then take it as final
                final_primary_keys[table] = pk_candidates[0]
            else:
                # More than one candidate then Decide based on reference count(whichever has max)
                max_reference = max(reference_count.get(table, {}).values(), default=0)
                best_pks = [pk for pk in pk_candidates if reference_count.get(table, {}).get(pk, 0) == max_reference]

                if len(best_pks) == 1:
                    final_primary_keys[table] = best_pks[0]  # Unique max FK reference count
                else:
                    # Tie-breaker: Choose PK referenced in the most child tables
                    max_child_tables = 0
                    best_pk = None

                    for pk in best_pks:
                        child_tables = len(foreign_keys.get(table, {}).get(pk, []))
                        if child_tables > max_child_tables:
                            max_child_tables = child_tables
                            best_pk = pk

                    if best_pk:
                        final_primary_keys[table] = best_pk
                    else:
                        # Further tie-breaker: Prefer integers over strings
                        #int_pks = [pk for pk in best_pks if table_data_dict[table][pk].dtype.kind in 'iuf']  # Numeric types
                        int_pks = [
                                    pk for pk in best_pks 
                                    if all(
                                        table_data_dict[table][col].dtype.kind in 'iuf' 
                                        for col in (pk if isinstance(pk, tuple) else [pk])
                                    )
                                ]

                        if int_pks:
                            final_primary_keys[table] = sorted(int_pks)[0]  # Choose the smallest numeric key
                        else:
                            final_primary_keys[table] = sorted(best_pks)[0]  # Choose alphabetically

        # Step 2: Determine Final Foreign Keys based on the chosen Primary Keys
        for table, final_pk in final_primary_keys.items():
            if table in foreign_keys and final_pk in foreign_keys[table]:
                final_foreign_keys[table] = {final_pk: foreign_keys[table][final_pk]}

        return final_primary_keys, final_foreign_keys


    @staticmethod
    def detect_relationships(table_data_dict, final_primary_keys, final_foreign_keys):
        """
        Determine relationship type (1:1, 1:M) based on primary and foreign key references.

        Parameters:
        - table_data_dict (dict): Dictionary of DataFrames where keys are table names.
        - final_primary_keys (dict): Dictionary with table names as keys and finalized primary keys as values.
        - final_foreign_keys (dict): Dictionary with table names as keys and confirmed foreign keys.

        Returns:
        - relationships (dict): Dictionary storing relationships in the format:
        {
            'parent_table': {
                'primary_key': [(child_table, foreign_key, relationship_type)]
            }
        }
        """
        relationships = {}

        for parent_table, fk_dict in final_foreign_keys.items():
            for pk_col, child_tables in fk_dict.items():
                for child_table, fk_col in child_tables:
                    child_df = table_data_dict[child_table]

                    # Ensure fk_col is stored correctly
                    if isinstance(fk_col, tuple):  
                        fk_col_stored = fk_col  # Keep as tuple for composite keys
                    else:
                        fk_col_stored = fk_col  # Keep as string for single foreign key

                    # Check One-to-One (1:1)
                    if child_df[list(fk_col) if isinstance(fk_col, tuple) else [fk_col]].drop_duplicates().shape[0] == child_df.shape[0]:  
                        relation_type = "One-to-One (1:1)"
                    else:
                        relation_type = "One-to-Many (1:M)"

                    # Store relationship info
                    relationships.setdefault(parent_table, {}).setdefault(pk_col, []).append(
                        (child_table, fk_col_stored, relation_type)
                    )

        return relationships

    
    @staticmethod
    def convert_options_to_dict(primarykey_options):
        """Function to convert user selected PK options back to function friendly dictionary format"""
        selected_primary_key = {}
        selected_composite_key = {}
        combined_primary_key = {}

        for option in primarykey_options:
            table, col = option.split(" → ")

            # Initialize list for the table if not present
            if table not in combined_primary_key:
                combined_primary_key[table] = []

            # Convert to tuple if needed
            if col.startswith('(') and col.endswith(')'):
                try:
                    col = ast.literal_eval(col)  # Safely convert string to tuple
                except (SyntaxError, ValueError):
                    pass  # If conversion fails, keep it as a string
            
            combined_primary_key[table].append(col)

        # Separate into primary and composite keys
        for table, pk_list in combined_primary_key.items():
            for col in pk_list:
                if isinstance(col, tuple):
                    if table not in selected_composite_key:
                        selected_composite_key[table] = []
                    selected_composite_key[table].append(col)
                else:
                    if table not in selected_primary_key:
                        selected_primary_key[table] = []
                    selected_primary_key[table].append(col)
                    
        return selected_primary_key, selected_composite_key, combined_primary_key
    

    @staticmethod
    def convert_relationship_options_to_dict(relationship_options):
        """Function to convert user selected relation options back to function friendly dictionary format"""
        relationships_detected = {}

        for option in relationship_options:
            # Split the option string into components
            parent_part, child_part = option.split(" → ")
            
            # Extract parent table and primary key
            parent, pk_str = parent_part.strip(")").split(" (")
            
            # Extract child table, foreign key, and relationship type
            child_info, rel_type = child_part.split(" as (")
            child_table, fk_str = child_info.strip(")").split(" (")

            # Convert string representations of tuples to actual tuples
            try:
                pk = ast.literal_eval(pk_str) if pk_str.startswith("(") else pk_str
                fk = ast.literal_eval(fk_str) if fk_str.startswith("(") else fk_str
            except (ValueError, SyntaxError):
                pk = pk_str
                fk = fk_str

            # Initialize the dictionary structure if not present
            if parent not in relationships_detected:
                relationships_detected[parent] = {}
            if pk not in relationships_detected[parent]:
                relationships_detected[parent][pk] = []

            # Append the relationship details to the list
            relationships_detected[parent][pk].append((child_table, fk, rel_type.strip(")")))

        return relationships_detected