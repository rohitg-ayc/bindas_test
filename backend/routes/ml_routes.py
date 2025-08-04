from fastapi import APIRouter
from backend.services.column_classifier import ColumnClassifier
from backend.services.target_detector import TargetColumnDetector
from backend.utils.response_generator import success_response, error_response
from fastapi import FastAPI, UploadFile, File
import logging
import pandas as pd
import numpy as np
import duckdb


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

con = duckdb.connect(r"backend\core\bindas_project.duckdb")

@router.post("/column-segregator-and-target")
async def column_segregator_and_target():
    try:
        # Fetch all tables
        tables = con.execute("SHOW TABLES").fetchdf()['name'].tolist()
        logger.info(f"All tables in DuckDB: {tables}")

        # Filter tables with prefix 'uploaded_data'
        tables = [t for t in tables if t.startswith('uploaded_data')]
        logger.info(f"Filtered tables (prefix=uploaded_data): {tables}")

        if not tables:
            return error_response(message="No uploaded_data tables found in DuckDB.")

        # Load both models
        try:
            classifier = ColumnClassifier(model_path=r"C:\Users\aniketd\aYc_BINDAS 7-9-25\BINDAS\backend\models\Column_classifier.pkl")
            detector = TargetColumnDetector(model_path=r"C:\Users\aniketd\aYc_BINDAS 7-9-25\BINDAS\backend\models\target_detector.pkl")
        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")
            return error_response(message=f"Failed to load models: '{str(e)}'")

        success_tables = []
        failed_tables = []

        for table_name in tables:
            try:
                # Fetch data from table
                df = con.execute(f'SELECT * FROM "{table_name}"').fetchdf()

                #Column Classification
                classification_result = classifier.predict(df)
                if classification_result.get('status') != 'success':
                    raise ValueError(classification_result.get('message', 'Unknown error in column classification'))

                #Target Detection
                target_result = detector.detect_target_column(df)
                if target_result.get('status') != 'success':
                    raise ValueError(target_result.get('message', 'Unknown error in target detection'))

                target_column = target_result.get('target_column')
                target_confidence = target_result.get('confidence')

                # Insert combined results into ml_tags
                for col_name in df.columns:
                    ml_result = classification_result['data'].get(col_name, {})
                    ml_tag = ml_result.get('prediction')
                    classification_confidence = ml_result.get('confidence')

                    is_target = (col_name == target_column)
                    final_target_confidence = target_confidence if is_target else None

                    con.execute("""
                        INSERT INTO ml_tags (table_name, column_name, ml_tag, classification_confidence, is_target, target_confidence)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (table_name, col_name, ml_tag, classification_confidence, is_target, final_target_confidence))

                logger.info(f"Processed table {table_name} and stored combined results in ml_tags")
                success_tables.append(table_name)

            except Exception as e:
                logger.error(f"Failed processing table {table_name}: {str(e)}")
                failed_tables.append({"table": table_name, "error": str(e)})

        if success_tables and not failed_tables:
            return success_response(
                message="All uploaded_data tables processed. Combined results stored in ml_tags.",
                data={"success_tables": success_tables}
            )
        elif failed_tables and not success_tables:
            return error_response(
                message="Failed to process all uploaded_data tables.",
                data={"failed_tables": failed_tables}
            )
        
        else:
            return error_response(
                message="Some uploaded_data tables failed to process.",
                data={
                    "success_tables": success_tables,
                    "failed_tables": failed_tables
                }
            )

    except Exception as e:
        logger.error(f"Unexpected error in column_segregator_and_target: {str(e)}")
        return error_response(message=f"Internal server error: '{str(e)}'")

# @router.post("/target-col")
# async def target_detection(table_name: str):
#     # Fetch data from DuckDB table
#     try:
#         df = con.execute(f'SELECT * FROM "{table_name}"').fetchdf()
#     except Exception as e:
#         return error_response(message=f"Failed to fetch table from DuckDB: {str(e)}")

#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("Table could not be read as a DataFrame.")

#     detector = TargetColumnDetector(model_path=r"C:\Users\aniketd\aYc_BINDAS 7-9-25\BINDAS\backend\models\target_detector.pkl")
#     result = detector.detect_target_column(df)
#     return result


# @router.post("/column-segregator")
# async def column_segregator():
#     tables = con.execute("SHOW TABLES").fetchdf()['name'].tolist()
#     print("Tables in DuckDB--------------------->", tables)

#     for table_name in tables:
#         df = con.execute(f'SELECT * FROM "{table_name}"').fetchdf()
#         classifier = ColumnClassifier(model_path=r"C:\Users\aniketd\aYc_BINDAS 7-9-25\BINDAS\backend\models\Column_classifier.pkl")
#         result = classifier.predict(df)

#         if result['status'] == 'success':
#             for col_name, ml_result in result['data'].items():
#                 prediction = ml_result['prediction']
#                 confidence = ml_result['confidence']

#                 con.execute("""
#                     INSERT INTO ml_tags (table_name, column_name, ml_tag, confidence)
#                     VALUES (?, ?, ?, ?)
#                 """, (table_name, col_name, prediction, confidence))
#             logger.info("All tables processed and results stored in ml_tags")

#         return success_response(message="All tables processed and results stored in ml_tags")



# @router.post("/column-segregator")
# async def column_segregator():
#     try:
#         tables = con.execute("SHOW TABLES").fetchdf()['name'].tolist()
#         logger.info(f"Tables in DuckDB: {tables}")

#         if not tables:
#             return error_response(message="No tables found in DuckDB.")
        
#         # Load model
#         try:
#             classifier = ColumnClassifier(model_path=r"C:\Users\aniketd\aYc_BINDAS 7-9-25\BINDAS\backend\models\Column_classifier.pkl")
#         except Exception as e:
#             logger.error(f"Failed to load model: {str(e)}")
#             return error_response(message=f"Failed to load ML model: '{str(e)}'")

#         success_tables = []
#         failed_tables = []

#         for table_name in tables:
#             try:
#                 df = con.execute(f'SELECT * FROM "{table_name}"').fetchdf()
#                 result = classifier.predict(df)

#                 if result.get('status') == 'success':
#                     data_items = result.get('data', {}).items()
#                     for col_name, ml_result in data_items:
#                         con.execute("""
#                             INSERT INTO ml_tags (table_name, column_name, ml_tag, confidence)
#                             VALUES (?, ?, ?, ?)
#                         """, (table_name, col_name, ml_result['prediction'], ml_result['confidence']))

#                     logger.info(f"Processed table {table_name} and stored results in ml_tags")
#                     success_tables.append(table_name)
#                 else:
#                     raise ValueError(result.get('message', 'Unknown error from model prediction'))

#             except Exception as e:
#                 logger.error(f"Failed processing table {table_name}: {str(e)}")
#                 failed_tables.append({"table": table_name, "error": str(e)})

#         # Response
#         if success_tables and not failed_tables:
#             return success_response(
#                 message="All tables processed and results stored in ml_tags.",
#                 data={"success_tables": success_tables}
#             )
#         elif failed_tables and not success_tables:
#             return error_response(
#                 message="Failed to process all tables.",
#                 data={"failed_tables": failed_tables}
#             )
#         else:
#             # Mixed outcome
#             return error_response(
#                 message="Some tables failed to process.",
#                 data={
#                     "success_tables": success_tables,
#                     "failed_tables": failed_tables
#                 }
#             )

#     except Exception as e:
#         logger.error(f"Unexpected error in column_segregator: {str(e)}")
#         return error_response(message=f"Internal server error: '{str(e)}'")





# @router.post("/target-col")
# async def target_detection():
#     try:
#         # Fetch all tables
#         tables = con.execute("SHOW TABLES").fetchdf()['name'].tolist()
#         logger.info(f"Tables in DuckDB: {tables}")

#         if not tables:
#             return error_response(message="No tables found in DuckDB.")

#         # Load model once before processing
#         try:
#             detector = TargetColumnDetector(model_path=r"C:\Users\aniketd\aYc_BINDAS 7-9-25\BINDAS\backend\models\target_detector.pkl")
#         except Exception as e:
#             logger.error(f"Failed to load target detector model: {str(e)}")
#             return error_response(message=f"Failed to load Target Detector model: '{str(e)}'")

#         success_tables = []
#         failed_tables = []

#         # Process each table
#         for table_name in tables:
#             try:
#                 df = con.execute(f'SELECT * FROM "{table_name}"').fetchdf()
                
#                 if not isinstance(df, pd.DataFrame):
#                     raise TypeError(f"Table {table_name} could not be read as a DataFrame.")

#                 result = detector.detect_target_column(df)

#                 if result.get('status') == 'success':
#                     target_col = result.get('target_column')
#                     confidence = result.get('confidence', None)

#                     # Insert into target_tags table
#                     con.execute("""
#                         INSERT INTO target_tags (table_name, target_column, confidence)
#                         VALUES (?, ?, ?)
#                     """, (table_name, target_col, confidence))

#                     logger.info(f"Detected target column for table {table_name}: {target_col}")
#                     success_tables.append({
#                         "table": table_name,
#                         "target_column": target_col,
#                         "confidence": confidence
#                     })
#                 else:
#                     raise ValueError(result.get('message', 'Unknown error from target detector'))

#             except Exception as e:
#                 logger.error(f"Failed processing table {table_name}: {str(e)}")
#                 failed_tables.append({"table": table_name, "error": str(e)})

#         # Final response
#         if success_tables and not failed_tables:
#             return success_response(
#                 message="Target column detected for all tables.",
#                 data={"success_tables": success_tables}
#             )
#         elif failed_tables and not success_tables:
#             return error_response(
#                 message="Failed to detect target column for all tables.",
#                 data={"failed_tables": failed_tables}
#             )
#         else:
#             # Mixed outcome
#             return error_response(
#                 message="Target column detection failed for some tables.",
#                 data={
#                     "success_tables": success_tables,
#                     "failed_tables": failed_tables
#                 }
#             )

#     except Exception as e:
#         logger.error(f"Unexpected error in target_detection: {str(e)}")
#         return error_response(message=f"Internal server error: '{str(e)}'")







# router = APIRouter()

# con = duckdb.connect('mydb.duckdb')

# @router.post("/target-col")
# async def target_detection(table_name: str):
#     # Fetch data from DuckDB table
#     try:
#         df = con.execute(f'SELECT * FROM "{table_name}"').fetchdf()
#     except Exception as e:
#         return error_response(message=f"Failed to fetch table from DuckDB: {str(e)}")

#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("Table could not be read as a DataFrame.")

#     detector = TargetColumnDetector(model_path=r"C:\Users\aniketd\aYc_BINDAS 7-9-25\BINDAS\backend\models\target_detector.pkl")
#     result = detector.detect_target_column(df)
#     return result

 
 
# @router.post("/col-segregater")
# async def column_segregater(table_name: str):
#     try:
#         df = con.execute(f'SELECT * FROM "{table_name}"').fetchdf()
#     except Exception as e:
#         return error_response(message=f"Falied to fetch table from DuckDB: {str(e)}")
    
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("Table could not be read as a DataFrame.")

#     classifier = ColumnClassifier(model_path=r"C:\Users\aniketd\aYc_BINDAS 7-9-25\BINDAS\backend\models\Column_classifier.pkl")
#     result = classifier.predict(df)
#     return result



# @router.post("/target-col")
# async def target_detection(file: UploadFile = File(...)):
#     # Read the uploaded Excel file into pandas DataFrame
    
#     df = pd.read_csv(file.file)
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("Uploaded file could not be read as a DataFrame.")

#     detector  = TargetColumnDetector(model_path=r"C:\Users\aniketd\Downloads\BINDAS Repo Structure\backend\app\services\target_detector.pkl")
#     result = detector.detect_target_column(df)
#     return result





