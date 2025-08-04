from pydantic import BaseModel
from typing import Literal, Dict, Optional
from backend.core.duckdb_project_manager import DuckDBProjectManager
from backend.services.cleaning_engine import DataCleaner
from backend.services.DataFormatter import DataFormatter
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
import numpy as np
import pandas as pd

# ✅ Router must be defined outside of class
router = APIRouter()

# ✅ Request model should not wrap route definitions
class OperationRequest(BaseModel):
    table_name: str
    operation: str
    parameters: Optional[Dict] = {}
    mode: Literal["apply", "save"] = "apply"

@router.get("/available_operations")
def list_available_operations():
    cleaner_methods = [method for method in dir(DataCleaner)
                       if callable(getattr(DataCleaner, method)) and not method.startswith("_")]

    formatter_methods = [method for method in dir(DataFormatter)
                         if callable(getattr(DataFormatter, method)) and not method.startswith("_")]

    return {
        "DataCleaner": cleaner_methods,
        "DataFormatter": formatter_methods
    }

@router.post("/apply_operation")
def apply_operation(request: OperationRequest):
    db = DuckDBProjectManager("example.duckdb")

    try:
        df = db.conn.execute(f'SELECT * FROM "{request.table_name}"').fetchdf()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch data: {e}")

    cleaner = DataCleaner(df=df, table_name=request.table_name, db_manager=db)
    method = getattr(cleaner, request.operation, None)

    if not method:
        raise HTTPException(status_code=400, detail="Invalid cleaning operation")

    try:
        method(inplace=True, **request.parameters)
    except Exception as e:
        import traceback
        traceback.print_exc()  # ✅ Print full traceback
        raise HTTPException(status_code=500, detail=f"Operation failed: {e}")

    if request.mode == "apply":
        return {
            "status": "success",
            "message": f"{request.operation} applied temporarily",
            "data": jsonable_encoder(cleaner.df.to_dict(orient="records"))
        }

    # Save to DuckDB
    db.conn.register("temp_df", cleaner.df)
    db.conn.execute(f'CREATE OR REPLACE TABLE "{request.table_name}" AS SELECT * FROM temp_df')

    return {
        "status": "success",
        "message": f"{request.operation} applied and saved to DuckDB"
    }

