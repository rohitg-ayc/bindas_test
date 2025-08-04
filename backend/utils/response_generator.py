
from fastapi.responses import JSONResponse

def success_response(message: str, data=None, status_code: int = 200):
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "success",
            "message": message,
            "data": data or [],
        },
    )


def error_response(message: str, data=None, status_code: int = 500):
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "message": message,
            "data": data or [],
        },
    )

from fastapi.responses import JSONResponse
from typing import Optional, Any
from backend.models.schemas import StandardResponse

def standard_json_response(success: bool,
                            message: Optional[str] = None,
                            data: Optional[Any] = None,
                            metadata: Optional[Any] = None,
                            status_code: int = 200
                        ) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=StandardResponse(
            success=success,
            message=message,
            data=data,
            metadata=metadata
        ).dict()
    )
