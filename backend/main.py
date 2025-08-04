# from fastapi import FastAPI
# from backend.core.config import SESSION_KEY
# from starlette.middleware.sessions import SessionMiddleware
# from backend.routes import import_routes, cleaning_routes, relationship_routes, ml_routes
# from backend.routes import auth_routes as auth_router

# app = FastAPI()

# app.add_middleware(
#     SessionMiddleware,
#     secret_key=SESSION_KEY,
#     https_only=False
# )

# app.include_router(auth_router, prefix="/auth")
# app.include_router(import_routes.router, prefix="/data_import")
# app.include_router(cleaning_routes.router, prefix="/data_cleaning")
# app.include_router(relationship_routes.router, prefix="/relation_detection")
# app.include_router(ml_routes.router, prefix="/column_tagging")

# @app.get("/")
# def health_check():
#     return {"status": "BINDAS API running"}


from fastapi import FastAPI
from backend.core.config import SESSION_KEY
from starlette.middleware.sessions import SessionMiddleware

# Import routers
from backend.routes.import_routes import router as import_router
from backend.routes.cleaning_routes import router as cleaning_router
from backend.routes.relationship_routes import router as relationship_router
from backend.routes.ml_routes import router as ml_router
from backend.routes.auth_routes import router as auth_router  

app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_KEY,
    https_only=False
)

# Register routers
app.include_router(auth_router, prefix="/auth")
app.include_router(import_router, prefix="/data_import")
app.include_router(cleaning_router, prefix="/data_cleaning")
app.include_router(relationship_router, prefix="/relation_detection")
app.include_router(ml_router, prefix="/column_tagging")

@app.get("/")
def health_check():
    return {"status": "BINDAS API running"}
