from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.upload import router as upload_router
from routes.analyze import router as analyze_router
from routes.report import router as report_router
from routes.dashboard import router as dashboard_router
app = FastAPI(title="Expelexia Lab Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Test route
@app.get("/")
def root():
    return {"message": "Backend is running"}

# Register upload route
app.include_router(upload_router, prefix="/api")
app.include_router(analyze_router, prefix="/api")
app.include_router(report_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")