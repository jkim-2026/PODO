from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os

# Import routers and database
from routers import detect, stats
from database.db import init_db

# Create images directory if it doesn't exist
os.makedirs("images/defects", exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize database
    await init_db()
    yield
    # Shutdown: (cleanup if needed)


app = FastAPI(
    title="PCB Defect Detection API",
    description="Backend for Jetson Edge AI and Streamlit Frontend",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for images
app.mount("/images", StaticFiles(directory="images"), name="images")

# Apply Routers
app.include_router(detect.router)
app.include_router(stats.router)

@app.get("/")
async def root():
    return {"message": "PCB Defect Detection API is running. Check /docs for API documentation."}
