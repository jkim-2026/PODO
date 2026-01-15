from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os

# Import routers
from routers import detect, stats

# Create images directory if it doesn't exist
os.makedirs("images/defects", exist_ok=True)

app = FastAPI(
    title="PCB Defect Detection API",
    description="Backend for Jetson Edge AI and Streamlit Frontend",
    version="1.0.0"
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
