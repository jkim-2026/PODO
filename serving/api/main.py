# CI/CD 테스트
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Import routers, database, and config
from routers import detect, stats, sessions, monitoring
from database.db import init_db
from database import db
from config import settings
from utils.slack_notifier import send_slack_alert

# 스케줄러 전역 변수
scheduler = AsyncIOScheduler()


async def check_and_send_alerts():
    """
    주기적으로 health 체크하고 Slack 알림 전송
    """
    try:
        health_data = await db.get_health("latest")

        if health_data["status"] == "critical":
            await send_slack_alert(
                status=health_data["status"],
                alerts=health_data["alerts"],
                session_info=health_data["session_info"]
            )
    except Exception as e:
        print(f"알림 체크 실패: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Create directories and initialize database
    settings.IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    await init_db()

    # 스케줄러 시작
    scheduler.add_job(
        check_and_send_alerts,
        'interval',
        minutes=settings.SLACK_CHECK_INTERVAL_MINUTES,
        id='slack_alert_check'
    )
    scheduler.start()

    yield

    # Shutdown
    scheduler.shutdown()
    await db.close_db()


app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.CORS_ALLOW_METHODS,
    allow_headers=settings.CORS_ALLOW_HEADERS,
)

# Mount static files for images
app.mount("/images", StaticFiles(directory="images"), name="images")

# Apply Routers
app.include_router(detect.router)
app.include_router(stats.router)
app.include_router(sessions.router)
app.include_router(monitoring.router)


@app.get("/")
async def root():
    return {"message": f"{settings.API_TITLE} is running. Check /docs for API documentation."}
