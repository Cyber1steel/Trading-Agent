from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    description="Foundation for an AI trading research platform.",
)


@app.get("/", tags=["root"])
def read_root() -> dict[str, str]:
    return {"name": settings.app_name, "status": "running"}


app.include_router(health_router)
