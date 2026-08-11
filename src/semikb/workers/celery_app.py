"""Celery configuration. Redis is required outside demo mode."""

from celery import Celery

from semikb.config import get_settings

settings = get_settings()
celery_app = Celery("semikb", include=["semikb.workers.tasks"])
celery_app.conf.update(
    broker_url=settings.redis_url or "memory://",
    result_backend=settings.redis_url or "cache+memory://",
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    task_time_limit=1800,
    task_soft_time_limit=1500,
    worker_prefetch_multiplier=1,
)
