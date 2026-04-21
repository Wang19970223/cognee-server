"""
app/celery_app.py
─────────────────────────────────────────────────────────────────────────────
Celery application instance.

The worker is started separately:
    celery -A app.celery_app worker --loglevel=info -P threads

We use the 'threads' pool because Cognee relies heavily on asyncio and the
default 'prefork' pool forks processes which can conflict with event loops.
"""

from celery import Celery

from app.config import get_settings


def create_celery() -> Celery:
    settings = get_settings()
    app = Celery(
        "cognee_service",
        broker=settings.redis_url,
        backend=settings.redis_url,
        include=["app.tasks.index_tasks"],
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # Keep results for 24 hours
        result_expires=86400,
        # Acknowledge after the task completes (not on receipt) so a crash
        # doesn't silently drop the task.
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        # Generous time limits — cognify can take 10-30 minutes for large docs
        task_soft_time_limit=3600,  # 60 min: raises SoftTimeLimitExceeded
        task_time_limit=4200,       # 70 min: hard kill
        # Keep the Redis broker/backend connection alive during long tasks.
        # The cognify step can take 10+ minutes, during which the connection
        # would otherwise be closed by the Redis server's idle timeout.
        broker_transport_options={
            "socket_keepalive": True,
            "retry_on_timeout": True,
            "socket_connect_timeout": 30,
            "socket_timeout": 30,
        },
        result_backend_transport_options={
            "socket_keepalive": True,
            "retry_on_timeout": True,
        },
        broker_connection_retry_on_startup=True,
        broker_connection_max_retries=10,
    )
    return app


celery_app = create_celery()
