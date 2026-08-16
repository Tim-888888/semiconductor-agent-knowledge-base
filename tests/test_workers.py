from semikb.workers.celery_app import celery_app


def test_celery_app_registers_ingestion_tasks() -> None:
    celery_app.loader.import_default_modules()

    assert "semikb.ingestion.process" in celery_app.tasks
    assert "semikb.ingestion.retry" in celery_app.tasks
    assert "semikb.corpus.publish" in celery_app.tasks
