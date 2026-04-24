setup:
	pip install -r requirements.txt

pipeline:
	python scripts/run_pipeline.py

backend:
	uvicorn backend.app.main:app --reload

test:
	pytest
