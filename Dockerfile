FROM python:3.11-slim

# libgomp1 -- OpenMP runtime LightGBM's compiled extension needs at import time;
# not present in the slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

CMD ["python", "run_pipeline.py"]
