# ---- FIFA WC 2026 dashboard: container image -------------------------------
# Small, reproducible image that serves the Streamlit dashboard.
# Only the slim dashboard requirements are installed (no ML stack needed to
# render the pre-computed forecast CSVs).

FROM python:3.12-slim

WORKDIR /app

# install dependencies first so this layer caches across code changes
COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt

# copy the rest of the project (dashboard code + reports/*.csv)
COPY . .

# Streamlit serves on a port; on Render/Railway/etc. the platform injects $PORT
ENV PORT=8501
EXPOSE 8501

# shell form so ${PORT} expands at runtime
CMD streamlit run dashboard/app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true
