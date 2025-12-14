FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files from the repo root (build context must be repo root)
COPY envs/kpi_recommender_system/pyproject.toml .
COPY envs/kpi_recommender_system/uv.lock .

# Install dependencies into /app/.venv
RUN uv sync --frozen

# Copy the project code into the container
COPY projects/kpi_recommender_system .

EXPOSE 8501

# Run Streamlit from the uv-managed virtual environment
CMD ["/app/.venv/bin/streamlit", "run", "app/streamlit_kpi_app.py", "--server.port=8501", "--server.address=0.0.0.0"]