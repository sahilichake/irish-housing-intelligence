FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Build data + models at image build time (set --build-arg SAMPLE=1 to use synthetic data)
ARG SAMPLE=0
RUN if [ "$SAMPLE" = "1" ]; then python -m src.pipeline --sample; else python -m src.pipeline; fi

EXPOSE 8501 8000
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
