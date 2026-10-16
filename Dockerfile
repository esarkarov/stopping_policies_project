FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies first so Docker can cache this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code
COPY configs/ configs/
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/

ENV PYTHONUNBUFFERED=1

CMD ["python", "scripts/speed_test.py"]
