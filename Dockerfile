# Use slim python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_URI="models:/fraud-detection-v2@production"

WORKDIR /app

# Create a non-root user
RUN addgroup --system app && adduser --system --group app

# Install dependencies (Layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R app:app /app

# Switch to non-root user
USER app

EXPOSE 8000

# Run with gunicorn for production (via uvicorn worker)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]