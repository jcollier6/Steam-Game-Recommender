# Use an official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy gatherer code
COPY gather-data.py .

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Default command (can override when running)
CMD ["python", "gather-data.py"]
