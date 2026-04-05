# Use official Python 3.11 image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files into container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic pandas requests

# Expose port 8000
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
