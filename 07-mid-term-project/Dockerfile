# Use official Python image
FROM python:3.12-slim

# Set working directory in the container
WORKDIR /app

# Copy requirement file first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose Flask port
EXPOSE 5000

# Run the flask app
CMD ["python", "app.py"]