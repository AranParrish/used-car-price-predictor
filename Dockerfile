FROM python:3.12-slim

WORKDIR /app

# Copy requirements.txt into the image
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY . .

# Default command
CMD ["python"]