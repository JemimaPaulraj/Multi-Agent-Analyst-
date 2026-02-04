FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /opt/program

# System deps (often needed for prophet/cmdstanpy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY serve .

# Fix Windows line endings (CRLF -> LF) and make executable
RUN sed -i 's/\r$//' /opt/program/serve && chmod +x /opt/program/serve

# Add program directory to PATH so SageMaker can find 'serve' command
ENV PATH="/opt/program:${PATH}"

EXPOSE 8080

# No ENTRYPOINT - SageMaker will run: docker run <image> serve
# which executes /opt/program/serve (since it's in PATH)
