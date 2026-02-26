# Template training container - connects to existing llama.cpp server
FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    jinja2 \
    openai \
    pyyaml \
    rich \
    httpx

# Copy the training framework
COPY template_trainer/ /app/template_trainer/
COPY templates/ /app/templates/
COPY tests/ /app/tests/
COPY run.py /app/

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "run.py"]
