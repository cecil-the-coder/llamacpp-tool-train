# Template training container for llama.cpp function calling
# Systematically improves Jinja chat templates for tool calling

FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive

# Install build dependencies for llama.cpp
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git wget curl \
        libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone and build llama.cpp with server support
ARG LLAMACPP_REF=b5399
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git \
    && cd llama.cpp \
    && git fetch --depth 1 origin ${LLAMACPP_REF} \
    && git checkout ${LLAMACPP_REF} \
    && make llama-server llama-cli -j$(nproc)

# Install Python dependencies for template testing
RUN pip install --no-cache-dir \
    jinja2 \
    openai \
    pyyaml \
    rich \
    pytest \
    pytest-asyncio \
    httpx

# Copy the training framework
COPY template_trainer/ /app/template_trainer/
COPY templates/ /app/templates/
COPY tests/ /app/tests/
COPY run.py /app/

# Create directories for models and results
RUN mkdir -p /models /results

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "run.py"]
