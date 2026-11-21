# Multi-stage build: Node.js 20 is bundled alongside Python 3.11 so denpex_local
# runs the deterministic canonical offline engine without network egress.
FROM node:20.20.2-bookworm-slim@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0 AS node-runtime
FROM python:3.11.15-slim-bookworm@sha256:d29f48a31a8b408ed19272ca1e7b10ebae13b240a27e862d3d4217c528e2e0c3

COPY --from=node-runtime /usr/local/bin/node /usr/local/bin/node

LABEL org.opencontainers.image.source="https://github.com/Denpex/denpex-runpod" \
      org.opencontainers.image.url="https://denpex.com/docs/integrations" \
      org.opencontainers.image.version="1.1.36" \
      org.opencontainers.image.description="Denpex GPU crash, CUDA OOM, NCCL, and NVIDIA Xid diagnostics serverless worker for RunPod Hub." \
      org.opencontainers.image.licenses="MIT"

ENV DENPEX_LOCAL=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir denpex==1.1.36 "runpod>=1.7,<2"

WORKDIR /app
COPY handler.py /app/handler.py

CMD ["python3", "-u", "/app/handler.py"]
