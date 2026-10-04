FROM python:3.12-slim

WORKDIR /app

# CPU build of PyTorch, installed first so that pip does not pull the CUDA libraries
# (several gigabytes): the bot only computes query embeddings. If the PyTorch index is
# unreachable (for example, behind a TLS-intercepting proxy), the default build from PyPI
# is installed by the next command instead - the image works the same, but is much larger.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    || echo "PyTorch CPU index is unreachable, falling back to PyPI"
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_bot/ rag_bot/
COPY scripts/ scripts/
COPY knowledge_base/ knowledge_base/

# The embedding model and the FAISS index are mounted as volumes (see docker-compose.yml).
ENV HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "rag_bot"]
