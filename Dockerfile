FROM python:3.11-slim AS base

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && GOMP=$(find /usr/local/lib -name "libgomp-*.so*" 2>/dev/null | head -1) \
    && ln -sf "$GOMP" /usr/local/lib/libgomp.so.1

# scikit-learn's vendored libgomp has a hashed SONAME baked in by auditwheel,
# so ldconfig indexes it under that hash regardless of the symlink name above --
# the cache-based lookup for "libgomp.so.1" never resolves. LD_LIBRARY_PATH
# forces a direct directory scan that matches the symlink's filename instead.
ENV LD_LIBRARY_PATH=/usr/local/lib

COPY src/ src/
COPY models/ models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
