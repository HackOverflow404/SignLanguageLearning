FROM python:3.12-slim
RUN useradd -m -u 1000 user
WORKDIR /app
RUN chown user /app
# Compilers for llama-cpp-python, which has no prebuilt wheel for ARM (the
# Oracle VM) and builds llama.cpp from source there.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --chown=user . .
USER user
ENV PYTHONPATH=/app/src PYTHONUNBUFFERED=1
# Bake the model weights into the image so a cold start doesn't download them.
RUN python -c "import llm; llm.fetch_all()"
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
