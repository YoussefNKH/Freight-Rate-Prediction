FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

COPY requirements.txt .
RUN python -c "from pathlib import Path; p=Path('requirements.txt'); raw=p.read_bytes(); text=raw.decode('utf-16') if raw.startswith((b'\\xff\\xfe', b'\\xfe\\xff')) else raw.decode('utf-8'); Path('requirements-docker.txt').write_text(text, encoding='utf-8')" \
    && python -m pip install --no-cache-dir -r requirements-docker.txt \
    && rm requirements.txt requirements-docker.txt

COPY app.py .
COPY models ./models

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
