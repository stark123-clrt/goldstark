FROM python:3.10-slim

# Installation des dépendances système nécessaires pour TensorFlow/TA-Lib si besoin
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# On expose le port 10000 (standard pour Render) au lieu de 7860
EXPOSE 10000

# On lance le script directement car tu as ajouté le bloc uvicorn.run dans app.py
CMD ["python", "app.py"]