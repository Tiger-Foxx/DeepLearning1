# Utiliser une image de base Python légère
FROM python:3.9-slim

# Variables d'environnement utiles
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

WORKDIR /app

# Copier requirements et installer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application (app.py + mnist_model.h5 attendu ici)
COPY . .

# Exposer le port
EXPOSE 5000

# Commande par défaut : Gunicorn (meilleur pour la prod)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--workers", "2", "--threads", "4"]
