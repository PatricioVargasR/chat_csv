# Imagen base con Python
FROM python:3.12.1-slim

# Variables de entorno

# Evita que Python genere archivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1

# Hace que la salida de print() se muestre inmediatamente en consola (sin esperar).
ENV PYTHONUNBUFFERED=1 

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos de requisitos e instalarlos
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Exponer el puerto Flask (por defecto)
EXPOSE 5000

# Comando para iniciar la app
CMD ["python", "app.py"]
