FROM python:3.12

WORKDIR /app

RUN pip install --no-cache-dir \
     mlflow \
     boto3 \
     psycopg2

EXPOSE 8085

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "8085", \
     "--default-artifact-root", "/app/mlruns"]
