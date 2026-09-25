FROM python:3.11-slim

WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY server/requirements.txt /srv/server/requirements.txt
RUN pip install --no-cache-dir -r /srv/server/requirements.txt

COPY server /srv/server
COPY ml /srv/ml

ENV PYTHONPATH=/srv/server:/srv
ENV JWT_SECRET=change-me-in-production
EXPOSE 5674
WORKDIR /srv/server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5674"]
