#! /usr/bin/bash
docker build . --tag gcr.io/marc-385119/marc-api:latest
docker push gcr.io/marc-385119/marc-api:latest

gcloud init
gcloud config set run/region europe-west9
gcloud run deploy --image gcr.io/marc-385119/marc-api:latest --cpu 6 --concurrency 1 --memory 16Gi --platform managed --min-instances 0 --timeout 2m --port 80