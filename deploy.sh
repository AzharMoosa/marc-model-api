#! /usr/bin/bash
docker build . --tag gcr.io/marc-385119/marc-api:latest
docker push gcr.io/marc-385119/marc-api:latest