#!/bin/bash

SERVICE_NAME="deeplearning"

echo "Running the container for service '$SERVICE_NAME'..."
docker compose up -d $SERVICE_NAME
docker compose exec $SERVICE_NAME bash
docker compose down -v

IMAGE_ID=$(docker images --filter=reference='*deeplearning:latest' -q)
if [ -n "$IMAGE_ID" ]; then
  echo "Removing the image with ID: $IMAGE_ID"
  docker rmi $IMAGE_ID
else
  echo "No image found for service '$SERVICE_NAME'."
fi
