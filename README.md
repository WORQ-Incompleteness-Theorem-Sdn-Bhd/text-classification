# Text Classification Application

This repository contains a text classification application. Below are the instructions to run the application, train the model, and deploy it using Docker.

## Table of Contents
- [Running the Application](#running-the-application)
- [Training the Model](#training-the-model)
- [Classifying Text via API](#classifying-text-via-api)
- [Building and Deploying with Docker](#building-and-deploying-with-docker)

---

## Running the Application

To run the application, use the following command:

```bash
python app.py
```
## Training the Model

To train the text classification model, run the following command:

```bash
python train_model.py
```

## Training the Model
To train the text classification model, run the following command:
```bash
python train_model.py
```
## Classifying Text via API
You can classify text by sending a POST request to the /classify endpoint. Use the following curl command:
```bash
curl -X POST http://127.0.0.1:5000/classify -H "Content-Type: application/json" -d '{"text": "Thank you for being so reliable."}'
```
## Building and Deploying with Docker
Build Docker Image
To build the Docker image, run the following command:
```bash
docker build -t text-classification .
```
## Authenticate with Google Cloud Platform (GCP)
Authenticate with Google Cloud Platform (GCP)
Authenticate Docker with GCP using the following command:
```bash
gcloud auth configure-docker
```
## Tag Docker Image
Tag Docker Image
Tag the Docker image for GCP Container Registry:
```bash
docker tag text-classification gcr.io/PROJECT_ID/text-classification
```
Replace PROJECT_ID with your GCP project ID.

## Push Docker Image to GCP
Push Docker Image to GCP
Push the Docker image to GCP Container Registry:
```bash
docker push gcr.io/PROJECT_ID/text-classification
```
