import os
import logging
import json
import numpy
import joblib
from transformers import pipeline
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"),"trained_model")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    logging.info("Init complete")


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    """
    logging.info("request received")
    data = json.loads(raw_data)["inputs"]
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"),"trained_model")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    result = classifier(data)
    logging.info("Request processed")
    return result