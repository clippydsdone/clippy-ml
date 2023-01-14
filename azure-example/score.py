import os
from transformers import pipeline, AutoTokenizer, TFBartForConditionalGeneration
import logging
import json

def init():
    global summarizer

    # AZUREML_MODEL_DIR is an environment variable created during deployment
    model_path = os.path.join(os.environ["AZUREML_MODEL_DIR"], "model")

    # load the model
    tokenizer = AutoTokenizer.from_pretrained(model_path, truncation=True, max_length=1024)

    model = TFBartForConditionalGeneration.from_pretrained(model_path)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, num_beams=5, do_sample=True, no_repeat_ngram_size=3)


def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    text = json.loads(raw_data)["text"]
    return summarizer(text, truncation=True)