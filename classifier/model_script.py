#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn import metrics

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import precision_recall_fscore_support
from io import StringIO

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.calibration import CalibratedClassifierCV

from sklearn.svm import SVC

import json

# from sklearn.externals import joblib

import joblib

from sklearn.pipeline import Pipeline

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )

    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])

    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    args = parser.parse_args()

    training_file = os.path.join(args.train, "input_train.csv")

    df_input = pd.read_csv(training_file, engine="python")

    df_input.columns = ["document_label", "document_text"]
    df_input = df_input[pd.notnull(df_input["document_text"])]

    df_input.shape
    X = df_input["document_text"]
    y = df_input["document_label"]

    X.describe()

    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=1000)
    vector_content = vectorizer.fit_transform(X.apply(lambda x: np.str_(x)))

    (X_train, X_test, y_train, y_test) = train_test_split(
        vector_content, y, test_size=0.1, random_state=1
    )

    (X_train, X_val, y_train, y_val) = train_test_split(
        X_train, y_train, test_size=0.10, random_state=1
    )
    
    
    # calibrated classifier to get the confidence score (probability)
    base_model = SGDClassifier(max_iter=1000, tol=1e-3)

    sgd_model = CalibratedClassifierCV(base_model)

    sgd_model.fit(X_train, y_train)

    joblib.dump(sgd_model, os.path.join(args.model_dir, "model.joblib"))


def input_fn(input_data, content_type):
    if content_type == "text/csv":
        # Read the raw input data as CSV.
        df_input = pd.read_csv(StringIO(input_data), header=None)
        df_input.columns = ["document_label", "document_text"]
        df_input = df_input[pd.notnull(df_input["document_text"])]

        X_ = df_input["document_text"]
        y_ = df_input["document_label"]

        print("input data")
        vectorize = CountVectorizer(ngram_range=(1, 2), max_features=1000)
        vec_content = vectorize.fit_transform(X_.apply(lambda x: np.str_(x)))
        print("vector_content")
        print(vec_content.shape)
        return vec_content

    else:

        raise ValueError("{} not supported by script!".format(content_type))


def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    pred_prob = model.predict_proba(input_data)
    pred_proba = pred_prob.ravel()
    return np.array([prediction, pred_proba], dtype="object")


def output_fn(prediction, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []

        print("predicted")
        predicted = prediction[0].tolist()
     
        print("confidence")
        confidence = prediction[1].tolist()

        for row in zip(predicted, confidence):
            instances.append({"prediction": row[0], "confidence": row[1]})

        json_output = {"instances": instances}
        json_data = json.dumps(json_output)
        print(json_data)

        return json_data

    else:
        raise ValueError("{} not supported by script!".format(accept))


def model_fn(model_dir):

    sgd_model = joblib.load(os.path.join(model_dir, "model.joblib"))
    return sgd_model
