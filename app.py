#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import json

# read the model pickle file
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
transformer = model['transformer']
features_selected = model['features_selected']
classifier = model['classifier']

app = Flask(__name__)

@app.route('/')
def hello():
    # A welcome message to test the api
    return "<h1>Welcome to my prediction api!</h1>"

@app.route('/api/', methods=['POST'])
def get_prediction():
    # read the real time input
    json_data = request.get_json()
    # convert json data to dictionnary
    dict_data = json.loads(json_data)
    # convert to pandas df
    data = pd.DataFrame(dict_data)
    # get predictions
    data_tr = pd.DataFrame(transformer.transform(data),
                           columns=features_selected,
                           index=data.index)
    data_tr['probability'] = classifier.predict_proba(data_tr)[:, 1]
    return jsonify(data_tr.to_dict())

if __name__ == '__main__':
    app.run(port=5000)