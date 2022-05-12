#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask, request
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

@app.route('/api/', methods=['POST'])
def get_prediction():
    # read the real time input
    json_data = request.get_json()
    # convert json data to dictionnary
    dict_data = json.loads(json_data)
    # convert to pandas df
    data = pd.DataFrame(dict_data)
    #transform DataFrame
    data_tr = pd.DataFrame(transformer.transform(data),
                           columns=features_selected,
                           index=data.index)
    data_tr['probability'] = classifier.predict_proba(data_tr)[:, 1]
    return data_tr.to_json() 

if __name__ == '__main__':
    app.run(port=5000)