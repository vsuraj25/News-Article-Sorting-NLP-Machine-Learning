import yaml
import json
import pickle
import numpy as np
import os
from src.utils import read_yaml


param_path = 'params.yaml'
config = read_yaml(param_path)
prediction_model = open(config['web_model_dir'], 'rb')
cv_transform_model = open(config['cv_transform_model'],'rb')
model =  pickle.load(prediction_model)
cv =  pickle.load(cv_transform_model)

## Exception Classes

class TooManyNumbers(Exception):
    def __init__(self, message = 'Text contains many numbers! Please enter a suitable text.'):
        self.message = message
        super().__init__(self.message)

class NullText(Exception):
    def __init__(self, message = 'Please enter a sufficient length of sentence!'):
        self.message = message
        super().__init__(self.message)

def __null_text(dict_req):
    text = str(list(dict_req.values())[0])
    if len(text) <= 5:
        raise NullText
    else:
        return True



def _validate_values(dict_req):
        text = str(list(dict_req.values())[0])
        num_len_counter = 0
        for i in text:
             if i.isnumeric():
                  num_len_counter += 1

        if num_len_counter >= (len(text)/2):
             raise TooManyNumbers
        else:
             return True

def form_response(dict_request):
    if __null_text(dict_request):
        data =  list(dict_request.values())
        data = str(data[0])
        print(data)
        response =  predict(data)
        return response
     
def api_response(dict_request):
    try:
        dict_request = dict(dict_request)
        data = str(list(dict_request.values())[0])
        response = predict(data)
        response = {'response' : response}
        return response
    except Exception as e:
        print(e)
        response = {'response': str(e)}
        return response
        
def predict(data):
    transformed_data =  cv.transform([data])
    prediction = model.predict(transformed_data).tolist()[0]
    if prediction ==  0:
        prediction = 'Business'
    elif prediction ==  1:
        prediction = 'Entertainment'
    elif prediction ==  2:
        prediction = 'Politics'
    elif prediction ==  3:
        prediction = 'Sport'
    elif prediction ==  4:
        prediction = 'Tech'

    return prediction
    


    