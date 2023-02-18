import pytest
import logging
import os
import joblib
from prediction_service.prediction import form_response, api_response
import prediction_service

input_data = {
    "correct_input":
    {
    "txt": "Hour ago, I contemplated retirement for a lot of reasons. I felt like people were not sensitive enough to my injuries. I felt like a lot of people were backed, why not me? I have done no less. I have won a lot of games for the team, and I am not feeling backed, said Ashwin"
    },

    "number_input":
    {
    "txt": "893js34"
    },

    "null_input":
    {
    "txt": ""
    }
}

result_category = ['Buisness', 'Entertainment','Politics', 'Sport' ,'Tech']

def test_form_response_correct_input(data = input_data["correct_input"]):
    res = form_response(data)
    assert res in result_category 

def test_from_response_number_error(data = input_data["number_input"]):
    with pytest.raises(prediction_service.prediction.TooManyNumbers):
        res = form_response(data)

def test_from_response_null_input(data = input_data["null_input"]):
    with pytest.raises(prediction_service.prediction.NullText):
        res = form_response(data)

def test_api_response_correct_input(data = input_data["correct_input"]):
    res = api_response(data)
    assert res['response'] in result_category 

def test_api_response_number_error(data = input_data["number_input"]):
    res =  api_response(data)
    assert res['response'] == prediction_service.prediction.TooManyNumbers().message

def test_api_response_null_input(data = input_data["null_input"]):
    res =  api_response(data)
    assert res['response'] == prediction_service.prediction.NullText().message