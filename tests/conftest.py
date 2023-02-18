from prediction_service.prediction import form_response, api_response
import pytest
import yaml


@pytest.fixture
def config(config_path = 'params.yaml'):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config