import yaml

def read_yaml(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config