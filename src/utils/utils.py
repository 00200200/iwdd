import yaml


def load_models_config(config_path: str = "config/models_config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def list_available_models():
    config = load_models_config()
    return list(config["models"].keys())


def get_model_config(model_name):
    config = load_models_config()
    return config["models"][model_name]
