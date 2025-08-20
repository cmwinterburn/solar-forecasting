import yaml

def load_config():
    """Loads yaml configuration containing necessary filepaths for data pipeline execution"""

    with open("/solar-forecasting/app/config/config.yaml", "r") as f:
        return yaml.safe_load(f)


