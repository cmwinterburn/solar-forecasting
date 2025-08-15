import os
import yaml

def load_config():

    # Path to this file (config.py)
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to the config.yaml file
    config_path = os.path.join(this_dir, "config.yaml")

    # Load the config file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Determine repo root (2 levels up from config.py)
    repo_root = os.path.abspath(os.path.join(this_dir, "../../"))

    # Recursively resolve all relative paths under 'paths'
    if "paths" in config:
        for key, path in config["paths"].items():
            if isinstance(path, str) and not os.path.isabs(path):
                config["paths"][key] = os.path.normpath(os.path.join(repo_root, path))

    return config


