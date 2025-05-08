import os
import sys
import yaml


def load_config(config_file="config.yaml"):
    """Load training configuration from YAML file"""
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    # Ensure the config file exists
    if not os.path.exists(config_file):
        print(f"Error: Config file '{config_file}' not found.")
        sys.exit(1)

    # Load the configuration file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from {config_file}")

    return config