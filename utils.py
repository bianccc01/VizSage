import os
import json
import yaml
import sys


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


def prepare_streaming_dataset(streaming_dataset, config):
    """
    Prepare the streaming dataset for training.

    Args:
        streaming_dataset (IterableDataset): Streaming dataset to be processed
        config (dict): Configuration dictionary containing training parameters

    Returns:
        Dataset: Processed dataset ready for training
    """
    if streaming_dataset is None:
        return None

    # Apply limit to the dataset if specified
    buffer_size = config.get("stream_buffer_size", 1000)
    dataset = streaming_dataset.shuffle(buffer_size=buffer_size)

    # Convert every example to a conversation format
    def convert_to_conversation_streaming(example):
        return dp.convert_to_conversation(example)

    # Apply the conversion function to the streaming dataset
    processed_dataset = dataset.map(convert_to_conversation_streaming)

    return processed_dataset