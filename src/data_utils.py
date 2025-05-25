import os
import json
from PIL import Image
from datasets import Dataset, IterableDataset



def extract_image(image_name, base_path="data"):
    """Extract and load an image from the specified path."""
    images_path = os.path.join(base_path, "Images")
    image_path = os.path.join(images_path, image_name)
    image = Image.open(image_path).convert("RGB")
    return image


def get_dataset(base_path="data", dataset="AQUA", external_knowledge=False, use_streaming=False):
    """
    Get the dataset from the base path with optional streaming support.

    Args:
        base_path (str): Base path for dataset
        dataset (str): Dataset name
        external_knowledge (bool): Whether to include external knowledge samples
        use_streaming (bool): Whether to use streaming mode for large datasets

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, train_size[, test_size])
    """
    dataset_path = os.path.join(base_path, dataset)
    train_data = None
    val_data = None
    test_data = None

    # First load the dataset files
    for file in os.listdir(dataset_path):
        if file.endswith("train.json"):
            with open(os.path.join(dataset_path, file), 'r') as f:
                train_data = json.load(f)
        elif file.endswith("val.json"):
            with open(os.path.join(dataset_path, file), 'r') as f:
                val_data = json.load(f)
        elif file.endswith("test.json"):
            with open(os.path.join(dataset_path, file), 'r') as f:
                test_data = json.load(f)
        else:
            continue

    # Filter the dataset based on external knowledge requirement
    if not external_knowledge:
        train_data = [sample for sample in train_data if not sample.get("need_external_knowledge", False)]
        if val_data:
            val_data = [sample for sample in val_data if not sample.get("need_external_knowledge", False)]
        if test_data:
            test_data = [sample for sample in test_data if not sample.get("need_external_knowledge", False)]

    # If not using streaming, return the datasets as normal
    if not use_streaming:
        return train_data, val_data, test_data, len(train_data)

    # If using streaming, convert the datasets to streaming format
    else:
        print(f"Converting dataset {dataset} to streaming format...")

        # Helper function to convert a list of data to a streaming dataset
        def convert_to_streaming_dataset(data_list):
            if not data_list:
                return None

            # edit generator to yield examples
            def gen_examples():
                for example in data_list:
                    yield example

            # Create a streaming dataset from the generator
            streaming_dataset = IterableDataset.from_generator(gen_examples)

            return streaming_dataset

        print(f"Training dataset size: {len(train_data)}")
        len_train_data = len(train_data)
        len_test_data = len(test_data) if test_data else 0
        train_dataset = convert_to_streaming_dataset(train_data) if train_data else None
        val_dataset = convert_to_streaming_dataset(val_data) if val_data else None
        test_dataset = convert_to_streaming_dataset(test_data) if test_data else None

        print(f"Successfully created streaming datasets")
        return train_dataset, val_dataset, test_dataset, len_train_data, len_test_data


def convert_to_conversation(sample, semart_dataset=None, is_test=False, base_path="data"):
    """Convert a sample to a conversation format for model training."""
    instruction = "You are an expert art historian. Answer the questions you will be asked about the image."

    if sample["need_external_knowledge"]:
        description = semart_dataset.loc[semart_dataset['image_file'] == sample["image"], 'description'].values[0]
        conversation = [
            { "role": "user",
              "content" : [
                  {"type" : "text",  "text"  : instruction},
                  {"type" : "image", "image" : extract_image(sample["image"], base_path=base_path)},
                  {"type" : "text",  "text"  : description},
                  {"type" : "text",  "text"  : sample["question"]}
              ]
              }
        ]
        if not is_test:
            conversation.append(
                { "role" : "assistant",
                  "content" : [
                      {"type" : "text",  "text"  : sample["answer"]} ]
                  }
            )
    else:
        conversation = [
            { "role": "user",
              "content" : [
                  {"type" : "text",  "text"  : instruction},
                  {"type" : "image", "image" : extract_image(sample["image"], base_path=base_path)},
                  {"type" : "text",  "text"  : sample["question"]} ]
              }
        ]
        if not is_test:
            conversation.append(
                { "role" : "assistant",
                  "content" : [
                      {"type" : "text",  "text"  : sample["answer"]} ]
                  }
            )
    return { "messages" : conversation }


def prepare_streaming_dataset(streaming_dataset, config, semart_dataset=None, base_path="data"):
    """
    Prepare the streaming dataset for training.

    Args:
        streaming_dataset (IterableDataset): Streaming dataset to be processed
        config (dict): Configuration dictionary containing training parameters
        semart_dataset: Dataset containing semantic art descriptions
        base_path (str): Base path for dataset files

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
        return convert_to_conversation(example, semart_dataset=semart_dataset, base_path=base_path)

    # Apply the conversion function to the streaming dataset
    processed_dataset = dataset.map(convert_to_conversation_streaming)

    return processed_dataset



