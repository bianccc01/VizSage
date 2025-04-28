from random import sample

from PIL import Image
import os
import json


def extract_image(image_name, base_path="data"):
    images_path = os.path.join(base_path, "Images")
    image_path = os.path.join(images_path, image_name)
    image = Image.open(image_path).convert("RGB")
    return image


def get_dataset(base_path="data", dataset="AQUA", external_knowledge=False):
    """
    Get the dataset from the base path.
    """
    dataset_path = os.path.join(base_path, dataset)
    for file in os.listdir(dataset_path):
        if file.endswith("train.json"):
            with open(os.path.join(dataset_path, file), 'r') as f:
                train_dataset = json.load(f)
        elif file.endswith("val.json"):
            with open(os.path.join(dataset_path, file), 'r') as f:
                val_dataset = json.load(f)
        elif file.endswith("test.json"):
            with open(os.path.join(dataset_path, file), 'r') as f:
                test_dataset = json.load(f)
        else:
            continue

    if not external_knowledge:
        train_dataset = [sample for sample in train_dataset if not sample["need_external_knowledge"]]
        val_dataset = [sample for sample in val_dataset if not sample["need_external_knowledge"]]
        test_dataset = [sample for sample in test_dataset if not sample["need_external_knowledge"]]

    return train_dataset, val_dataset, test_dataset


def convert_to_conversation(sample):
    instruction = "You are an expert art historian. Answer the questions you will be asked about the image."

    if sample["need_external_knowledge"]:
        conversation = [
            { "role": "user",
              "content" : [
                {"type" : "text",  "text"  : instruction},
                {"type" : "text",  "text"  : sample["question"]},
                {"type" : "image", "image" : extract_image(sample["image"])},
                #TODO: METTERE DESCRIZIONE QUADRO
                {"type" : "text",  "text"  : sample["external_knowledge"]} ]
            },
            { "role" : "assistant",
              "content" : [
                {"type" : "text",  "text"  : sample["answer"]} ]
            },
        ]

    else:
        conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "text",  "text"  : sample["question"]},
            {"type" : "image", "image" : extract_image(sample["image"])} ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["answer"]} ]
        },
    ]
    return { "messages" : conversation }
pass