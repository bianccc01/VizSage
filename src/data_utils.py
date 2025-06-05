"""
Enhanced data utilities with better error handling and validation
"""

import os
import json
from PIL import Image
from datasets import Dataset, IterableDataset
from typing import Optional, Tuple, List, Dict, Any, List
import pandas as pd


def extract_image(image_name: str, base_path: str = "data") -> Image.Image:
    """
    Extract and load an image from the specified path with error handling

    Args:
        image_name: Name of the image file
        base_path: Base directory path

    Returns:
        PIL Image object in RGB format

    Raises:
        FileNotFoundError: If image file doesn't exist
        Exception: If image cannot be loaded
    """
    images_path = os.path.join(base_path, "Images")
    image_path = os.path.join(images_path, image_name)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        raise Exception(f"Error loading image {image_path}: {e}")


def validate_dataset_structure(dataset_path: str) -> Tuple[bool, List[str]]:
    """
    Validate dataset directory structure

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    if not os.path.exists(dataset_path):
        errors.append(f"Dataset directory does not exist: {dataset_path}")
        return False, errors

    # Check for required files
    required_files = ["train.json"]
    optional_files = ["val.json", "test.json"]

    found_files = []
    for file in os.listdir(dataset_path):
        if file.endswith(".json"):
            found_files.append(file)

    # Check if at least train.json exists
    if not any(f.endswith("train.json") for f in found_files):
        errors.append("No train.json file found in dataset directory")

    # Validate JSON files
    for file in found_files:
        file_path = os.path.join(dataset_path, file)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    errors.append(f"{file} should contain a list of samples")
                elif len(data) == 0:
                    errors.append(f"{file} is empty")
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in {file}: {e}")
        except Exception as e:
            errors.append(f"Error reading {file}: {e}")

    return len(errors) == 0, errors


def load_dataset_files(dataset_path: str) -> Tuple[Optional[List], Optional[List], Optional[List]]:
    """
    Load dataset files from directory

    Args:
        dataset_path: Path to dataset directory

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    train_data = None
    val_data = None
    test_data = None

    try:
        for file in os.listdir(dataset_path):
            file_path = os.path.join(dataset_path, file)

            if file.endswith("train.json"):
                with open(file_path, 'r') as f:
                    train_data = json.load(f)
                    print(f"Loaded training data: {len(train_data)} samples")

            elif file.endswith("val.json"):
                with open(file_path, 'r') as f:
                    val_data = json.load(f)
                    print(f"Loaded validation data: {len(val_data)} samples")

            elif file.endswith("test.json"):
                with open(file_path, 'r') as f:
                    test_data = json.load(f)
                    print(f"Loaded test data: {len(test_data)} samples")

    except Exception as e:
        print(f"Error loading dataset files: {e}")
        raise

    return train_data, val_data, test_data


def filter_external_knowledge_samples(data: List[Dict], include_external: bool = False) -> List[Dict]:
    """
    Filter samples based on external knowledge requirement

    Args:
        data: List of data samples
        include_external: Whether to include samples requiring external knowledge

    Returns:
        Filtered list of samples
    """
    if data is None:
        return None

    if include_external:
        return data
    else:
        filtered = [sample for sample in data if not sample.get("need_external_knowledge", False)]
        print(f"Filtered {len(data) - len(filtered)} samples requiring external knowledge")
        return filtered


def get_dataset_with_separate_streaming(
        base_path: str = "data",
        dataset: str = "AQUA",
        external_knowledge: bool = False,
        use_train_streaming: bool = False,
        use_val_streaming: bool = False
) -> Tuple:
    """
    Get the dataset with separate streaming control for train and validation datasets.

    Args:
        base_path: Base path for dataset
        dataset: Dataset name
        external_knowledge: Whether to include external knowledge samples
        use_train_streaming: Whether to use streaming mode for training dataset
        use_val_streaming: Whether to use streaming mode for validation dataset

    Returns:
        (train_dataset, val_dataset, test_dataset, train_size, test_size)
    """
    dataset_path = os.path.join(base_path, dataset)

    # Validate dataset structure
    is_valid, errors = validate_dataset_structure(dataset_path)
    if not is_valid:
        print("❌ Dataset validation failed:")
        for error in errors:
            print(f"  • {error}")
        raise ValueError("Invalid dataset structure")

    # Load dataset files
    train_data, val_data, test_data = load_dataset_files(dataset_path)

    if train_data is None:
        raise ValueError("No training data found")

    # Filter based on external knowledge requirement
    train_data = filter_external_knowledge_samples(train_data, external_knowledge)
    val_data = filter_external_knowledge_samples(val_data, external_knowledge)
    test_data = filter_external_knowledge_samples(test_data, external_knowledge)

    # Get sizes before conversion
    len_train_data = len(train_data) if train_data else 0
    len_test_data = len(test_data) if test_data else 0

    # Convert datasets based on streaming preferences
    processed_train_dataset = _convert_dataset_if_streaming(train_data, use_train_streaming, "training")
    processed_val_dataset = _convert_dataset_if_streaming(val_data, use_val_streaming, "validation")

    # Test dataset is typically not streamed for inference purposes
    processed_test_dataset = test_data  # Keep as regular list for easier random access

    print(f"\n📊 Dataset Configuration:")
    print(f"  • Training: {'streaming' if use_train_streaming else 'regular'} ({len_train_data} samples)")
    print(f"  • Validation: {'streaming' if use_val_streaming else 'regular'} ({len(val_data) if val_data else 0} samples)")
    print(f"  • Test: regular ({len_test_data} samples)")

    return processed_train_dataset, processed_val_dataset, processed_test_dataset, len_train_data, len_test_data


def _convert_dataset_if_streaming(data: Optional[List], use_streaming: bool, dataset_type: str) -> Optional:
    """
    Convert dataset to streaming format if requested, otherwise return as-is

    Args:
        data: Data list to potentially convert
        use_streaming: Whether to convert to streaming
        dataset_type: Type of dataset (for logging)

    Returns:
        IterableDataset if streaming requested, otherwise original data list
    """
    if not data:
        return None

    if use_streaming:
        print(f"Converting {dataset_type} dataset to streaming format...")

        def gen_examples():
            for example in data:
                yield example

        streaming_dataset = IterableDataset.from_generator(gen_examples)
        print(f"Successfully created streaming {dataset_type} dataset")
        return streaming_dataset
    else:
        print(f"Using regular format for {dataset_type} dataset")
        return data


def get_dataset(
        base_path: str = "data",
        dataset: str = "AQUA",
        external_knowledge: bool = False,
        use_streaming: bool = False,
        use_streaming_val_dataset: bool = None
) -> Tuple:
    """
    Get the dataset from the base path with optional streaming support.
    This function maintains backward compatibility while supporting the new separate streaming feature.

    Args:
        base_path: Base path for dataset
        dataset: Dataset name
        external_knowledge: Whether to include external knowledge samples
        use_streaming: Whether to use streaming mode for training dataset
        use_streaming_val_dataset: Whether to use streaming mode for validation dataset
                                  (if None, uses same value as use_streaming for backward compatibility)

    Returns:
        For streaming: (train_dataset, val_dataset, test_dataset, train_size, test_size)
        For regular: (train_dataset, val_dataset, test_dataset, train_size)
    """
    # Handle backward compatibility
    if use_streaming_val_dataset is None:
        use_streaming_val_dataset = use_streaming

    # Use the new function with separate streaming control
    train_dataset, val_dataset, test_dataset, len_train_data, len_test_data = get_dataset_with_separate_streaming(
        base_path=base_path,
        dataset=dataset,
        external_knowledge=external_knowledge,
        use_train_streaming=use_streaming,
        use_val_streaming=use_streaming_val_dataset
    )

    # Maintain backward compatibility for return values
    if use_streaming:
        return train_dataset, val_dataset, test_dataset, len_train_data, len_test_data
    else:
        return train_dataset, val_dataset, test_dataset, len_train_data


def _convert_to_streaming_datasets(
        train_data: List,
        val_data: Optional[List],
        test_data: Optional[List]
) -> Tuple:
    """
    Convert data lists to streaming datasets
    DEPRECATED: Use get_dataset_with_separate_streaming instead

    Args:
        train_data: Training data list
        val_data: Validation data list (optional)
        test_data: Test data list (optional)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, len_train, len_test)
    """
    print("⚠️  Warning: _convert_to_streaming_datasets is deprecated. Use get_dataset_with_separate_streaming instead.")

    print("Converting datasets to streaming format...")

    def convert_to_streaming_dataset(data_list: Optional[List]) -> Optional[IterableDataset]:
        if not data_list:
            return None

        def gen_examples():
            for example in data_list:
                yield example

        return IterableDataset.from_generator(gen_examples)

    len_train_data = len(train_data) if train_data else 0
    len_test_data = len(test_data) if test_data else 0

    train_dataset = convert_to_streaming_dataset(train_data)
    val_dataset = convert_to_streaming_dataset(val_data)
    test_dataset = convert_to_streaming_dataset(test_data)

    print(f"Successfully created streaming datasets")
    print(f"  • Training: {len_train_data} samples")
    print(f"  • Validation: {len(val_data) if val_data else 0} samples")
    print(f"  • Test: {len_test_data} samples")

    return train_dataset, val_dataset, test_dataset, len_train_data, len_test_data


def validate_sample_structure(sample: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate individual sample structure

    Args:
        sample: Data sample dictionary

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    required_fields = ["image", "question", "answer"]
    for field in required_fields:
        if field not in sample:
            errors.append(f"Missing required field: {field}")
        elif not sample[field]:
            errors.append(f"Empty field: {field}")

    # Check if need_external_knowledge is boolean
    if "need_external_knowledge" in sample:
        if not isinstance(sample["need_external_knowledge"], bool):
            errors.append("need_external_knowledge should be boolean")

    return len(errors) == 0, errors


def convert_to_conversation(
        sample: Dict[str, Any],
        semart_dataset: Optional[pd.DataFrame] = None,
        is_test: bool = False,
        base_path: str = "data",
        instruction: str = None
) -> Dict[str, Any]:
    """
    Convert a sample to a conversation format for model training with enhanced validation

    Args:
        sample: Data sample
        semart_dataset: External knowledge dataset
        is_test: Whether this is for testing (no assistant response)
        base_path: Base path for images
        instruction: Custom instruction (optional)

    Returns:
        Formatted conversation dictionary
    """
    # Validate sample structure
    is_valid, errors = validate_sample_structure(sample)
    if not is_valid:
        print(f"Warning: Invalid sample structure: {errors}")
        return {"messages": []}

    # Default instruction
    if instruction is None:
        instruction = "You are an expert art historian. Answer the questions you will be asked about the image."

    try:
        # Load image
        image = extract_image(sample["image"], base_path=base_path)

        # Build conversation content
        content = [
            {"type": "text", "text": instruction},
            {"type": "image", "image": image}
        ]

        # Add description if external knowledge is needed
        if sample.get("need_external_knowledge", False) and semart_dataset is not None:
            try:
                matching_rows = semart_dataset[semart_dataset['image_file'] == sample["image"]]
                if not matching_rows.empty:
                    description = matching_rows['description'].values[0]
                    content.append({"type": "text", "text": description})
                else:
                    print(f"Warning: No description found for image {sample['image']}")
            except Exception as e:
                print(f"Warning: Error getting description for {sample['image']}: {e}")

        # Add question
        content.append({"type": "text", "text": sample["question"]})

        # Build conversation
        conversation = [{"role": "user", "content": content}]

        # Add assistant response if not test
        if not is_test:
            conversation.append({
                "role": "assistant",
                "content": [{"type": "text", "text": sample["answer"]}]
            })

        return {"messages": conversation}

    except Exception as e:
        print(f"Error converting sample to conversation: {e}")
        return {"messages": []}


def prepare_streaming_dataset(
        streaming_dataset: IterableDataset,
        config: Dict[str, Any],
        semart_dataset: Optional[pd.DataFrame] = None,
        base_path: str = "data"
) -> Optional[IterableDataset]:
    """
    Prepare the streaming dataset for training with enhanced configuration

    Args:
        streaming_dataset: Streaming dataset to be processed
        config: Configuration dictionary containing training parameters
        semart_dataset: Dataset containing semantic art descriptions
        base_path: Base path for dataset files

    Returns:
        Processed dataset ready for training
    """
    if streaming_dataset is None:
        return None

    try:
        # Apply shuffle with buffer size from config
        buffer_size = config.get("stream_buffer_size", 1000)
        dataset = streaming_dataset.shuffle(buffer_size=buffer_size)
        print(f"Applied shuffle with buffer size: {buffer_size}")

        # Get instruction from config
        instruction = config.get("instruction")

        # Convert every example to a conversation format
        def convert_to_conversation_streaming(example):
            return convert_to_conversation(
                example,
                semart_dataset=semart_dataset,
                base_path=base_path,
                instruction=instruction
            )

        # Apply the conversion function to the streaming dataset
        processed_dataset = dataset.map(convert_to_conversation_streaming)

        print("Successfully prepared streaming dataset")
        return processed_dataset

    except Exception as e:
        print(f"Error preparing streaming dataset: {e}")
        raise


def load_external_knowledge(
        external_knowledge_path: str,
        encoding: str = "latin1",
        separator: str = "\t"
) -> pd.DataFrame:
    """
    Load external knowledge dataset with validation

    Args:
        external_knowledge_path: Path to external knowledge CSV file
        encoding: File encoding
        separator: CSV separator

    Returns:
        Pandas DataFrame with external knowledge

    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: If file cannot be loaded
    """
    if not os.path.exists(external_knowledge_path):
        raise FileNotFoundError(f"External knowledge file not found: {external_knowledge_path}")

    try:
        df = pd.read_csv(external_knowledge_path, sep=separator, encoding=encoding, header=0)

        # Validate required columns
        required_columns = ['image_file', 'description']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns in external knowledge file: {missing_columns}")

        print(f"Loaded external knowledge: {len(df)} entries")
        return df

    except Exception as e:
        raise Exception(f"Error loading external knowledge from {external_knowledge_path}: {e}")


def get_dataset_statistics(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about a dataset

    Args:
        data: List of data samples

    Returns:
        Dictionary with dataset statistics
    """
    if not data:
        return {"total_samples": 0}

    stats = {
        "total_samples": len(data),
        "with_external_knowledge": sum(1 for sample in data if sample.get("need_external_knowledge", False)),
        "without_external_knowledge": sum(1 for sample in data if not sample.get("need_external_knowledge", False)),
        "unique_images": len(set(sample.get("image", "") for sample in data if sample.get("image"))),
    }

    # Calculate average question/answer lengths
    questions = [sample.get("question", "") for sample in data if sample.get("question")]
    answers = [sample.get("answer", "") for sample in data if sample.get("answer")]

    if questions:
        stats["avg_question_length"] = sum(len(q.split()) for q in questions) / len(questions)

    if answers:
        stats["avg_answer_length"] = sum(len(a.split()) for a in answers) / len(answers)

    return stats


def print_dataset_info(train_data: List, val_data: Optional[List], test_data: Optional[List]):
    """
    Print comprehensive dataset information

    Args:
        train_data: Training dataset
        val_data: Validation dataset (optional)
        test_data: Test dataset (optional)
    """
    print("\n" + "=" * 50)
    print("📊 DATASET INFORMATION")
    print("=" * 50)

    if train_data:
        train_stats = get_dataset_statistics(train_data)
        print(f"🚂 Training Dataset:")
        print(f"   • Total samples: {train_stats['total_samples']}")
        print(f"   • With external knowledge: {train_stats['with_external_knowledge']}")
        print(f"   • Without external knowledge: {train_stats['without_external_knowledge']}")
        print(f"   • Unique images: {train_stats['unique_images']}")
        if 'avg_question_length' in train_stats:
            print(f"   • Avg question length: {train_stats['avg_question_length']:.1f} words")
        if 'avg_answer_length' in train_stats:
            print(f"   • Avg answer length: {train_stats['avg_answer_length']:.1f} words")

    if val_data:
        val_stats = get_dataset_statistics(val_data)
        print(f"\n✅ Validation Dataset:")
        print(f"   • Total samples: {val_stats['total_samples']}")
        print(f"   • With external knowledge: {val_stats['with_external_knowledge']}")
        print(f"   • Without external knowledge: {val_stats['without_external_knowledge']}")
        print(f"   • Unique images: {val_stats['unique_images']}")

    if test_data:
        test_stats = get_dataset_statistics(test_data)
        print(f"\n🧪 Test Dataset:")
        print(f"   • Total samples: {test_stats['total_samples']}")
        print(f"   • With external knowledge: {test_stats['with_external_knowledge']}")
        print(f"   • Without external knowledge: {test_stats['without_external_knowledge']}")
        print(f"   • Unique images: {test_stats['unique_images']}")

    print("=" * 50)


def validate_image_paths(data: List[Dict[str, Any]], base_path: str = "data") -> Tuple[List[str], List[str]]:
    """
    Validate that all image paths in dataset exist

    Args:
        data: Dataset samples
        base_path: Base path for images

    Returns:
        Tuple of (valid_images, missing_images)
    """
    if not data:
        return [], []

    images_path = os.path.join(base_path, "Images")
    valid_images = []
    missing_images = []

    unique_images = set(sample.get("image", "") for sample in data if sample.get("image"))

    for image_name in unique_images:
        image_path = os.path.join(images_path, image_name)
        if os.path.exists(image_path):
            valid_images.append(image_name)
        else:
            missing_images.append(image_name)

    if missing_images:
        print(f"⚠️  Warning: {len(missing_images)} images not found:")
        for img in missing_images[:5]:  # Show first 5
            print(f"   • {img}")
        if len(missing_images) > 5:
            print(f"   • ... and {len(missing_images) - 5} more")

    return valid_images, missing_images


def sample_dataset_preview(data: List[Dict[str, Any]], num_samples: int = 3) -> None:
    """
    Print preview of dataset samples

    Args:
        data: Dataset samples
        num_samples: Number of samples to preview
    """
    if not data:
        print("No data to preview")
        return

    print(f"\n📋 Dataset Preview ({min(num_samples, len(data))} samples):")
    print("-" * 50)

    for i, sample in enumerate(data[:num_samples]):
        print(f"Sample {i + 1}:")
        print(f"  Image: {sample.get('image', 'N/A')}")
        print(f"  Question: {sample.get('question', 'N/A')[:100]}...")
        print(f"  Answer: {sample.get('answer', 'N/A')[:100]}...")
        print(f"  External knowledge: {sample.get('need_external_knowledge', False)}")
        print("-" * 30)