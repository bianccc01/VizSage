"""
Text formatting utilities for vision model training
"""

import re
from typing import Dict, Any, Optional


def create_formatting_function(tokenizer):
    """
    Factory function to create formatting function for dataset processing

    Args:
        tokenizer: Tokenizer for chat template application

    Returns:
        Formatting function for use with SFTTrainer
    """

    def formatting_func(example: Dict[str, Any]) -> str:
        """
        Format the conversation for vision model training
        Converts multimodal conversations (text + images) to model format

        Args:
            example: Dataset example containing messages

        Returns:
            Formatted conversation string
        """
        try:
            messages = example.get("messages", [])

            if not messages:
                print("Warning: Empty messages in example")
                return ""

            # Debug counter for first few examples
            if not hasattr(formatting_func, 'debug_count'):
                formatting_func.debug_count = 0

            # Debug output for first 3 examples
            if formatting_func.debug_count < 3:
                _debug_messages(messages, formatting_func.debug_count)

            # Apply chat template to convert messages to model format
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )

            # Cleanup corrupted characters and control characters
            if formatted:
                formatted = _clean_text(formatted)

            # Debug output
            if formatting_func.debug_count < 3:
                _debug_formatted_output(formatted, formatting_func.debug_count)
                formatting_func.debug_count += 1

            return formatted or ""

        except Exception as e:
            print(f"Error in formatting_func: {e}")
            print(f"Example keys: {list(example.keys()) if isinstance(example, dict) else 'Not a dict'}")
            return ""

    return formatting_func


def _debug_messages(messages: list, debug_count: int) -> None:
    """Debug helper to print message structure"""
    # Uncomment for debugging
    # print(f"\n=== Formatting Debug {debug_count} ===")
    # print(f"Messages: {len(messages)} messages")

    for i, msg in enumerate(messages):
        # print(f"  Message {i} - Role: {msg.get('role', 'NO_ROLE')}")
        content = msg.get('content', [])

        if isinstance(content, list):
            for j, content_item in enumerate(content):
                content_type = content_item.get('type', 'unknown')
                if content_type == 'text':
                    text_preview = content_item.get('text', '')[:50]
                    # print(f"    Content {j} (text): {text_preview}...")
                # else:
                # print(f"    Content {j} ({content_type})")
        # else:
        # print(f"    Content: {str(content)[:50]}...")


def _debug_formatted_output(formatted: Optional[str], debug_count: int) -> None:
    """Debug helper to print formatted output"""
    # Uncomment for debugging
    # print(f"Formatted length: {len(formatted) if formatted else 0}")
    # print(f"Formatted preview: {formatted[:200] if formatted else 'EMPTY'}...")
    # print("=" * 50)
    pass


def _clean_text(text: str) -> str:
    """
    Clean corrupted characters and control characters from text

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    # Remove specific corrupted character
    text = text.replace('锦', '')

    # Remove control characters (keeping newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

    # Strip whitespace
    text = text.strip()

    return text


def validate_example_structure(example: Dict[str, Any]) -> bool:
    """
    Validate that example has the required structure for formatting

    Args:
        example: Dataset example to validate

    Returns:
        True if example is valid, False otherwise
    """
    if not isinstance(example, dict):
        return False

    messages = example.get("messages", [])
    if not isinstance(messages, list) or len(messages) == 0:
        return False

    # Check if messages have required structure
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if 'role' not in msg or 'content' not in msg:
            return False

    return True


def count_tokens_in_example(example: Dict[str, Any], tokenizer) -> int:
    """
    Count tokens in a formatted example

    Args:
        example: Dataset example
        tokenizer: Tokenizer for encoding

    Returns:
        Number of tokens in formatted example
    """
    try:
        formatting_func = create_formatting_function(tokenizer)
        formatted_text = formatting_func(example)

        if not formatted_text:
            return 0

        tokens = tokenizer.encode(formatted_text)
        return len(tokens)

    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0


def filter_examples_by_length(examples: list, tokenizer, max_length: int = 2048) -> list:
    """
    Filter examples by token length

    Args:
        examples: List of dataset examples
        tokenizer: Tokenizer for encoding
        max_length: Maximum allowed token length

    Returns:
        Filtered list of examples
    """
    filtered_examples = []

    for example in examples:
        if not validate_example_structure(example):
            continue

        token_count = count_tokens_in_example(example, tokenizer)

        if token_count <= max_length and token_count > 0:
            filtered_examples.append(example)

    return filtered_examples