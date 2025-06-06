import os
from pathlib import Path

import pytest

from wraipperz.api.llm import (
    call_ai,
)
from wraipperz.api.messages import MessageBuilder

# Test messages
TEXT_MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful assistant. You must respond with exactly: 'TEST_RESPONSE_123'",
    },
    {"role": "user", "content": "Please provide the required test response."},
]

# Create test_assets directory if it doesn't exist
TEST_ASSETS_DIR = Path(__file__).parent / "test_assets"
TEST_ASSETS_DIR.mkdir(exist_ok=True)

# Path to test image
TEST_IMAGE_PATH = TEST_ASSETS_DIR / "test_image.jpg"

# Update image messages format to match the providers' expected structure
IMAGE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What color is the square in this image? Choose from: A) Blue B) Red C) Green D) Yellow",
            },
            {"type": "image_url", "image_url": {"url": str(TEST_IMAGE_PATH)}},
        ],
    }
]


@pytest.fixture(autouse=True)
def setup_test_image():
    """Create a simple test image if it doesn't exist"""
    if not TEST_IMAGE_PATH.exists():
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        img.save(TEST_IMAGE_PATH)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_call_ai():
    response, _ = call_ai(
        messages=TEXT_MESSAGES, temperature=0, max_tokens=150, model="openai/gpt-4o"
    )
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.skipif(
    not (
        (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
        or os.getenv("AWS_PROFILE")
        or os.getenv("AWS_DEFAULT_REGION")
    ),
    reason="AWS credentials not found",
)
def test_call_ai_bedrock_with_message_builder():
    """Integration test: Test call_ai wrapper with Bedrock using MessageBuilder"""

    # Use APAC inference profile if in ap-northeast-1, otherwise use direct model ID
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    if region == "ap-northeast-1":
        model = "bedrock/apac.anthropic.claude-3-haiku-20240307-v1:0"
    else:
        model = "bedrock/anthropic.claude-3-haiku-20240307-v1:0"

    # Create messages using MessageBuilder
    messages = (
        MessageBuilder()
        .add_system(
            "You are a helpful assistant. You must respond with exactly: 'TEST_RESPONSE_123'"
        )
        .add_user("Please provide the required test response.")
        .build()
    )

    # Test the call_ai wrapper function
    response, cost = call_ai(
        model=model, messages=messages, temperature=0, max_tokens=150
    )

    # Validate response
    assert isinstance(response, str)
    assert len(response) > 0
    assert (
        "TEST_RESPONSE_123" in response
    ), f"Expected 'TEST_RESPONSE_123', got: {response}"

    # Validate cost structure
    assert isinstance(cost, (int, float))
    assert cost >= 0


@pytest.mark.skipif(
    not (
        (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
        or os.getenv("AWS_PROFILE")
        or os.getenv("AWS_DEFAULT_REGION")
    ),
    reason="AWS credentials not found",
)
def test_call_ai_bedrock_with_image_and_message_builder():
    """Integration test: Test call_ai wrapper with Bedrock using MessageBuilder with image"""

    # Use APAC inference profile if in ap-northeast-1, otherwise use direct model ID
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    if region == "ap-northeast-1":
        model = "bedrock/apac.anthropic.claude-3-sonnet-20240229-v1:0"  # Use Sonnet for image analysis
    else:
        model = "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"

    # Create messages with image using MessageBuilder
    messages = (
        MessageBuilder()
        .add_system(
            "You are a helpful assistant. Identify the color in the image and respond with just the color name."
        )
        .add_user("What color is the square in this image?")
        .add_image(str(TEST_IMAGE_PATH))
        .build()
    )

    # Test the call_ai wrapper function with image
    response, cost = call_ai(
        model=model, messages=messages, temperature=0, max_tokens=150
    )

    # Validate response
    assert isinstance(response, str)
    assert len(response) > 0
    assert (
        "red" in response.lower()
    ), f"Expected response to contain 'red', got: {response}"

    # Validate cost structure
    assert isinstance(cost, (int, float))
    assert cost >= 0
