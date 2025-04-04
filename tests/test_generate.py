import os
from pathlib import Path

import pytest
from PIL import Image

from wraipperz.api.llm import (
    generate,
)

# Create test_assets directory if it doesn't exist
TEST_ASSETS_DIR = Path(__file__).parent / "test_assets"
TEST_ASSETS_DIR.mkdir(exist_ok=True)

# Path to test image
TEST_IMAGE_PATH = TEST_ASSETS_DIR / "test_image.jpg"


# Create test image if it doesn't exist
def create_test_image():
    if not TEST_IMAGE_PATH.exists():
        img = Image.new("RGB", (100, 100), color="red")
        img.save(TEST_IMAGE_PATH)


# Basic messages for text-to-image generation
TEXT_TO_IMAGE_MESSAGES = [
    # {
    #    "role": "system",
    #    "content": "You are a creative AI assistant that specializes in generating images.",
    # },
    {
        "role": "user",
        "content": "Generate a simple image of a red circle on a white background, and also say hi to the user.",
    },
]

# Messages for image-to-image generation
IMAGE_TO_IMAGE_MESSAGES = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Transform this red square into a blue circle.",
            },
            {"type": "image_url", "image_url": {"url": str(TEST_IMAGE_PATH)}},
        ],
    }
]


@pytest.fixture(autouse=True)
def setup():
    create_test_image()


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not found")
def test_generate():
    # Test text-to-image generation
    result, cost = generate(
        messages=TEXT_TO_IMAGE_MESSAGES,
        temperature=0.7,
        max_tokens=4096,
        model="gemini/gemini-2.0-flash-exp-image-generation",
    )

    # Check structure of response
    assert isinstance(result, dict)
    assert "text" in result
    assert "images" in result

    # Check text response
    assert isinstance(result["text"], str)
    assert len(result["text"]) > 0

    # Check image response
    assert isinstance(result["images"], list)
    # Note: Some responses might not include images depending on the model's decision
    if result["images"]:
        for img in result["images"]:
            assert isinstance(img, Image.Image)

    # Save generated images for inspection
    for i, image in enumerate(result["images"]):
        output_path = TEST_ASSETS_DIR / f"generated_image_{i}.png"
        image.save(output_path)
        print(f"Generated image saved to {output_path}")


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not found")
def test_generate_with_input_image():
    # Test image-to-image generation
    result, cost = generate(
        messages=IMAGE_TO_IMAGE_MESSAGES,
        temperature=0.7,
        max_tokens=4096,
        model="gemini/gemini-2.0-flash-exp-image-generation",
    )

    assert isinstance(result, dict)
    assert "text" in result
    assert "images" in result

    # Check text response
    assert isinstance(result["text"], str)

    # Check image response
    assert isinstance(result["images"], list)
    # Note: Some responses might not include images depending on the model's decision
    if result["images"]:
        for img in result["images"]:
            assert isinstance(img, Image.Image)

    # Save generated images for inspection
    for i, image in enumerate(result["images"]):
        output_path = TEST_ASSETS_DIR / f"transformed_image_{i}.png"
        image.save(output_path)
        print(f"Transformed image saved to {output_path}")


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not found")
def test_invalid_model():
    print("Starting test_invalid_model")

    try:
        # Test with a model that doesn't support image generation
        with pytest.raises(ValueError, match="does not support image generation"):
            generate(
                messages=TEXT_TO_IMAGE_MESSAGES,
                temperature=0.7,
                max_tokens=4096,
                model="openai/gpt-4o",  # OpenAI models don't support image generation in our implementation
            )
        print("Invalid model test passed")
    except Exception as e:
        print(f"Error in test_invalid_model: {type(e).__name__}: {e}")
        raise
