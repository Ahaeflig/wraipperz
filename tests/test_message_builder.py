import os
from pathlib import Path

import pytest
from PIL import Image

from wraipperz.api.llm import AnthropicProvider, GeminiProvider, OpenAIProvider
from wraipperz.api.messages import MessageBuilder

# Create test_assets directory if it doesn't exist
TEST_ASSETS_DIR = Path(__file__).parent / "test_assets"
TEST_ASSETS_DIR.mkdir(exist_ok=True)

# Path to test image
TEST_IMAGE_PATH = TEST_ASSETS_DIR / "test_image.jpg"


@pytest.fixture(autouse=True)
def setup_test_image():
    """Create a simple test image if it doesn't exist"""
    if not TEST_IMAGE_PATH.exists():
        img = Image.new("RGB", (100, 100), color="red")
        img.save(TEST_IMAGE_PATH)


def test_message_builder_basic():
    """Test basic MessageBuilder functionality"""
    # Create simple messages
    messages = (
        MessageBuilder()
        .add_system("You are a helpful assistant.")
        .add_user("Hello!")
        .add_assistant("How can I help you today?")
        .build()
    )

    # Verify the structure
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello!"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "How can I help you today?"


def test_message_builder_with_image():
    """Test MessageBuilder with image content"""
    # Create message with an image
    messages = (
        MessageBuilder()
        .add_user("What's in this image?")
        .add_image(str(TEST_IMAGE_PATH))
        .build()
    )

    # Verify the structure
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], list)
    assert len(messages[0]["content"]) == 2
    assert messages[0]["content"][0]["type"] == "text"
    assert messages[0]["content"][1]["type"] == "image_url"
    assert TEST_IMAGE_PATH.name in messages[0]["content"][1]["image_url"]["url"]


def test_message_builder_complex():
    """Test MessageBuilder with complex, mixed content"""
    # Create a more complex message structure
    messages = (
        MessageBuilder()
        .add_system("You are a helpful assistant.")
        .add_user("Hello! I have a question about this image.")
        .add_image(str(TEST_IMAGE_PATH), "What color is this?")
        .add_assistant("I see the image shows a red square.")
        .add_user("Thank you. Can you describe its shape more precisely?")
        .build()
    )

    # Verify the structure
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."

    assert messages[1]["role"] == "user"
    assert isinstance(messages[1]["content"], list)
    assert messages[1]["content"][0]["type"] == "text"
    assert messages[1]["content"][1]["type"] == "image_url"
    assert messages[1]["content"][2]["type"] == "text"

    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "I see the image shows a red square."

    assert messages[3]["role"] == "user"
    assert (
        messages[3]["content"]
        == "Thank you. Can you describe its shape more precisely?"
    )


def test_message_builder_empty_user():
    """Test MessageBuilder with an empty user message followed by image"""
    messages = (
        MessageBuilder()
        .add_user()  # Empty user message
        .add_image(str(TEST_IMAGE_PATH))
        .build()
    )

    # Verify the structure - should have one user message with just the image
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert isinstance(messages[0]["content"], list)
    assert len(messages[0]["content"]) == 1  # Should only have the image
    assert messages[0]["content"][0]["type"] == "image_url"


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_openai_with_message_builder():
    """Integration test: Test OpenAI with MessageBuilder-created messages"""
    provider = OpenAIProvider()

    messages = (
        MessageBuilder()
        .add_system("You are a helpful assistant. Identify the color in the image.")
        .add_user("What color is the square in this image?")
        .add_image(str(TEST_IMAGE_PATH))
        .build()
    )

    response = provider.call_ai(
        messages=messages, temperature=0, max_tokens=150, model="gpt-4o"
    )

    assert isinstance(response, str)
    assert len(response) > 0
    assert (
        "red" in response.lower()
    ), f"Expected response to contain 'red', got: {response}"


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not found"
)
def test_anthropic_with_message_builder():
    """Integration test: Test Anthropic with MessageBuilder-created messages"""
    provider = AnthropicProvider()

    messages = (
        MessageBuilder()
        .add_system("You are a helpful assistant. Identify the color in the image.")
        .add_user("What color is the square in this image?")
        .add_image(str(TEST_IMAGE_PATH))
        .build()
    )

    response = provider.call_ai(
        messages=messages,
        temperature=0,
        max_tokens=150,
        model="claude-3-5-sonnet-20240620",
    )

    assert isinstance(response, str)
    assert len(response) > 0
    assert (
        "red" in response.lower()
    ), f"Expected response to contain 'red', got: {response}"


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not found")
def test_gemini_with_message_builder():
    """Integration test: Test Gemini with MessageBuilder-created messages"""
    provider = GeminiProvider()

    messages = (
        MessageBuilder()
        .add_system("You are a helpful assistant. Identify the color in the image.")
        .add_user("What color is the square in this image?")
        .add_image(str(TEST_IMAGE_PATH))
        .build()
    )

    response = provider.call_ai(
        messages=messages, temperature=0, max_tokens=150, model="gemini-2.0-flash-exp"
    )

    assert isinstance(response, str)
    assert len(response) > 0
    assert (
        "red" in response.lower()
    ), f"Expected response to contain 'red', got: {response}"
