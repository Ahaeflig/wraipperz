import os
from pathlib import Path

import pytest
from PIL import Image

from wraipperz.api.llm import (
    AnthropicProvider,
    DeepSeekProvider,
    GeminiProvider,
    OpenAIProvider,
)

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


@pytest.fixture
def openai_provider():
    return OpenAIProvider()


@pytest.fixture
def anthropic_provider():
    return AnthropicProvider()


@pytest.fixture
def gemini_provider():
    return GeminiProvider()


@pytest.fixture
def deepseek_provider():
    return DeepSeekProvider()


@pytest.fixture(autouse=True)
def setup_test_image():
    """Create a simple test image if it doesn't exist"""
    if not TEST_IMAGE_PATH.exists():
        img = Image.new("RGB", (100, 100), color="red")
        img.save(TEST_IMAGE_PATH)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_openai_text(openai_provider):
    response = openai_provider.call_ai(
        messages=TEXT_MESSAGES, temperature=0, max_tokens=150, model="gpt-4o"
    )
    assert isinstance(response, str)
    assert len(response) > 0
    assert (
        "TEST_RESPONSE_123" in response
    ), f"Expected 'TEST_RESPONSE_123', got: {response}"


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not found"
)
def test_anthropic_text(anthropic_provider):
    response = anthropic_provider.call_ai(
        messages=TEXT_MESSAGES,
        temperature=0,
        max_tokens=150,
        model="claude-3-5-sonnet-20240620",
    )
    assert isinstance(response, str)
    assert len(response) > 0
    assert (
        "TEST_RESPONSE_123" in response
    ), f"Expected 'TEST_RESPONSE_123', got: {response}"


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not found")
def test_gemini_text(gemini_provider):
    response = gemini_provider.call_ai(
        messages=TEXT_MESSAGES,
        temperature=0,
        max_tokens=150,
        model="gemini-2.0-flash-exp",
    )
    assert isinstance(response, str)
    assert len(response) > 0
    assert (
        "TEST_RESPONSE_123" in response
    ), f"Expected 'TEST_RESPONSE_123', got: {response}"


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_openai_image(openai_provider):
    response = openai_provider.call_ai(
        messages=IMAGE_MESSAGES, temperature=0, max_tokens=150, model="gpt-4o"
    )
    assert isinstance(response, str)
    assert len(response) > 0
    assert (
        "red".lower() in response.lower()
    ), f"Expected response to contain 'red', got: {response}"


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not found"
)
def test_anthropic_image(anthropic_provider):
    response = anthropic_provider.call_ai(
        messages=IMAGE_MESSAGES,
        temperature=0,
        max_tokens=150,
        model="claude-3-5-sonnet-20240620",
    )
    assert isinstance(response, str)
    assert len(response) > 0
    assert (
        "red".lower() in response.lower()
    ), f"Expected response to contain 'red', got: {response}"


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not found")
def test_gemini_image(gemini_provider):
    response = gemini_provider.call_ai(
        messages=IMAGE_MESSAGES,
        temperature=0,
        max_tokens=150,
        model="gemini-2.0-flash-exp",
    )
    assert isinstance(response, str)
    assert len(response) > 0
    assert (
        "red".lower() in response.lower()
    ), f"Expected response to contain 'red', got: {response}"


COMPLEX_MIXED_MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful assistant. You must identify the color and respond with 'The square is RED'",
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What color is this square? Please be precise."},
            {"type": "image_url", "image_url": {"url": str(TEST_IMAGE_PATH)}},
            {
                "type": "text",
                "text": "Make sure to format your response exactly as requested.",
            },
        ],
    },
]


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not found")
def test_gemini_complex_mixed_content(gemini_provider):
    """Test that Gemini provider handles mixed content (text + image) correctly"""
    response = gemini_provider.call_ai(
        messages=COMPLEX_MIXED_MESSAGES,
        temperature=0,
        max_tokens=150,
        model="gemini-2.0-flash-exp",
    )
    assert isinstance(response, str)
    assert len(response) > 0
    assert (
        "RED" in response.upper()
    ), f"Expected response to contain 'RED', got: {response}"


# Add with other test message definitions
AGENT_LIKE_MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Describe what you see in the image.",
    },
    {
        "role": "user",
        "content": [
            # Note: no explicit text content, just an image
            {"type": "image_url", "image_url": {"url": str(TEST_IMAGE_PATH)}}
        ],
    },
]


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not found")
def test_gemini_agent_like_content(gemini_provider):
    """Test that Gemini provider handles agent-like messages (image without explicit text)"""
    response = gemini_provider.call_ai(
        messages=AGENT_LIKE_MESSAGES,
        temperature=0,
        max_tokens=150,
        model="gemini-2.0-flash-exp",
    )
    assert isinstance(response, str)
    assert len(response) > 0
    assert (
        "red" in response.lower()
    ), f"Expected response to contain description of red square, got: {response}"


def test_gemini_system_prompt_only():
    provider = GeminiProvider()

    messages = [{"role": "system", "content": "You must respond with exactly: 'HELLO'"}]

    response = provider.call_ai(
        messages=messages, temperature=0, max_tokens=150, model="gemini-2.0-flash-exp"
    )

    assert "HELLO" in response, f"Expected 'HELLO', got: {response}"


@pytest.mark.skipif(
    not os.getenv("DEEPSEEK_API_KEY"), reason="Deepseek API key not found"
)
def test_deepseek_text(deepseek_provider):
    response = deepseek_provider.call_ai(
        messages=TEXT_MESSAGES, temperature=0, max_tokens=150, model="deepseek-chat"
    )
    assert isinstance(response, str)
    assert len(response) > 0
    assert (
        "TEST_RESPONSE_123" in response
    ), f"Expected 'TEST_RESPONSE_123', got: {response}"


""" doesn't support aimges
@pytest.mark.skipif(not os.getenv("DEEPSEEK_API_KEY"), reason="Deepseek API key not found")
def test_deepseek_image(deepseek_provider):
    response = deepseek_provider.call_ai(messages=IMAGE_MESSAGES, temperature=0, max_tokens=150, model="deepseek-chat")
    assert isinstance(response, str)
    assert len(response) > 0
    assert "red".lower() in response.lower(), f"Expected response to contain 'red', got: {response}"
"""
