import os
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

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

# Path to test video
TEST_VIDEO_PATH = TEST_ASSETS_DIR / "test_video.mp4"

# Path to test PDF
TEST_PDF_PATH = TEST_ASSETS_DIR / "test_actor.pdf"

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

# Video messages for testing
VIDEO_MESSAGES = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Analyze this video and describe what you see."},
            {"type": "video_url", "video_url": {"url": str(TEST_VIDEO_PATH)}},
        ],
    }
]

# PDF messages for testing
PDF_MESSAGES = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What is the name of the actor mentioned in this PDF? Respond with just the actor's name.",
            },
            {"type": "input_url", "input_url": {"url": str(TEST_PDF_PATH)}},
        ],
    }
]


# Pydantic models for structured output testing
class SimpleVideoAnalysis(BaseModel):
    """Simple video analysis response for testing."""

    description: str = Field(
        description="Brief description of what happens in the video"
    )

    duration_estimate: str = Field(
        description="Estimated duration category: short, medium, or long"
    )

    contains_motion: bool = Field(
        description="Whether the video contains significant motion"
    )


@pytest.fixture(autouse=True)
def setup_test_assets():
    """Create test image, video, and PDF if they don't exist"""
    # Create test image
    if not TEST_IMAGE_PATH.exists():
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        img.save(TEST_IMAGE_PATH)

    # Create a simple test video (red square moving)
    if not TEST_VIDEO_PATH.exists():
        try:
            import cv2
            import numpy as np

            # Create a simple 2-second video with a red square moving
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video = cv2.VideoWriter(str(TEST_VIDEO_PATH), fourcc, 10.0, (200, 200))

            for i in range(20):  # 2 seconds at 10 fps
                frame = np.zeros((200, 200, 3), dtype=np.uint8)
                # Draw a red square that moves
                x = 10 + i * 5
                cv2.rectangle(frame, (x, 50), (x + 40, 90), (0, 0, 255), -1)
                video.write(frame)

            video.release()
        except ImportError:
            # If OpenCV not available, create a minimal placeholder video
            # This won't be a real video but will exist for path testing
            TEST_VIDEO_PATH.write_text("placeholder video file for testing")

    # Create test PDF with actor information
    if not TEST_PDF_PATH.exists():
        try:
            from reportlab.lib.pagesizes import letter
            from reportlab.pdfgen import canvas

            c = canvas.Canvas(str(TEST_PDF_PATH), pagesize=letter)
            c.setFont("Helvetica-Bold", 20)
            c.drawString(100, 750, "Actor Information")
            c.setFont("Helvetica", 14)
            c.drawString(100, 700, "Name: Tom Hanks")
            c.drawString(100, 680, "Profession: Actor")
            c.drawString(
                100, 660, "Known For: Forrest Gump, Cast Away, Saving Private Ryan"
            )
            c.drawString(100, 640, "Born: July 9, 1956")
            c.drawString(100, 620, "Nationality: American")
            c.save()
        except ImportError:
            # If reportlab not available, create a minimal placeholder PDF
            TEST_PDF_PATH.write_text("placeholder PDF file for testing")


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_call_ai():
    response, _ = call_ai(
        messages=TEXT_MESSAGES, temperature=0, max_tokens=150, model="openai/gpt-4o"
    )
    assert isinstance(response, str)
    assert len(response) > 0


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_call_ai_with_pdf():
    """Integration test: Test call_ai with PDF file input using OpenAI"""

    # Create messages with PDF using MessageBuilder
    messages = (
        MessageBuilder()
        .add_system(
            "You are a helpful assistant. Extract information from documents accurately."
        )
        .add_user("What is the name of the actor mentioned in this PDF?")
        .add_pdf(str(TEST_PDF_PATH))
        .build()
    )

    # Test the call_ai wrapper function with PDF
    response, cost = call_ai(
        model="openai/gpt-4o",
        messages=messages,
        temperature=0,
        max_tokens=150,
    )

    # Validate response
    assert isinstance(response, str)
    assert len(response) > 0
    assert (
        "tom hanks" in response.lower()
    ), f"Expected response to contain 'Tom Hanks', got: {response}"

    # Validate cost structure
    assert isinstance(cost, (int, float))
    assert cost >= 0

    print(f"✅ OpenAI PDF analysis test passed! Response: {response}")


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_call_ai_with_pdf_using_messages():
    """Integration test: Test call_ai with PDF using raw message format"""

    # Test with raw message format (not using MessageBuilder)
    response, cost = call_ai(
        model="openai/gpt-4o-mini",
        messages=PDF_MESSAGES,
        temperature=0,
        max_tokens=150,
    )

    # Validate response
    assert isinstance(response, str)
    assert len(response) > 0
    assert (
        "tom hanks" in response.lower()
    ), f"Expected response to contain 'Tom Hanks', got: {response}"

    # Validate cost structure
    assert isinstance(cost, (int, float))
    assert cost >= 0

    print(f"✅ OpenAI PDF (raw messages) test passed! Response: {response}")


# Helper function for dynamic Bedrock model selection
def get_bedrock_model_by_region(base_model_id: str, region: str = None) -> str:
    """
    Get the appropriate Bedrock model ID based on AWS region.

    Args:
        base_model_id: The base model ID (e.g., "anthropic.claude-3-haiku-20240307-v1:0")
        region: AWS region (defaults to AWS_DEFAULT_REGION env var)

    Returns:
        Full model ID with appropriate inference profile prefix
    """
    if region is None:
        region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

    # Map regions to inference profile prefixes
    if region.startswith("ap-"):
        # Asia Pacific regions
        prefix = "apac."
    elif region.startswith("us-"):
        # US regions
        prefix = "us."
    elif region.startswith("eu-"):
        # Europe regions
        prefix = "eu."
    else:
        # For other regions or fallback, use direct model ID
        prefix = ""

    # Return the full model ID with bedrock prefix
    if prefix:
        return f"bedrock/{prefix}{base_model_id}"
    else:
        return f"bedrock/{base_model_id}"


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

    # Dynamically select appropriate model based on region
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    model = get_bedrock_model_by_region(
        "anthropic.claude-3-haiku-20240307-v1:0", region
    )

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

    # Dynamically select appropriate model based on region
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    model = get_bedrock_model_by_region(
        "anthropic.claude-3-sonnet-20240229-v1:0", region
    )  # Use Sonnet for image analysis

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


@pytest.mark.skipif(
    not (
        (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
        or os.getenv("AWS_PROFILE")
        or os.getenv("AWS_DEFAULT_REGION")
    ),
    reason="AWS credentials not found",
)
def test_call_ai_bedrock_claude_opus_4():
    """Integration test: Test call_ai wrapper with Bedrock Claude Opus 4"""

    # Dynamically select appropriate Claude Opus 4 model based on region
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    model = get_bedrock_model_by_region("anthropic.claude-opus-4-20250514-v1:0", region)

    # Create messages using MessageBuilder
    messages = (
        MessageBuilder()
        .add_system(
            "You are a helpful assistant. You must respond with exactly: 'OPUS_4_BEDROCK_TEST_789'"
        )
        .add_user("Please provide the required test response.")
        .build()
    )

    try:
        # Test the call_ai wrapper function with Claude Opus 4
        response, cost = call_ai(
            model=model, messages=messages, temperature=0, max_tokens=150
        )

        # Validate response
        assert isinstance(response, str)
        assert len(response) > 0
        assert (
            "OPUS_4_BEDROCK_TEST_789" in response
        ), f"Expected 'OPUS_4_BEDROCK_TEST_789', got: {response}"

        # Validate cost structure
        assert isinstance(cost, (int, float))
        assert cost >= 0

        print(
            f"✅ Claude Opus 4 Bedrock access confirmed! Response: {response[:100]}..."
        )

    except Exception as e:
        # If user doesn't have access to Opus 4, we'll get a specific error
        error_msg = str(e).lower()
        if any(
            access_error in error_msg
            for access_error in [
                "model not found",
                "invalid model",
                "not available",
                "access",
                "permission",
                "unauthorized",
                "accessdeniedexception",
                "validationexception",
            ]
        ):
            pytest.skip(
                f"Claude Opus 4 Bedrock model not accessible in region {region}: {e}"
            )
        else:
            # Re-raise if it's a different error
            raise


@pytest.mark.skipif(
    not (
        (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
        or os.getenv("AWS_PROFILE")
        or os.getenv("AWS_DEFAULT_REGION")
    ),
    reason="AWS credentials not found",
)
def test_call_ai_bedrock_claude_opus_4_with_image():
    """Integration test: Test call_ai wrapper with Bedrock Claude Opus 4 with image analysis"""

    # Dynamically select appropriate Claude Opus 4 model based on region
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    model = get_bedrock_model_by_region("anthropic.claude-opus-4-20250514-v1:0", region)

    # Create messages with image using MessageBuilder
    messages = (
        MessageBuilder()
        .add_system(
            "You are a helpful assistant. Analyze images carefully and identify the color in the image."
        )
        .add_user("What color is the square in this image?")
        .add_image(str(TEST_IMAGE_PATH))
        .build()
    )

    try:
        # Test the call_ai wrapper function with Claude Opus 4 and image
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

        print(
            f"✅ Claude Opus 4 Bedrock image analysis confirmed! Response: {response[:100]}..."
        )

    except Exception as e:
        # If user doesn't have access to Opus 4, we'll get a specific error
        error_msg = str(e).lower()
        if any(
            access_error in error_msg
            for access_error in [
                "model not found",
                "invalid model",
                "not available",
                "access",
                "permission",
                "unauthorized",
                "accessdeniedexception",
                "validationexception",
            ]
        ):
            pytest.skip(
                f"Claude Opus 4 Bedrock model not accessible in region {region}: {e}"
            )
        else:
            # Re-raise if it's a different error
            raise


@pytest.mark.parametrize(
    "model_base_id,expected_text",
    [
        ("anthropic.claude-3-haiku-20240307-v1:0", "HAIKU_BEDROCK_TEST"),
        ("anthropic.claude-3-sonnet-20240229-v1:0", "SONNET_BEDROCK_TEST"),
        ("anthropic.claude-3-5-sonnet-20240620-v1:0", "SONNET_35_BEDROCK_TEST"),
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", "SONNET_35_V2_BEDROCK_TEST"),
        ("anthropic.claude-opus-4-20250514-v1:0", "OPUS_4_BEDROCK_TEST"),
        ("anthropic.claude-sonnet-4-20250514-v1:0", "SONNET_4_BEDROCK_TEST"),
    ],
)
@pytest.mark.skipif(
    not (
        (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
        or os.getenv("AWS_PROFILE")
        or os.getenv("AWS_DEFAULT_REGION")
    ),
    reason="AWS credentials not found",
)
def test_call_ai_bedrock_models_parametrized(model_base_id, expected_text):
    """Parametrized test for multiple Bedrock Claude models with region-aware selection"""

    # Dynamically select appropriate model based on region
    region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    model = get_bedrock_model_by_region(model_base_id, region)

    # Create messages using MessageBuilder
    messages = (
        MessageBuilder()
        .add_system(
            f"You are a helpful assistant. You must respond with exactly: '{expected_text}'"
        )
        .add_user("Please provide the required test response.")
        .build()
    )

    try:
        # Test the call_ai wrapper function
        response, cost = call_ai(
            model=model, messages=messages, temperature=0, max_tokens=150
        )

        # Validate response
        assert isinstance(response, str)
        assert len(response) > 0
        assert expected_text in response, f"Expected '{expected_text}', got: {response}"

        # Validate cost structure
        assert isinstance(cost, (int, float))
        assert cost >= 0

        print(f"✅ {model_base_id} Bedrock access confirmed in region {region}!")

    except Exception as e:
        # If user doesn't have access to this model, we'll get a specific error
        error_msg = str(e).lower()
        if any(
            access_error in error_msg
            for access_error in [
                "model not found",
                "invalid model",
                "not available",
                "access",
                "permission",
                "unauthorized",
                "accessdeniedexception",
                "validationexception",
            ]
        ):
            pytest.skip(
                f"{model_base_id} Bedrock model not accessible in region {region}: {e}"
            )
        else:
            # Re-raise if it's a different error
            raise


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not found"
)
def test_call_ai_anthropic_claude_sonnet_4():
    """Integration test: Test call_ai wrapper with Anthropic Claude Sonnet 4"""

    # Test the call_ai wrapper function with Claude Sonnet 4
    response, cost = call_ai(
        model="anthropic/claude-sonnet-4-20250514",
        messages=TEXT_MESSAGES,
        temperature=0,
        max_tokens=150,
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
    not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not found"
)
def test_call_ai_anthropic_claude_sonnet_4_with_image():
    """Integration test: Test call_ai wrapper with Anthropic Claude Sonnet 4 with image"""

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
        model="anthropic/claude-sonnet-4-20250514",
        messages=messages,
        temperature=0,
        max_tokens=150,
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


# ===== REASONING MODELS TESTS (o1, o3 series) =====


# ===== VERTEX AI PROVIDER TESTS =====


@pytest.mark.skipif(
    not (
        os.getenv("VERTEX_PROJECT_ID") and os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    ),
    reason="Vertex AI credentials not found",
)
def test_call_ai_vertex_claude_sonnet():
    """Integration test: Test call_ai wrapper with Vertex AI Claude Sonnet"""

    # Debug: Print environment variables
    print("\n🔍 Vertex AI Configuration:")
    print(f"   Project ID: {os.getenv('VERTEX_PROJECT_ID')}")
    print(f"   Location: {os.getenv('VERTEX_LOCATION', 'us-east5 (default)')}")
    print(f"   Credentials: {os.getenv('GOOGLE_APPLICATION_CREDENTIALS')}")

    # Test the call_ai wrapper function with Vertex AI Claude Sonnet
    try:
        response, cost = call_ai(
            model="vertex/claude-sonnet-4@20250514",
            messages=TEXT_MESSAGES,
            temperature=0,
            max_tokens=150,
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

        print(
            f"✅ Vertex AI Claude Sonnet access confirmed! Response: {response[:100]}..."
        )

    except Exception as e:
        error_msg = str(e).lower()
        # Check if this is a region availability issue
        if "us-central1" in error_msg and "not found" in error_msg:
            pytest.skip(
                f"Claude models may not be available in us-central1. "
                f"Try setting VERTEX_LOCATION=us-east5. Error: {e}"
            )
        elif any(err in error_msg for err in ["not found", "not available", "access"]):
            pytest.skip(f"Vertex AI Claude model not accessible in current region: {e}")
        else:
            raise


@pytest.mark.skipif(
    not (
        os.getenv("VERTEX_PROJECT_ID") and os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    ),
    reason="Vertex AI credentials not found",
)
def test_call_ai_vertex_claude_with_image():
    """Integration test: Test call_ai wrapper with Vertex AI Claude with image analysis"""

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
        model="vertex/claude-sonnet-4@20250514",
        messages=messages,
        temperature=0,
        max_tokens=150,
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

    print(
        f"✅ Vertex AI Claude image analysis confirmed! Response: {response[:100]}..."
    )


@pytest.mark.skipif(
    not (
        os.getenv("VERTEX_PROJECT_ID") and os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    ),
    reason="Vertex AI credentials not found",
)
def test_call_ai_vertex_claude_opus_4():
    """Integration test: Test call_ai wrapper with Vertex AI Claude Opus 4"""

    # Create messages using MessageBuilder
    messages = (
        MessageBuilder()
        .add_system(
            "You are a helpful assistant. You must respond with exactly: 'VERTEX_OPUS_4_TEST_456'"
        )
        .add_user("Please provide the required test response.")
        .build()
    )

    try:
        # Test the call_ai wrapper function with Vertex AI Claude Opus 4
        response, cost = call_ai(
            model="vertex/claude-opus-4@20250514",
            messages=messages,
            temperature=0,
            max_tokens=150,
        )

        # Validate response
        assert isinstance(response, str)
        assert len(response) > 0
        assert (
            "VERTEX_OPUS_4_TEST_456" in response
        ), f"Expected 'VERTEX_OPUS_4_TEST_456', got: {response}"

        # Validate cost structure
        assert isinstance(cost, (int, float))
        assert cost >= 0

        print(
            f"✅ Vertex AI Claude Opus 4 access confirmed! Response: {response[:100]}..."
        )

    except Exception as e:
        # If user doesn't have access to Opus 4, we'll get a specific error
        error_msg = str(e).lower()
        if any(
            access_error in error_msg
            for access_error in [
                "model not found",
                "invalid model",
                "not available",
                "access",
                "permission",
                "unauthorized",
                "permissiondenied",
                "forbidden",
            ]
        ):
            pytest.skip(f"Vertex AI Claude Opus 4 model not accessible: {e}")
        else:
            # Re-raise if it's a different error
            raise


@pytest.mark.skipif(
    not (
        os.getenv("VERTEX_PROJECT_ID") and os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    ),
    reason="Vertex AI credentials not found",
)
def test_call_ai_vertex_claude_sonnet_4():
    """Integration test: Test call_ai wrapper with Vertex AI Claude Sonnet 4"""

    # Create messages using MessageBuilder
    messages = (
        MessageBuilder()
        .add_system(
            "You are a helpful assistant. You must respond with exactly: 'VERTEX_SONNET_4_TEST_789'"
        )
        .add_user("Please provide the required test response.")
        .build()
    )

    try:
        # Test the call_ai wrapper function with Vertex AI Claude Sonnet 4
        response, cost = call_ai(
            model="vertex/claude-sonnet-4@20250514",
            messages=messages,
            temperature=0,
            max_tokens=150,
        )

        # Validate response
        assert isinstance(response, str)
        assert len(response) > 0
        assert (
            "VERTEX_SONNET_4_TEST_789" in response
        ), f"Expected 'VERTEX_SONNET_4_TEST_789', got: {response}"

        # Validate cost structure
        assert isinstance(cost, (int, float))
        assert cost >= 0

        print(
            f"✅ Vertex AI Claude Sonnet 4 access confirmed! Response: {response[:100]}..."
        )

    except Exception as e:
        # If user doesn't have access to Sonnet 4, we'll get a specific error
        error_msg = str(e).lower()
        if any(
            access_error in error_msg
            for access_error in [
                "model not found",
                "invalid model",
                "not available",
                "access",
                "permission",
                "unauthorized",
                "permissiondenied",
                "forbidden",
            ]
        ):
            pytest.skip(f"Vertex AI Claude Sonnet 4 model not accessible: {e}")
        else:
            # Re-raise if it's a different error
            raise


@pytest.mark.parametrize(
    "model_id,expected_text",
    [
        ("vertex/claude-opus-4@20250514", "VERTEX_OPUS_4_TEST"),
        ("vertex/claude-sonnet-4@20250514", "VERTEX_SONNET_4_TEST"),
    ],
)
@pytest.mark.skipif(
    not (
        os.getenv("VERTEX_PROJECT_ID") and os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    ),
    reason="Vertex AI credentials not found",
)
def test_call_ai_vertex_models_parametrized(model_id, expected_text):
    """Parametrized test for multiple Vertex AI Claude models"""

    # Create messages using MessageBuilder
    messages = (
        MessageBuilder()
        .add_system(
            f"You are a helpful assistant. You must respond with exactly: '{expected_text}'"
        )
        .add_user("Please provide the required test response.")
        .build()
    )

    try:
        # Test the call_ai wrapper function
        response, cost = call_ai(
            model=model_id, messages=messages, temperature=0, max_tokens=150
        )

        # Validate response
        assert isinstance(response, str)
        assert len(response) > 0
        assert expected_text in response, f"Expected '{expected_text}', got: {response}"

        # Validate cost structure
        assert isinstance(cost, (int, float))
        assert cost >= 0

        print(f"✅ {model_id} Vertex AI access confirmed!")

    except Exception as e:
        # If user doesn't have access to this model, we'll get a specific error
        error_msg = str(e).lower()
        if any(
            access_error in error_msg
            for access_error in [
                "model not found",
                "invalid model",
                "not available",
                "access",
                "permission",
                "unauthorized",
                "permissiondenied",
                "forbidden",
            ]
        ):
            pytest.skip(f"{model_id} Vertex AI model not accessible: {e}")
        else:
            # Re-raise if it's a different error
            raise


@pytest.mark.skipif(
    not (
        os.getenv("VERTEX_PROJECT_ID") and os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    ),
    reason="Vertex AI credentials not found",
)
def test_vertex_ai_provider_initialization():
    """Test that VertexAIProvider initializes correctly with environment variables"""

    from wraipperz.api.llm import VertexAIProvider

    try:
        # Should initialize successfully if env vars are set
        provider = VertexAIProvider()

        # Check that it has the expected attributes
        assert hasattr(provider, "sync_client")
        assert hasattr(provider, "async_client")
        assert hasattr(provider, "project_id")
        assert hasattr(provider, "location")
        assert provider.project_id == os.getenv("VERTEX_PROJECT_ID")

        # Check supported models
        assert len(provider.supported_models) > 0
        assert any("vertex/" in model for model in provider.supported_models)

        print(
            f"✅ VertexAIProvider initialization test passed! Project: {provider.project_id}, Location: {provider.location}"
        )

    except ImportError as e:
        pytest.skip(f"VertexAI dependencies not available: {e}")
    except Exception as e:
        pytest.fail(f"VertexAIProvider initialization failed: {e}")


@pytest.mark.skipif(
    not (
        os.getenv("VERTEX_PROJECT_ID") and os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    ),
    reason="Vertex AI credentials not found",
)
def test_vertex_ai_with_explicit_region():
    """Test Vertex AI with explicit us-east5 region (where Claude models are available)"""

    from wraipperz.api.llm import VertexAIProvider

    # Create provider with explicit us-east5 region
    try:
        provider = VertexAIProvider(
            project_id=os.getenv("VERTEX_PROJECT_ID"),
            location="us-east5",  # Explicitly use us-east5 where Claude models are available
        )

        print("\n🔍 Testing with explicit region:")
        print(f"   Project ID: {provider.project_id}")
        print(f"   Location: {provider.location}")

        # Test a simple call
        messages = [
            {"role": "user", "content": "Say 'Hello from Vertex AI' and nothing else."}
        ]

        response = provider.call_ai(
            messages=messages,
            temperature=0,
            max_tokens=50,
            model="vertex/claude-sonnet-4@20250514",  # Use a model that's actually available
        )

        assert isinstance(response, str)
        assert len(response) > 0
        print(f"✅ Vertex AI with explicit us-east5 region works! Response: {response}")

    except Exception as e:
        error_msg = str(e).lower()
        if any(
            err in error_msg
            for err in ["not found", "not available", "access", "permission"]
        ):
            pytest.skip(
                f"Claude models not accessible in us-east5. "
                f"Please verify model availability in your region. Error: {e}"
            )
        else:
            raise


# ===== REASONING MODELS TESTS (o1, o3 series) =====


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_call_ai_o1_mini_reasoning_model():
    """Integration test: Test call_ai wrapper with OpenAI o1-mini reasoning model"""

    # Create simple messages - o1-mini doesn't support system messages
    # Our wrapper should automatically convert system to user messages
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Solve this step by step.",
        },
        {"role": "user", "content": "What is 15 * 23? Show your reasoning."},
    ]

    try:
        # Test the call_ai wrapper function with o1-mini reasoning model
        # Note: Our wrapper handles the parameter conversion automatically
        response, cost = call_ai(
            model="openai/o1-mini",
            messages=messages,
            temperature=0.7,  # This will be ignored for reasoning models
            max_tokens=1000,  # This gets converted to max_completion_tokens
        )

        # Validate response
        assert isinstance(response, str)
        assert len(response) > 0
        assert (
            "345" in response
        ), f"Expected calculation result '345' in response, got: {response}"

        # Validate cost structure
        assert isinstance(cost, (int, float))
        assert cost >= 0

        print(f"✅ o1-mini model access confirmed! Response: {response[:100]}...")

    except Exception as e:
        # If user doesn't have access, we'll get a specific error
        error_msg = str(e).lower()
        if any(
            access_error in error_msg
            for access_error in [
                "model not found",
                "invalid model",
                "not available",
                "access",
                "permission",
                "unauthorized",
            ]
        ):
            pytest.skip(f"o1-mini model not accessible: {e}")
        else:
            # Re-raise if it's a different error
            raise


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_call_ai_o3_mini_reasoning_model():
    """Integration test: Test call_ai wrapper with OpenAI o3-mini reasoning model"""

    # o3-mini supports developer messages (converted from system)
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Solve this problem carefully.",
        },
        {
            "role": "user",
            "content": "If a train travels 60 mph for 2.5 hours, how far does it go?",
        },
    ]

    try:
        # Test the call_ai wrapper function with o3-mini reasoning model
        response, cost = call_ai(
            model="openai/o3-mini",
            messages=messages,
            temperature=0.5,  # This will be ignored for reasoning models
            max_tokens=1000,  # This gets converted to max_completion_tokens
        )

        # Validate response
        assert isinstance(response, str)
        assert len(response) > 0
        assert (
            "150" in response
        ), f"Expected calculation result '150' in response, got: {response}"

        # Validate cost structure
        assert isinstance(cost, (int, float))
        assert cost >= 0

        print(f"✅ o3-mini model access confirmed! Response: {response[:100]}...")

    except Exception as e:
        # If user doesn't have access, we'll get a specific error
        error_msg = str(e).lower()
        if any(
            access_error in error_msg
            for access_error in [
                "model not found",
                "invalid model",
                "not available",
                "access",
                "permission",
                "unauthorized",
            ]
        ):
            pytest.skip(f"o3-mini model not accessible: {e}")
        else:
            # Re-raise if it's a different error
            raise


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_call_ai_o1_with_vision():
    """Integration test: Test call_ai wrapper with OpenAI o1 reasoning model with vision"""

    # o1 supports vision - test with image analysis
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Analyze images carefully.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What color is the square in this image? Provide a clear, direct answer.",
                },
                {"type": "image_url", "image_url": {"url": str(TEST_IMAGE_PATH)}},
            ],
        },
    ]

    # Test the call_ai wrapper function with o1 vision
    response, cost = call_ai(
        model="openai/o1",
        messages=messages,
        temperature=0,  # This will be ignored for reasoning models
        max_tokens=500,  # This gets converted to max_completion_tokens
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


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_reasoning_model_parameter_handling():
    """Test that reasoning models properly handle parameter conversion"""

    # Test with parameters that should be filtered out for reasoning models
    messages = [{"role": "user", "content": "What is 2 + 2?"}]

    # Test with many unsupported parameters that should be filtered
    response, cost = call_ai(
        model="openai/o1-mini",
        messages=messages,
        temperature=0.8,  # Should be ignored
        max_tokens=500,  # Should become max_completion_tokens (increased)
        top_p=0.9,  # Should be ignored
        presence_penalty=0.5,  # Should be ignored
        frequency_penalty=0.3,  # Should be ignored
        logprobs=True,  # Should be ignored
    )

    # Should still work despite unsupported parameters
    assert isinstance(response, str)
    assert len(response) > 0
    assert "4" in response

    # Validate cost structure
    assert isinstance(cost, (int, float))
    assert cost >= 0


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_o1_system_message_conversion():
    """Test that o1-mini properly converts system messages to user messages"""

    # Test that system messages are properly converted for o1-mini
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Solve math problems step by step.",
        },
        {
            "role": "user",
            "content": "What is 17 * 19? Show your reasoning and calculation.",
        },
    ]

    try:
        response, cost = call_ai(
            model="openai/o1-mini",
            messages=messages,
            max_tokens=1000,  # Match the working test
        )

        # Should work without error (system message converted to user)
        assert isinstance(response, str)
        if len(response) == 0:
            pytest.fail(
                "Got empty response from o1-mini. This might indicate access issues or API problems."
            )

        assert "323" in response, f"Expected '323' in response, got: {response}"

        # Validate cost structure
        assert isinstance(cost, (int, float))
        assert cost >= 0

    except Exception as e:
        # If user doesn't have access, we'll get a specific error
        error_msg = str(e).lower()
        if any(
            access_error in error_msg
            for access_error in [
                "model not found",
                "invalid model",
                "not available",
                "access",
                "permission",
                "unauthorized",
            ]
        ):
            pytest.skip(f"o1-mini model not accessible: {e}")
        else:
            # Re-raise if it's a different error
            raise


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_o3_developer_message_conversion():
    """Test that o3-mini properly converts system messages to developer messages"""

    # Test that system messages are properly converted for o3-mini
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Be concise.",
        },
        {"role": "user", "content": "Explain photosynthesis in one sentence."},
    ]

    try:
        response, cost = call_ai(
            model="openai/o3-mini",
            messages=messages,
            max_tokens=500,  # Increased for reasoning models
        )

        # Should work without error (system message converted to developer)
        assert isinstance(response, str)
        assert len(response) > 0
        assert "photosynthesis" in response.lower()

        # Validate cost structure
        assert isinstance(cost, (int, float))
        assert cost >= 0

    except Exception as e:
        # If user doesn't have access, we'll get a specific error
        error_msg = str(e).lower()
        if any(
            access_error in error_msg
            for access_error in [
                "model not found",
                "invalid model",
                "not available",
                "access",
                "permission",
                "unauthorized",
            ]
        ):
            pytest.skip(f"o3-mini model not accessible: {e}")
        else:
            # Re-raise if it's a different error
            raise


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_call_ai_o3_full_model():
    """Integration test: Test call_ai wrapper with full OpenAI o3 model (non-mini)"""

    # Test the full o3 model to check access and functionality
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Provide accurate answers.",
        },
        {
            "role": "user",
            "content": "What is the capital of France? Answer with just the city name.",
        },
    ]

    try:
        response, cost = call_ai(
            model="openai/o3",
            messages=messages,
            max_tokens=50,
        )

        # Should work without error if user has access
        assert isinstance(response, str)
        assert len(response) > 0
        assert (
            "paris" in response.lower()
        ), f"Expected 'Paris' in response, got: {response}"

        # Validate cost structure
        assert isinstance(cost, (int, float))
        assert cost >= 0

        print(f"✅ o3 model access confirmed! Response: {response}")

    except Exception as e:
        # If user doesn't have access, we'll get a specific error
        error_msg = str(e).lower()
        if any(
            access_error in error_msg
            for access_error in [
                "model not found",
                "invalid model",
                "not available",
                "access",
                "permission",
                "unauthorized",
            ]
        ):
            pytest.skip(f"o3 model not accessible: {e}")
        else:
            # Re-raise if it's a different error
            raise


# ===== ANTHROPIC REASONING MODELS TESTS (Extended Thinking) =====


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not found"
)
def test_call_ai_claude_sonnet_4_reasoning():
    """Integration test: Test call_ai wrapper with Anthropic Claude Sonnet 4 with extended thinking"""

    # Test complex reasoning task that benefits from extended thinking
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Think step by step to solve problems accurately.",
        },
        {
            "role": "user",
            "content": "A farmer has 120 apples. He sells 40% to a market, gives 25% to his neighbors, and keeps the rest. How many apples does he keep? Show your reasoning step by step.",
        },
    ]

    # Test with automatic thinking budget (thinking=True)
    response, cost = call_ai(
        model="anthropic/claude-sonnet-4-20250514",
        messages=messages,
        temperature=1,  # Required for thinking models
        max_tokens=1500,  # Needs to be > 1024 to accommodate minimum thinking budget
        thinking=True,  # Enable extended thinking with automatic budget
    )

    # Validate response
    assert isinstance(response, str)
    assert len(response) > 0
    # 120 - (120*0.4) - (120*0.25) = 120 - 48 - 30 = 42
    assert (
        "42" in response
    ), f"Expected calculation result '42' in response, got: {response}"

    # Validate cost structure
    assert isinstance(cost, (int, float))
    assert cost >= 0

    print(f"✅ Claude Sonnet 4 reasoning test passed! Response: {response[:150]}...")


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not found"
)
def test_call_ai_claude_opus_4_reasoning():
    """Integration test: Test call_ai wrapper with Anthropic Claude Opus 4 with extended thinking"""

    # Test complex logical reasoning that benefits from extended thinking
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use careful logical reasoning to solve problems.",
        },
        {
            "role": "user",
            "content": "If all cats are mammals, and all mammals are animals, and Fluffy is a cat, what can we conclude about Fluffy? Explain your logical reasoning.",
        },
    ]

    try:
        # Test with manual thinking budget control
        response, cost = call_ai(
            model="anthropic/claude-opus-4-20250514",
            messages=messages,
            temperature=1,  # Required for thinking models
            max_tokens=5000,  # Needs to be > thinking budget
            thinking={
                "type": "enabled",
                "budget_tokens": 4000,
            },  # Manual budget control
        )

        # Validate response
        assert isinstance(response, str)
        assert len(response) > 0
        assert (
            "animal" in response.lower()
        ), f"Expected logical conclusion about animals in response, got: {response}"

        # Validate cost structure
        assert isinstance(cost, (int, float))
        assert cost >= 0

        print(f"✅ Claude Opus 4 reasoning test passed! Response: {response[:150]}...")

    except Exception as e:
        # If user doesn't have access to Opus 4, we'll get a specific error
        error_msg = str(e).lower()
        if any(
            access_error in error_msg
            for access_error in [
                "model not found",
                "invalid model",
                "not available",
                "access",
                "permission",
                "unauthorized",
            ]
        ):
            pytest.skip(f"Claude Opus 4 model not accessible: {e}")
        else:
            # Re-raise if it's a different error
            raise


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not found"
)
def test_call_ai_claude_37_reasoning():
    """Integration test: Test call_ai wrapper with Anthropic Claude 3.7 Sonnet with extended thinking"""

    # Test mathematical reasoning problem
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Solve mathematical problems carefully with detailed reasoning.",
        },
        {
            "role": "user",
            "content": "What is 23 × 17? Calculate this step by step using any method you prefer.",
        },
    ]

    # Test with Claude 3.7 Sonnet (returns full thinking output, not summarized)
    response, cost = call_ai(
        model="anthropic/claude-3-7-sonnet-20250219",
        messages=messages,
        temperature=1,  # Required for thinking models
        max_tokens=10000,  # Needs to be > thinking budget
        thinking={
            "type": "enabled",
            "budget_tokens": 8000,
        },  # Larger budget for detailed thinking
    )

    # Validate response
    assert isinstance(response, str)
    assert len(response) > 0
    # 23 × 17 = 391
    assert (
        "391" in response
    ), f"Expected calculation result '391' in response, got: {response}"

    # Validate cost structure
    assert isinstance(cost, (int, float))
    assert cost >= 0

    print(f"✅ Claude 3.7 Sonnet reasoning test passed! Response: {response[:150]}...")


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not found"
)
def test_anthropic_reasoning_model_detection():
    """Test that we can properly detect which Anthropic models support extended thinking"""

    from wraipperz.api.llm import AnthropicProvider

    provider = AnthropicProvider()

    # Test reasoning model detection
    assert provider.supports_extended_thinking("anthropic/claude-opus-4-20250514")
    assert provider.supports_extended_thinking("anthropic/claude-sonnet-4-20250514")
    assert provider.supports_extended_thinking("anthropic/claude-3-7-sonnet-20250219")

    # Test non-reasoning models
    assert not provider.supports_extended_thinking(
        "anthropic/claude-3-5-sonnet-20240620"
    )
    assert not provider.supports_extended_thinking("anthropic/claude-3-haiku-20240307")

    print("✅ Anthropic reasoning model detection works correctly!")


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not found"
)
def test_claude_reasoning_with_image_analysis():
    """Integration test: Test Claude reasoning model with image analysis and extended thinking"""

    # Create messages with image using MessageBuilder for complex visual reasoning
    messages = (
        MessageBuilder()
        .add_system(
            "You are a helpful assistant. Analyze images carefully and think through your observations step by step."
        )
        .add_user(
            "Look at this image carefully. What color is the main shape? Think through what you observe and provide a detailed analysis."
        )
        .add_image(str(TEST_IMAGE_PATH))
        .build()
    )

    # Test reasoning with image analysis
    response, cost = call_ai(
        model="anthropic/claude-sonnet-4-20250514",
        messages=messages,
        temperature=1,  # Required for thinking models
        max_tokens=1500,  # Needs to be > 1024 to accommodate minimum thinking budget
        thinking=True,  # Enable extended thinking for image analysis
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

    print(
        f"✅ Claude reasoning with image analysis test passed! Response: {response[:150]}..."
    )


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"), reason="Anthropic API key not found"
)
def test_claude_reasoning_parameter_constraints():
    """Test that Claude reasoning models properly handle parameter constraints"""

    messages = [
        {"role": "user", "content": "What is 12 + 8? Think through this calculation."}
    ]

    # Test with parameters that should be adjusted for thinking compatibility
    response, cost = call_ai(
        model="anthropic/claude-sonnet-4-20250514",
        messages=messages,
        temperature=0.8,  # Will be overridden to 1 for thinking models
        max_tokens=2500,  # Needs to be > thinking budget
        top_p=0.5,  # Should be adjusted to ≥ 0.95 for thinking
        top_k=40,  # Should be removed for thinking
        thinking={"type": "enabled", "budget_tokens": 2000},
    )

    # Should still work despite parameter adjustments
    assert isinstance(response, str)
    assert len(response) > 0
    assert "20" in response, f"Expected '20' in response, got: {response}"

    # Validate cost structure
    assert isinstance(cost, (int, float))
    assert cost >= 0

    print("✅ Claude reasoning parameter constraints test passed!")


# ===== AZURE OPENAI PROVIDER TESTS =====


@pytest.mark.skipif(
    not (os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY")),
    reason="Azure OpenAI credentials not found",
)
def test_azure_openai_provider_initialization():
    """Test that AzureOpenAIProvider initializes correctly with environment variables"""

    from wraipperz.api.llm import AzureOpenAIProvider

    try:
        # Should initialize successfully if env vars are set
        provider = AzureOpenAIProvider()

        # Check that it has the expected attributes
        assert hasattr(provider, "sync_client")
        assert hasattr(provider, "async_client")
        assert hasattr(provider, "supported_models")

        # Check if deployments are configured
        deployments = os.getenv("AZURE_OPENAI_DEPLOYMENTS", "")
        if deployments:
            expected_models = [
                f"azure/{d.strip()}" for d in deployments.split(",") if d.strip()
            ]
            assert provider.supported_models == expected_models

        print("✅ AzureOpenAIProvider initialization test passed!")

    except ValueError as e:
        pytest.skip(f"Azure OpenAI environment variables not properly configured: {e}")
    except Exception as e:
        pytest.fail(f"AzureOpenAIProvider initialization failed: {e}")


@pytest.mark.skipif(
    not (os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY")),
    reason="Azure OpenAI credentials not found",
)
def test_azure_openai_simple():
    """Simple Azure OpenAI test"""

    # Use the deployment name from your example
    deployment = "ai-parsing-gpt-5-chat"  # YOUR ACTUAL DEPLOYMENT NAME

    response, cost = call_ai(
        model=f"azure/{deployment}",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        temperature=0,
        max_tokens=50,
    )

    assert isinstance(response, str)
    assert len(response) > 0
    print(f"Response: {response}")


@pytest.mark.skipif(
    not (os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv("AZURE_OPENAI_API_KEY")),
    reason="Azure OpenAI credentials not found",
)
def test_azure_openai_with_pdf():
    """Integration test: Test Azure OpenAI with PDF file input"""

    # Get deployment name from environment or use default
    deployments = os.getenv("AZURE_OPENAI_DEPLOYMENTS", "ai-parsing-gpt-5-chat")
    deployment = deployments.split(",")[0].strip()

    # Create messages with PDF using MessageBuilder
    messages = (
        MessageBuilder()
        .add_system(
            "You are a helpful assistant. Extract information from documents accurately."
        )
        .add_user("What is the name of the actor mentioned in this PDF?")
        .add_pdf(str(TEST_PDF_PATH))
        .build()
    )

    try:
        # Test the call_ai wrapper function with PDF
        response, cost = call_ai(
            model=f"azure/{deployment}",
            messages=messages,
            temperature=0,
            max_tokens=150,
        )

        # Validate response
        assert isinstance(response, str)
        assert len(response) > 0
        assert (
            "tom hanks" in response.lower()
        ), f"Expected response to contain 'Tom Hanks', got: {response}"

        # Validate cost structure
        assert isinstance(cost, (int, float))
        assert cost >= 0

        print(f"✅ Azure OpenAI PDF analysis test passed! Response: {response}")

    except Exception as e:
        error_msg = str(e).lower()
        # Check if this is a model capability issue
        if any(err in error_msg for err in ["not support", "invalid", "capability"]):
            pytest.skip(
                f"Azure deployment may not support PDF inputs. "
                f"Ensure you're using a vision-capable model like gpt-4o. Error: {e}"
            )
        else:
            raise


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not found")
def test_call_ai_gemini_25_pro():
    """Integration test: Test call_ai wrapper with Gemini 2.5 Pro"""

    # Test with the stable Gemini 2.5 Pro model
    response, cost = call_ai(
        model="gemini/gemini-2.5-pro",
        messages=TEXT_MESSAGES,
        temperature=0,
        max_tokens=1500,
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

    print(f"✅ Gemini 2.5 Pro access confirmed! Response: {response[:100]}...")


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not found")
def test_call_ai_gemini_25_flash():
    """Integration test: Test call_ai wrapper with Gemini 2.5 Flash"""

    # Test with the stable Gemini 2.5 Flash model
    response, cost = call_ai(
        model="gemini/gemini-2.5-flash",
        messages=TEXT_MESSAGES,
        temperature=0,
        max_tokens=150,
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

    print(f"✅ Gemini 2.5 Flash access confirmed! Response: {response[:100]}...")


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not found")
def test_call_ai_gemini_25_with_image():
    """Integration test: Test Gemini 2.5 models with image analysis"""

    # Create messages with image using MessageBuilder
    messages = (
        MessageBuilder()
        .add_system("You are a helpful assistant. Identify the color in the image.")
        .add_user("What color is the square in this image?")
        .add_image(str(TEST_IMAGE_PATH))
        .build()
    )

    # Test with Gemini 2.5 Pro
    response, cost = call_ai(
        model="gemini/gemini-2.5-pro",
        messages=messages,
        temperature=0,
        max_tokens=1500,
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

    print(f"✅ Gemini 2.5 Pro image analysis confirmed! Response: {response[:100]}...")


@pytest.mark.parametrize(
    "model_id,model_name",
    [
        ("gemini/gemini-2.5-pro", "Gemini 2.5 Pro"),
        ("gemini/gemini-2.5-flash", "Gemini 2.5 Flash"),
        ("genai/gemini-2.5-pro", "Gemini 2.5 Pro (genai prefix)"),
        ("genai/gemini-2.5-flash", "Gemini 2.5 Flash (genai prefix)"),
    ],
)
@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not found")
def test_call_ai_gemini_25_models_parametrized(model_id, model_name):
    """Parametrized test for Gemini 2.5 models with different prefixes"""

    messages = (
        MessageBuilder()
        .add_system(
            "You are a helpful assistant. You must respond with exactly: 'GEMINI_25_TEST'"
        )
        .add_user("Please provide the required test response.")
        .build()
    )

    try:
        # Test the call_ai wrapper function
        response, cost = call_ai(
            model=model_id,
            messages=messages,
            temperature=0,
            max_tokens=150,
        )

        # Validate response
        assert isinstance(response, str)
        assert len(response) > 0
        assert (
            "GEMINI_25_TEST" in response
        ), f"Expected 'GEMINI_25_TEST', got: {response}"

        # Validate cost structure
        assert isinstance(cost, (int, float))
        assert cost >= 0

        print(f"✅ {model_name} access confirmed!")

    except Exception as e:
        error_msg = str(e).lower()
        if any(
            access_error in error_msg
            for access_error in [
                "model not found",
                "invalid model",
                "not available",
                "access",
                "permission",
            ]
        ):
            pytest.skip(f"{model_name} not accessible: {e}")
        else:
            raise


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_call_ai_gpt_5_1_instant():
    """Integration test: Test call_ai wrapper with OpenAI GPT-5.1 Instant"""

    # Test with GPT-5.1 Instant
    response, cost = call_ai(
        model="openai/gpt-5.1-2025-11-13",
        messages=TEXT_MESSAGES,
        temperature=0,
        max_tokens=150,
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

    print(f"✅ OpenAI GPT-5.1 Instant access confirmed! Response: {response[:100]}...")


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not found")
def test_call_ai_gemini_3_pro_preview():
    """Integration test: Test call_ai wrapper with Gemini 3 Pro Preview"""

    # Test with Gemini 3 Pro Preview
    response, cost = call_ai(
        model="gemini/gemini-3-pro-preview",
        messages=TEXT_MESSAGES,
        temperature=0,
        max_tokens=150,
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

    print(f"✅ Gemini 3 Pro Preview access confirmed! Response: {response[:100]}...")


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="Google API key not found")
def test_call_ai_gemini_3_pro_safety_check():
    """Integration test: Test Gemini 3 Pro with sensitive content to verify safety settings"""

    # Content that previously triggered safety filters (kidnapping/violence context)
    sensitive_content = """
Piggy: 宿主冷静啊 杀了他 咱们的任务就失败了 你现在已经不是三千位面的女主神了 你必须完成任务 才能恢复自己的肉身和灵识 过几天他要是被齐媛救了 喜欢上她 那任务就失败了

The line to translate and adapt perfectly is:

让你跑 才卖掉没几天就逃出来了 老子的银子他们都还没付清呢 老子养条狗 都会给我叼块馒头回来 你呢 吃老子的喝老子的就算了 还一个子都赚不回来 老子留着你有什么用啊？啊？呸

which was said by Kidnapper Boss.
"""

    messages = [
        {
            "role": "system",
            "content": "You are a professional translator. Translate the text accurately. output the translation directly in English.",
        },
        {"role": "user", "content": sensitive_content},
    ]

    try:
        # Test with Gemini 3 Pro Preview
        response, cost = call_ai(
            model="gemini/gemini-3-pro-preview",
            messages=messages,
            temperature=0,
            max_tokens=5000,
        )

        # Validate response - if blocked, response might be None or raise an error depending on implementation
        assert isinstance(response, str)
        assert len(response) > 0

        print(f"✅ Gemini 3 Pro Safety Check Passed! Response: {response[:100]}...")

    except Exception as e:
        pytest.fail(
            f"Gemini 3 Pro failed on sensitive content (likely safety block): {e}"
        )
