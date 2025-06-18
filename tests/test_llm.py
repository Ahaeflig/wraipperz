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


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_o3_pro_model_access():
    """Test if user has access to o3-pro model"""

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {"role": "user", "content": "What is 5 + 7?"},
    ]

    try:
        response, cost = call_ai(
            model="openai/o3-pro",
            messages=messages,
            max_tokens=50,
        )

        # Should work without error if user has access
        assert isinstance(response, str)
        assert len(response) > 0
        assert "12" in response

        # Validate cost structure
        assert isinstance(cost, (int, float))
        assert cost >= 0

        print(f"✅ o3-pro model access confirmed! Response: {response}")

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
            pytest.skip(f"o3-pro model not accessible: {e}")
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
