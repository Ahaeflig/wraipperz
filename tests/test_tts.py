# AI GENERATED CODE BEWARE
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from wraipperz.api.tts import (
    CartesiaTTSProvider,
    ElevenLabsTTSProvider,
    MiniMaxiTTSProvider,
    OpenAIRealtimeTTSProvider,
    OpenAITTSProvider,
    TTSManager,
    TTSProvider,
    TTSRateLimitError,
    create_tts_manager,
)

# Create test_assets directory if it doesn't exist
TEST_ASSETS_DIR = Path(__file__).parent / "test_assets"
TEST_ASSETS_DIR.mkdir(exist_ok=True)

# Test output path
TEST_OUTPUT_PATH = TEST_ASSETS_DIR / "test_output.mp3"


class MockTTSProvider(TTSProvider):
    """Mock TTS provider for testing"""

    def __init__(self):
        self.generate_speech_called = False
        self.convert_speech_called = False
        self.find_similar_voices_called = False
        self.add_sharing_voice_called = False
        self.last_text = None
        self.last_output_path = None
        self.last_voice = None
        self.last_kwargs = None

    def generate_speech(self, text, output_path, voice, **kwargs):
        self.generate_speech_called = True
        self.last_text = text
        self.last_output_path = output_path
        self.last_voice = voice
        self.last_kwargs = kwargs

        # Create a dummy output file
        with open(output_path, "wb") as f:
            f.write(b"dummy audio data")

        return {"status": "success", "provider": "mock"}

    def convert_speech(self, input_path, output_path, voice, **kwargs):
        self.convert_speech_called = True
        return {"status": "success", "provider": "mock"}

    def find_similar_voices(self, audio_file, similarity_threshold=None, top_k=None):
        self.find_similar_voices_called = True
        return [
            {"voice_id": "voice1", "similarity": 0.9},
            {"voice_id": "voice2", "similarity": 0.8},
        ]

    def add_sharing_voice(self, public_user_id, voice_id, new_name):
        self.add_sharing_voice_called = True
        return {"status": "success", "voice_id": voice_id, "name": new_name}


@pytest.fixture
def mock_provider():
    return MockTTSProvider()


@pytest.fixture
def tts_manager(mock_provider):
    manager = TTSManager()
    manager.add_provider("mock", mock_provider)
    return manager


def test_tts_manager_initialization():
    """Test that TTSManager initializes correctly"""
    manager = TTSManager()
    assert hasattr(manager, "providers")
    assert isinstance(manager.providers, dict)
    assert len(manager.providers) == 0


def test_add_provider(mock_provider):
    """Test adding a provider to the TTSManager"""
    manager = TTSManager()
    manager.add_provider("test_provider", mock_provider)

    assert "test_provider" in manager.providers
    assert manager.providers["test_provider"] == mock_provider


def test_generate_speech(tts_manager, mock_provider):
    """Test generating speech through the TTSManager"""
    test_text = "Hello, this is a test."

    result = tts_manager.generate_speech(
        provider_name="mock",
        text=test_text,
        output_path=str(TEST_OUTPUT_PATH),
        voice="test_voice",
        speed=1.2,
    )

    assert mock_provider.generate_speech_called
    assert mock_provider.last_text == test_text
    assert mock_provider.last_output_path == str(TEST_OUTPUT_PATH)
    assert mock_provider.last_voice == "test_voice"
    assert mock_provider.last_kwargs.get("speed") == 1.2
    assert result["status"] == "success"
    assert result["provider"] == "mock"


def test_generate_speech_provider_not_found(tts_manager):
    """Test error handling when provider is not found"""
    with pytest.raises(ValueError, match="Provider .* not found"):
        tts_manager.generate_speech(
            provider_name="nonexistent",
            text="Test",
            output_path=str(TEST_OUTPUT_PATH),
            voice="test_voice",
        )


def test_convert_speech(tts_manager, mock_provider):
    """Test converting speech through the TTSManager"""
    input_path = TEST_ASSETS_DIR / "test_input.mp3"

    # Create a dummy input file
    with open(input_path, "wb") as f:
        f.write(b"dummy input audio data")

    result = tts_manager.convert_speech(
        provider_name="mock",
        input_path=str(input_path),
        output_path=str(TEST_OUTPUT_PATH),
        voice="target_voice",
    )

    assert mock_provider.convert_speech_called
    assert result["status"] == "success"
    assert result["provider"] == "mock"

    # Clean up
    if input_path.exists():
        input_path.unlink()


def test_find_similar_voices(tts_manager, mock_provider):
    """Test finding similar voices through the TTSManager"""
    audio_file = TEST_ASSETS_DIR / "test_audio.mp3"

    # Create a dummy audio file
    with open(audio_file, "wb") as f:
        f.write(b"dummy audio data")

    result = tts_manager.find_similar_voices(
        provider_name="mock",
        audio_file=str(audio_file),
        similarity_threshold=0.7,
        top_k=2,
    )

    assert mock_provider.find_similar_voices_called
    assert len(result) == 2
    assert result[0]["voice_id"] == "voice1"
    assert result[0]["similarity"] == 0.9

    # Clean up
    if audio_file.exists():
        audio_file.unlink()


def test_add_sharing_voice(tts_manager, mock_provider):
    """Test adding a shared voice through the TTSManager"""
    result = tts_manager.add_sharing_voice(
        provider_name="mock",
        public_user_id="user123",
        voice_id="voice456",
        new_name="My Shared Voice",
    )

    assert mock_provider.add_sharing_voice_called
    assert result["status"] == "success"
    assert result["voice_id"] == "voice456"
    assert result["name"] == "My Shared Voice"


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_openai_provider_initialization():
    """Test OpenAI TTS provider initialization"""
    provider = OpenAITTSProvider()
    assert provider.client is not None


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_openai_provider_generate_speech():
    """Test OpenAI TTS provider generate_speech method with actual API call"""
    provider = OpenAITTSProvider()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        output_path = temp_file.name

        _ = provider.generate_speech(
            text="This is a test of the OpenAI text to speech API.",
            output_path=output_path,
            voice="alloy",
            model="tts-1",
        )

        # Verify file was created and has content
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

        # Clean up
        Path(output_path).unlink(missing_ok=True)


@pytest.mark.skipif(
    not os.getenv("ELEVENLABS_API_KEY"), reason="ElevenLabs API key not found"
)
def test_elevenlabs_provider_generate_speech():
    """Test ElevenLabs TTS provider generate_speech method with actual API call"""
    provider = ElevenLabsTTSProvider()

    # Get available voices first to use a valid voice ID
    voices = provider.list_voices()
    if not voices:
        pytest.skip("No ElevenLabs voices available")

    voice_id = voices[0]["voice_id"]  # Use the first available voice

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        output_path = temp_file.name

        _ = provider.generate_speech(
            text="This is a test of the ElevenLabs text to speech API.",
            output_path=output_path,
            voice=voice_id,
            model_id="eleven_multilingual_v2",
        )

        # Verify file was created and has content
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

        # Clean up
        Path(output_path).unlink(missing_ok=True)


@pytest.mark.skipif(
    not (os.getenv("T2A_API_KEY") and os.getenv("MINIMAXI_GROUP_ID")),
    reason="MiniMaxi API key or group ID not found",
)
def test_minimaxi_provider_generate_speech():
    """Test MiniMaxi TTS provider generate_speech method with actual API call"""
    provider = MiniMaxiTTSProvider()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        output_path = temp_file.name

        _ = provider.generate_speech(
            text="This is a test of the MiniMaxi text to speech API.",
            output_path=output_path,
            voice="Calm_Woman",
            speed=1.0,
            volume=1.0,
            pitch=0.0,
        )

        # Verify file was created and has content
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

        # Clean up
        Path(output_path).unlink(missing_ok=True)


@pytest.mark.skipif(
    not os.getenv("CARTESIA_API_KEY"), reason="Cartesia API key not found"
)
def test_cartesia_provider_generate_speech():
    """Test Cartesia TTS provider generate_speech method with actual API call"""
    provider = CartesiaTTSProvider()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        output_path = temp_file.name

        _ = provider.generate_speech(
            text="This is a test of the Cartesia text to speech API.",
            output_path=output_path,
            voice="default",  # Use default voice or replace with a valid voice ID
        )

        # Verify file was created and has content
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

        # Clean up
        Path(output_path).unlink(missing_ok=True)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_openai_realtime_provider_generate_speech():
    """Test OpenAI Realtime TTS provider generate_speech method with actual API call"""
    provider = OpenAIRealtimeTTSProvider()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        output_path = temp_file.name

        _ = provider.generate_speech(
            text="This is a test of the OpenAI realtime text to speech API.",
            output_path=output_path,
            voice="alloy",
            model="tts-1",
        )

        # Verify file was created and has content
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

        # Clean up
        Path(output_path).unlink(missing_ok=True)


def test_create_tts_manager():
    """Test the create_tts_manager factory function"""
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test_openai_key",
            "ELEVENLABS_API_KEY": "test_elevenlabs_key",
            "T2A_API_KEY": "test_minimaxi_key",
            "MINIMAXI_GROUP_ID": "test_group_id",
            "CARTESIA_API_KEY": "test_cartesia_key",
        },
    ):
        manager = create_tts_manager()

        # Check that all providers were added
        assert "openai" in manager.providers
        assert "openai_realtime" in manager.providers
        assert "elevenlabs" in manager.providers
        assert "minimaxi" in manager.providers
        assert "cartesia" in manager.providers

        # Check provider types
        assert isinstance(manager.providers["openai"], OpenAITTSProvider)
        assert isinstance(
            manager.providers["openai_realtime"], OpenAIRealtimeTTSProvider
        )
        assert isinstance(manager.providers["elevenlabs"], ElevenLabsTTSProvider)
        assert isinstance(manager.providers["minimaxi"], MiniMaxiTTSProvider)
        assert isinstance(manager.providers["cartesia"], CartesiaTTSProvider)


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not found")
def test_integration_openai():
    """Integration test for OpenAI TTS provider"""
    manager = create_tts_manager()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        output_path = temp_file.name

        _ = manager.generate_speech(
            provider_name="openai",
            text="This is an integration test.",
            output_path=output_path,
            voice="alloy",
        )

        # Check that the file was created and has content
        assert Path(output_path).exists()
        assert Path(output_path).stat().st_size > 0

        # Clean up
        Path(output_path).unlink(missing_ok=True)


def test_retry_mechanism():
    """Test that the retry mechanism works correctly"""

    # Create a provider that fails with rate limit error first, then succeeds
    class RetryTestProvider(TTSProvider):
        def __init__(self):
            self.call_count = 0

        def generate_speech(self, text, output_path, voice, **kwargs):
            self.call_count += 1
            if self.call_count == 1:
                raise TTSRateLimitError("Rate limit exceeded")

            # Create a dummy output file on second attempt
            with open(output_path, "wb") as f:
                f.write(b"dummy audio data")

            return {"status": "success", "attempt": self.call_count}

    # Create manager with the test provider
    manager = TTSManager()
    provider = RetryTestProvider()
    manager.add_provider("retry_test", provider)

    # Test that it retries and succeeds
    result = manager.generate_speech(
        provider_name="retry_test",
        text="Test retry",
        output_path=str(TEST_OUTPUT_PATH),
        voice="test_voice",
    )

    assert provider.call_count == 2
    assert result["status"] == "success"
    assert result["attempt"] == 2

    # Clean up
    if TEST_OUTPUT_PATH.exists():
        TEST_OUTPUT_PATH.unlink()
