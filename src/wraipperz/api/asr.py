import abc
from pathlib import Path
from typing import Dict, List, Optional
from openai import OpenAI
import os
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions

load_dotenv(override=True)


class ASRError(Exception):
    """Base exception for ASR-related errors"""

    pass


class ASRResult:
    """Container for ASR results with standardized format"""

    def __init__(
        self,
        text: str,
        words: List[Dict[str, float]],
        duration: float,
        language: str = "english",
    ):
        self.text = text
        self.words = words
        self.duration = duration
        self.language = language

    def to_elevenlabs_alignment(self) -> Dict:
        """Convert ASR result to ElevenLabs-style alignment format"""
        characters = []
        char_starts = []
        char_ends = []

        # Process each word and its characters
        pos = 0
        for word_info in self.words:
            word = word_info["word"]
            word_start = word_info["start"]
            word_end = word_info["end"]
            word_duration = word_end - word_start

            # Calculate time per character within this word
            char_duration = word_duration / len(word)

            for i, char in enumerate(word):
                characters.append(char)
                char_start = word_start + (i * char_duration)
                char_end = char_start + char_duration
                char_starts.append(char_start)
                char_ends.append(char_end)

            # Add space after word (except for last word)
            if pos < len(self.words) - 1:
                next_word_start = self.words[pos + 1]["start"]
                space_duration = next_word_start - word_end

                characters.append(" ")
                char_starts.append(word_end)
                char_ends.append(word_end + space_duration)

            pos += 1

        return {
            "characters": characters,
            "character_start_times_seconds": char_starts,
            "character_end_times_seconds": char_ends,
        }


class ASRProvider(abc.ABC):
    """Abstract base class for ASR providers"""

    @abc.abstractmethod
    def transcribe(
        self, audio_path: str | Path, language: Optional[str] = None, **kwargs
    ) -> ASRResult:
        """Transcribe audio file and return structured result"""
        pass


class OpenAIASRProvider(ASRProvider):
    """OpenAI Whisper ASR provider"""

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def transcribe(
        self, audio_path: str | Path, language: Optional[str] = None, **kwargs
    ) -> ASRResult:
        """
        Transcribe audio using OpenAI's Whisper model

        Args:
            audio_path: Path to audio file
            language: Optional ISO-639-1 language code
            **kwargs: Additional arguments passed to OpenAI API

        Returns:
            ASRResult object containing transcription and timing information
        """
        with open(audio_path, "rb") as audio_file:
            # Use requests directly to get raw JSON response with word timestamps
            import requests

            headers = {"Authorization": f"Bearer {self.client.api_key}"}

            files = {
                "file": ("audio.wav", audio_file, "audio/wav"),
                "model": (None, "whisper-1"),
                "response_format": (None, "verbose_json"),
            }

            if language:
                files["language"] = (None, language)

            # Add timestamp_granularities parameter
            files["timestamp_granularities[]"] = (None, "word")

            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                files=files,
            )

            if response.status_code != 200:
                raise ASRError(f"OpenAI API error: {response.text}")

            data = response.json()

            return ASRResult(
                text=data["text"],
                words=data["words"],
                duration=data["duration"],
                language=data["language"],
            )


class DeepgramASRProvider(ASRProvider):
    """Deepgram ASR provider"""

    def __init__(self, api_key: str = None):
        self.client = DeepgramClient(api_key or os.getenv("DG_API_KEY"))

    def transcribe(
        self, audio_path: str | Path, language: Optional[str] = None, **kwargs
    ) -> ASRResult:
        """
        Transcribe audio using Deepgram's API

        Args:
            audio_path: Path to audio file
            language: Optional language code
            **kwargs: Additional arguments passed to Deepgram API

        Returns:
            ASRResult object containing transcription and timing information
        """
        try:
            # Configure options
            options = PrerecordedOptions(
                model="nova-2",  # Using Nova-2 model for best accuracy
                language=language or "en",
                smart_format=True,
            )

            # Open and read audio file
            with open(audio_path, "rb") as audio:
                source = {"buffer": audio.read()}

            # Get transcription
            response = self.client.listen.prerecorded.v("1").transcribe_file(
                source, options
            )

            # Extract words with timing information
            words = []
            for word in response.results.channels[0].alternatives[0].words:
                words.append({"word": word.word, "start": word.start, "end": word.end})

            # Get full text and duration
            text = response.results.channels[0].alternatives[0].transcript
            duration = response.metadata.duration

            return ASRResult(
                text=text, words=words, duration=duration, language=language or "en"
            )

        except Exception as e:
            raise ASRError(f"Deepgram API error: {str(e)}")


class ASRManager:
    """Manager class for handling multiple ASR providers"""

    def __init__(self):
        self.providers = {}

    def add_provider(self, name: str, provider: ASRProvider):
        self.providers[name] = provider

    def transcribe(
        self,
        provider_name: str,
        audio_path: str | Path,
        language: Optional[str] = None,
        **kwargs,
    ) -> ASRResult:
        """
        Transcribe audio using specified provider

        Args:
            provider_name: Name of ASR provider to use
            audio_path: Path to audio file
            language: Optional ISO-639-1 language code
            **kwargs: Additional provider-specific arguments

        Returns:
            ASRResult object containing transcription and timing information
        """
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not found")

        return self.providers[provider_name].transcribe(
            audio_path, language=language, **kwargs
        )


def create_asr_manager() -> ASRManager:
    """Create and configure ASR manager with available providers"""
    manager = ASRManager()

    if os.getenv("OPENAI_API_KEY"):
        manager.add_provider("openai", OpenAIASRProvider())

    if os.getenv("DG_API_KEY"):
        manager.add_provider("deepgram", DeepgramASRProvider())

    return manager


if __name__ == "__main__":
    asr_manager = create_asr_manager()

    test_file = "tmp/test.wav"
    if os.path.exists(test_file):
        result = asr_manager.transcribe("openai", test_file)

        # Print word timings
        print("\nWord timings:")
        for word in result.words:
            print(f"{word['word']}: {word['start']:.2f}s - {word['end']:.2f}s")

        # Print character-level alignment
        alignment = result.to_elevenlabs_alignment()
        print("\nCharacter timings:")
        for i, char in enumerate(alignment["characters"]):
            if char not in [" ", "\n"]:
                start = alignment["character_start_times_seconds"][i]
                end = alignment["character_end_times_seconds"][i]
                print(f"'{char}': {start:.3f}s - {end:.3f}s")
