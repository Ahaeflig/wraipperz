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
    """OpenAI ASR provider using Whisper and GPT-4o based transcription models"""

    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def transcribe(
        self,
        audio_path: str | Path,
        language: Optional[str] = None,
        model: str = "whisper-1",
        response_format: str = "verbose_json",
        prompt: Optional[str] = None,
        timestamp_granularities: Optional[List[str]] = None,
        **kwargs,
    ) -> ASRResult:
        """
        Transcribe audio using OpenAI's Whisper or GPT-4o based transcription models

        Args:
            audio_path: Path to audio file
            language: Optional ISO-639-1 language code
            model: Model to use ("whisper-1", "gpt-4o-mini-transcribe", or "gpt-4o-transcribe")
            response_format: Format of the response ("json", "text", "srt", "verbose_json", "vtt")
                             Note: gpt-4o models only support "json" or "text"
            prompt: Optional text to guide the model's style or help it pick up on certain words
                    Note: prompt is not supported by gpt-4o models
            timestamp_granularities: Optional list of timestamp granularity options (["word", "segment"])
                                     Note: requires "verbose_json" and is only supported by whisper-1
            **kwargs: Additional arguments passed to OpenAI API

        Returns:
            ASRResult object containing transcription and timing information
        """
        # Validate format compatibility with model
        if model in ["gpt-4o-mini-transcribe", "gpt-4o-transcribe"]:
            if response_format not in ["json", "text"]:
                response_format = (
                    "json"  # Default to json for gpt-4o models if unsupported format
                )
                print(
                    f"Warning: {model} only supports 'json' or 'text' response formats. Using 'json'."
                )

            if prompt:
                print(
                    f"Warning: {model} does not support the 'prompt' parameter. It will be ignored."
                )
                prompt = None

            if timestamp_granularities:
                print(
                    f"Warning: {model} does not support the 'timestamp_granularities' parameter. It will be ignored."
                )
                timestamp_granularities = None

        # For word-level timestamps, we need verbose_json
        if timestamp_granularities and "word" in timestamp_granularities:
            if response_format != "verbose_json":
                response_format = "verbose_json"
                print(
                    "Setting response_format to 'verbose_json' to support word-level timestamps"
                )

        with open(audio_path, "rb") as audio_file:
            # Prepare request parameters
            request_params = {
                "file": audio_file,
                "model": model,
                "response_format": response_format,
            }

            # Add optional parameters if provided
            if language:
                request_params["language"] = language

            if prompt and model == "whisper-1":
                request_params["prompt"] = prompt

            if (
                timestamp_granularities
                and model == "whisper-1"
                and response_format == "verbose_json"
            ):
                request_params["timestamp_granularities"] = timestamp_granularities

            # Add any additional parameters from kwargs
            for key, value in kwargs.items():
                request_params[key] = value

            try:
                # Make API request
                response = self.client.audio.transcriptions.create(**request_params)

                # Handle different response formats
                if response_format == "verbose_json":
                    # For verbose_json format, response is an object with text, words, etc.
                    return ASRResult(
                        text=response.text,
                        words=[
                            {"word": w.word, "start": w.start, "end": w.end}
                            for w in response.words
                        ],
                        duration=response.duration,
                        language=response.language,
                    )
                elif response_format == "json":
                    # For json format, response just has text
                    return ASRResult(
                        text=response.text,
                        words=[],  # No word-level timing info
                        duration=0.0,  # No duration info
                        language=language or "en",
                    )
                elif response_format == "text":
                    # For text format, response is just a string
                    return ASRResult(
                        text=response,
                        words=[],  # No word-level timing info
                        duration=0.0,  # No duration info
                        language=language or "en",
                    )
                else:
                    # For other formats, create a basic result
                    return ASRResult(
                        text=str(response),
                        words=[],  # No word-level timing info
                        duration=0.0,  # No duration info
                        language=language or "en",
                    )

            except Exception as e:
                raise ASRError(f"OpenAI API error: {str(e)}")


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

    if len(manager.providers) == 0:
        raise ValueError(
            "No ASR providers found, please add OPENAI_API_KEY or DG_API_KEY to your environment variables"
        )

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
