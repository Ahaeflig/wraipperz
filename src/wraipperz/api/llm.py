import abc
import base64
import io
import json
import mimetypes
import os
import threading
import time

# from tokencost import calculate_prompt_cost, calculate_completion_cost
from pathlib import Path
from typing import Any, Dict, List, Tuple

import anthropic
import requests
from dotenv import load_dotenv

# import google.generativeai as genai
from google import genai
from google.api_core import exceptions as google_exceptions
from google.genai import types
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    OpenAI,
    RateLimitError,
)
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# AWS Bedrock imports
try:
    import boto3
    from botocore.exceptions import BotoCoreError, ClientError

    BEDROCK_AVAILABLE = True
except ImportError:
    BEDROCK_AVAILABLE = False
    boto3 = None
    BotoCoreError = Exception
    ClientError = Exception

# Vertex AI imports
try:
    from anthropic import AnthropicVertex

    VERTEX_AVAILABLE = True
except ImportError:
    VERTEX_AVAILABLE = False
    AnthropicVertex = None

from .messages import Message

load_dotenv(override=True)


class AIProvider(abc.ABC):
    @abc.abstractmethod
    def call_ai(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
        model: str,
        **kwargs: Dict[str, Any],
    ) -> str:
        pass

    @abc.abstractmethod
    async def call_ai_async(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
        model: str,
        **kwargs: Dict[str, Any],
    ) -> str:
        pass

    @abc.abstractmethod
    def generate(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
        model: str,
        **kwargs: Dict[str, Any],
    ) -> str:
        pass

    @abc.abstractmethod
    async def generate_async(
        self,
        messages: List[Message],
        temperature: float,
        max_tokens: int,
        model: str,
        **kwargs: Dict[str, Any],
    ) -> str:
        pass


class LMStudioProvider(AIProvider):
    supported_models = ["lmstudio"]

    def __init__(self, ip_address="localhost", port=1234):
        self.base_url = f"http://{ip_address}:{port}/v1"

    def call_ai(self, messages, temperature, max_tokens, model=None, **kwargs):
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }
            if model:
                data["model"] = model

            response = requests.post(
                f"{self.base_url}/chat/completions", headers=headers, json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            raise e

    async def call_ai_async(
        self, messages, temperature, max_tokens, model=None, **kwargs
    ):
        # For simplicity, we'll use the synchronous version in an async context
        # In a real-world scenario, you might want to use an async HTTP client
        return self.call_ai(messages, temperature, max_tokens, model, **kwargs)

    def generate(self, messages, temperature, max_tokens, model=None, **kwargs):
        raise NotImplementedError("This provider does not support image generation")

    async def generate_async(
        self, messages, temperature, max_tokens, model=None, **kwargs
    ):
        raise NotImplementedError("This provider does not support image generation")


class OpenAIProvider(AIProvider):
    supported_models = [
        "openai/gpt-5.1-instant",
        "openai/gpt-5.1-2025-11-13",
        "openai/gpt-5.1-thinking",
        "openai/gpt-5.1",
        "openai/o1-mini-2024-09-12",
        "openai/o1-mini",
        "openai/gpt-4",
        "openai/gpt-4o-mini-2024-07-18",
        "openai/gpt-4o-2024-11-20",
        "openai/gpt-4o-2024-05-13",
        "openai/o1-preview",
        "openai/o1-preview-2024-09-12",
        "openai/o3-mini",
        "openai/o3-mini-2025-01-31",
        "openai/gpt-4o-mini",
        "openai/gpt-3.5-turbo-instruct-0914",
        "openai/gpt-4o-mini-search-preview",
        "openai/gpt-3.5-turbo-1106",
        "openai/gpt-4o-search-preview",
        "openai/gpt-4-turbo",
        "openai/gpt-3.5-turbo-instruct",
        "openai/o1-2024-12-17",
        "openai/o1",
        "openai/gpt-3.5-turbo-0125",
        "openai/gpt-4o-2024-08-06",
        "openai/gpt-3.5-turbo",
        "openai/gpt-4-turbo-2024-04-09",
        "openai/gpt-4o-realtime-preview",
        "openai/gpt-3.5-turbo-16k",
        "openai/gpt-4o",
        "openai/text-embedding-3-small",
        "openai/chatgpt-4o-latest",
        "openai/gpt-4-1106-preview",
        "openai/text-embedding-ada-002",
        "openai/gpt-4-0613",
        "openai/gpt-4.5-preview",
        "openai/gpt-4.5-preview-2025-02-27",
        "openai/gpt-4o-search-preview-2025-03-11",
        "openai/gpt-4-0125-preview",
        "openai/gpt-4-turbo-preview",
        "openai/gpt-4.1-mini-2025-04-14",
        "openai/gpt-4.1",
        "openai/gpt-4.1-2025-04-14",
        "openai/o4-mini-2025-04-16",
    ]

    def __init__(self, api_key=None):
        self.sync_client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.async_client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

        try:
            # Get models from API - only include chat/text generation models
            api_models = []
            models_response = self.sync_client.models.list()

            for model in models_response.data:
                # Filter for models that support text generation/chat completions
                # Exclude embedding, image, audio, and other specialized models
                model_id = model.id
                if any(
                    excluded in model_id.lower()
                    for excluded in [
                        "embed",
                        "whisper",
                        "tts",
                        "dall-e",
                        "davinci-002",
                        "babbage-002",
                    ]
                ):
                    continue

                # Add openai/ prefix to match our naming convention
                prefixed_model = f"openai/{model_id}"
                api_models.append(prefixed_model)

            # Add models that might not be returned by API (limited access/beta models)
            beta_models = [
                "openai/o3",
                "openai/o3-2025-04-16",
                "openai/o3-pro",
                "openai/o3-pro-2025-03-19",
                "openai/o4-mini",
                "openai/o4-mini-2025-04-16",
            ]
            api_models.extend(beta_models)

            # Add the API models to our supported models, avoiding duplicates
            if api_models:
                existing_models = set(self.supported_models)
                new_models = [m for m in api_models if m not in existing_models]
                self.supported_models.extend(new_models)

        except Exception as e:
            print(f"Warning: Could not fetch OpenAI models from API: {e}")
            # Continue with hardcoded list as fallback

    def call_ai(
        self, messages, temperature, max_tokens, model="openai/gpt-4o", **kwargs
    ):
        try:
            # Check if this is a reasoning model and handle accordingly
            if self._is_reasoning_model(model):
                api_params = self._prepare_reasoning_params(
                    messages, temperature, max_tokens, model, **kwargs
                )
                response = self.sync_client.chat.completions.create(**api_params)
            else:
                # Standard model handling
                prepared_messages = self._prepare_messages(messages)
                response = self.sync_client.chat.completions.create(
                    model=model.split("/")[-1],
                    messages=prepared_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            return response.choices[0].message.content
        except Exception as e:
            raise e

    async def call_ai_async(
        self, messages, temperature, max_tokens, model="openai/gpt-4o", **kwargs
    ):
        try:
            # Check if this is a reasoning model and handle accordingly
            if self._is_reasoning_model(model):
                api_params = self._prepare_reasoning_params(
                    messages, temperature, max_tokens, model, **kwargs
                )
                response = await self.async_client.chat.completions.create(**api_params)
            else:
                # Standard model handling
                prepared_messages = self._prepare_messages(messages)
                response = await self.async_client.chat.completions.create(
                    model=model.split("/")[-1],
                    messages=prepared_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
            return response.choices[0].message.content
        except Exception as e:
            raise e

    def generate(self, messages, temperature, max_tokens, model=None, **kwargs):
        raise NotImplementedError("This provider does not support image generation")

    async def generate_async(
        self, messages, temperature, max_tokens, model=None, **kwargs
    ):
        raise NotImplementedError("This provider does not support image generation")

    def _is_reasoning_model(self, model):
        """Check if the model is a reasoning model (o1, o3 series)"""
        model_name = model.split("/")[-1].lower()
        reasoning_models = [
            "o1",
            "o1-mini",
            "o1-preview",
            "o1-2024-12-17",
            "o3",
            "o3-mini",
            "o3-pro",
            "o4-mini",
            "gpt-5.1-thinking",
            "gpt-5",
        ]
        return any(model_name.startswith(rm) for rm in reasoning_models)

    def _prepare_reasoning_params(
        self, messages, temperature, max_tokens, model, **kwargs
    ):
        """Prepare parameters for reasoning models with their special requirements"""
        model_name = model.split("/")[-1]

        # Handle system/developer messages based on model capabilities
        prepared_messages = []
        for message in messages:
            if message["role"] == "system":
                # Convert system to developer for newer models, or to user for older ones
                if model_name in ["o1-mini", "o1-preview", "gpt-5"]:
                    # These models don't support system or developer messages
                    prepared_messages.append(
                        {"role": "user", "content": message["content"]}
                    )
                else:
                    # Newer o1, o3 models support developer messages
                    prepared_messages.append(
                        {"role": "developer", "content": message["content"]}
                    )
            else:
                # For regular messages, prepare them properly (handle images if supported)
                if model_name not in [
                    "o1-mini",
                    "o1-preview",
                    "o3-mini",
                ] and isinstance(message.get("content"), list):
                    # Models that support vision need proper message preparation
                    prepared_message = {"role": message["role"], "content": []}
                    for item in message["content"]:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            image_data = self._process_media(item["image_url"]["url"])
                            mime_type, _ = mimetypes.guess_type(
                                item["image_url"]["url"]
                            )
                            mime_type = mime_type or "image/jpeg"
                            prepared_message["content"].append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_data}"
                                    },
                                }
                            )
                        else:
                            prepared_message["content"].append(item)
                    prepared_messages.append(prepared_message)
                else:
                    prepared_messages.append(message)

        # Remove unsupported parameters for reasoning models
        cleaned_kwargs = {}
        unsupported_params = {
            "temperature",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "top_logprobs",
            "logit_bias",
        }

        for key, value in kwargs.items():
            if key not in unsupported_params:
                cleaned_kwargs[key] = value

        # Use max_completion_tokens instead of max_tokens
        api_params = {
            "model": model_name,
            "messages": prepared_messages,
            "max_completion_tokens": max_tokens,
            **cleaned_kwargs,
        }

        # Add reasoning_effort if supported and not already provided
        if (
            model_name not in ["o1-mini", "o1-preview"]
            and "reasoning_effort" not in cleaned_kwargs
        ):
            api_params["reasoning_effort"] = "medium"

        return api_params

    def _process_media(self, media_path):
        if isinstance(media_path, (str, Path)):
            path = Path(media_path)
            if path.is_file():
                with open(path, "rb") as media_file:
                    return base64.b64encode(media_file.read()).decode("utf-8")
            else:
                raise ValueError(f"File not found: {media_path}")
        elif isinstance(media_path, bytes):
            return base64.b64encode(media_path).decode("utf-8")
        else:
            raise ValueError(f"Unsupported media format: {type(media_path)}")

    def _process_image(self, image_path):
        if isinstance(image_path, (str, Path)):
            # Handle both string paths and Path objects
            path = Path(image_path)
            if path.is_file():
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")
            else:
                raise ValueError(f"File not found: {image_path}")
        elif isinstance(image_path, bytes):
            # Assume it's image data
            return base64.b64encode(image_path).decode("utf-8")
        elif isinstance(image_path, Image.Image):
            # It's a PIL Image
            buffered = io.BytesIO()
            image_path.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image format: {type(image_path)}")

    def _prepare_messages(self, messages):
        prepared_messages = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                prepared_messages.append({"role": message["role"], "content": content})
            elif isinstance(content, list):
                prepared_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image_url":
                        image_data = self._process_media(item["image_url"]["url"])
                        mime_type, _ = mimetypes.guess_type(item["image_url"]["url"])
                        # Default to jpeg if we can't determine the type
                        mime_type = mime_type or "image/jpeg"
                        prepared_content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_data}"
                                },
                            }
                        )
                    elif isinstance(item, dict) and item.get("type") == "input_url":
                        # Handle PDF files
                        pdf_url = item["input_url"]["url"]
                        if pdf_url.startswith("data:"):
                            # Already base64 encoded
                            prepared_content.append(item)
                        elif pdf_url.startswith(("http://", "https://")):
                            # URL - OpenAI can handle these directly
                            prepared_content.append(item)
                        else:
                            # Local file - need to encode
                            pdf_data = self._process_media(pdf_url)
                            prepared_content.append(
                                {
                                    "type": "input_url",
                                    "input_url": {
                                        "url": f"data:application/pdf;base64,{pdf_data}"
                                    },
                                }
                            )
                    elif isinstance(item, dict) and item.get("type") in [
                        "video_url",
                        "audio_url",
                    ]:
                        # Handle other media types if needed, not supported yet
                        # prepared_content.append(item)
                        pass
                    else:
                        prepared_content.append(item)
                prepared_messages.append(
                    {"role": message["role"], "content": prepared_content}
                )
        return prepared_messages


class AzureOpenAIProvider(AIProvider):
    """
    Azure OpenAI provider using the OpenAI SDK with Azure endpoints.

    Models are specified as "azure/deployment-name" where deployment-name
    is the name of your Azure OpenAI deployment.

    Required environment variables:
    - AZURE_OPENAI_ENDPOINT: Your Azure OpenAI endpoint URL
    - AZURE_OPENAI_API_KEY: Your Azure OpenAI API key

    Optional:
    - AZURE_OPENAI_DEPLOYMENTS: Comma-separated list of deployment names
    """

    def __init__(self, endpoint=None, api_key=None, api_version=None):
        endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint

        if not endpoint:
            raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")
        if not api_key:
            raise ValueError("AZURE_OPENAI_API_KEY environment variable is required")

        # Just use the endpoint as provided - don't modify it!
        self.sync_client = OpenAI(
            base_url=endpoint,
            api_key=api_key,
        )
        self.async_client = AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key,
        )

        # Azure deployments are dynamic, so we'll accept any model with azure/ prefix
        self.supported_models = []

        # Get configured deployments from environment (optional)
        # Format: AZURE_OPENAI_DEPLOYMENTS="deployment1,deployment2,deployment3"
        deployments = os.getenv("AZURE_OPENAI_DEPLOYMENTS", "")
        if deployments:
            self.supported_models = [
                f"azure/{d.strip()}" for d in deployments.split(",") if d.strip()
            ]

    def call_ai(self, messages, temperature, max_tokens, model, **kwargs):
        try:
            # Extract deployment name from model (format: "azure/deployment-name")
            deployment_name = model.split("/", 1)[1] if "/" in model else model

            # Prepare messages
            prepared_messages = self._prepare_messages(messages)

            # Call Azure OpenAI exactly like your example
            response = self.sync_client.chat.completions.create(
                model=deployment_name,
                messages=prepared_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise e

    async def call_ai_async(self, messages, temperature, max_tokens, model, **kwargs):
        try:
            deployment_name = model.split("/", 1)[1] if "/" in model else model
            prepared_messages = self._prepare_messages(messages)

            print(deployment_name)
            response = await self.async_client.chat.completions.create(
                model=deployment_name,
                messages=prepared_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise e

    def generate(self, messages, temperature, max_tokens, model=None, **kwargs):
        raise NotImplementedError("This provider does not support image generation")

    async def generate_async(
        self, messages, temperature, max_tokens, model=None, **kwargs
    ):
        raise NotImplementedError("This provider does not support image generation")

    def _prepare_messages(self, messages):
        """Prepare messages for Azure OpenAI API - similar to OpenAI but simpler"""
        prepared_messages = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                prepared_messages.append({"role": message["role"], "content": content})
            elif isinstance(content, list):
                # Azure OpenAI also supports multimodal content
                prepared_content = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        prepared_content.append(item)
                    elif isinstance(item, dict) and item.get("type") == "image_url":
                        # Process image for Azure (if vision models are deployed)
                        image_url = item["image_url"]["url"]
                        if image_url.startswith("data:"):
                            # Already base64 encoded
                            prepared_content.append(item)
                        elif image_url.startswith(("http://", "https://")):
                            # URL - Azure can handle these directly
                            prepared_content.append(item)
                        else:
                            # Local file - need to encode
                            with open(image_url, "rb") as img_file:
                                image_data = base64.b64encode(img_file.read()).decode(
                                    "utf-8"
                                )
                            mime_type, _ = mimetypes.guess_type(image_url)
                            mime_type = mime_type or "image/jpeg"
                            prepared_content.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{mime_type};base64,{image_data}"
                                    },
                                }
                            )
                    elif isinstance(item, dict) and item.get("type") == "input_url":
                        # Handle PDF files for Azure OpenAI
                        pdf_url = item["input_url"]["url"]
                        if pdf_url.startswith("data:"):
                            # Already base64 encoded
                            prepared_content.append(item)
                        elif pdf_url.startswith(("http://", "https://")):
                            # URL - Azure can handle these directly
                            prepared_content.append(item)
                        else:
                            # Local file - need to encode
                            with open(pdf_url, "rb") as pdf_file:
                                pdf_data = base64.b64encode(pdf_file.read()).decode(
                                    "utf-8"
                                )
                            prepared_content.append(
                                {
                                    "type": "input_url",
                                    "input_url": {
                                        "url": f"data:application/pdf;base64,{pdf_data}"
                                    },
                                }
                            )
                    else:
                        prepared_content.append(item)
                prepared_messages.append(
                    {"role": message["role"], "content": prepared_content}
                )
        return prepared_messages


class AnthropicProvider(AIProvider):
    """
    Anthropic Claude provider with support for extended thinking (reasoning) models.

    Extended thinking gives Claude enhanced reasoning capabilities for complex tasks,
    allowing it to show its step-by-step thought process before delivering a final answer.

    Supported reasoning models:
    - Claude Opus 4 (claude-opus-4-20250514): Most capable model with superior reasoning
    - Claude Sonnet 4 (claude-sonnet-4-20250514): High-performance model with reasoning
    - Claude 3.7 Sonnet (claude-3-7-sonnet-20250219): Extended thinking with full output

    Usage with extended thinking:
    - Pass thinking=True to enable with automatic budget calculation
    - Pass thinking={"type": "enabled", "budget_tokens": 10000} for manual control
    - Minimum budget is 1,024 tokens, recommended 16k+ for complex tasks
    - Claude 4 models return summarized thinking, Claude 3.7 returns full thinking

    Note: Extended thinking may increase response time and token usage.
    """

    supported_models = [
        "anthropic/claude-3-7-sonnet-20250219",
        "anthropic/claude-3-5-sonnet-20241022",
        "anthropic/claude-3-5-haiku-20241022",
        "anthropic/claude-3-5-sonnet-20240620",
        "anthropic/claude-3-haiku-20240307",
        "anthropic/claude-3-opus-20240229",
        "anthropic/claude-3-sonnet-20240229",
        "anthropic/claude-2.1",
        "anthropic/claude-2.0",
        "anthropic/claude-opus-4-20250514",
        "anthropic/claude-sonnet-4-20250514",
        "anthropic/claude-opus-4-5-20251101",
    ]

    # Models that support extended thinking (reasoning capabilities)
    extended_thinking_models = [
        "anthropic/claude-opus-4-20250514",
        "anthropic/claude-sonnet-4-20250514",
        "anthropic/claude-3-7-sonnet-20250219",
        "anthropic/claude-opus-4-5-20251101",
    ]

    def __init__(self, api_key=None):
        self.sync_client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        self.async_client = anthropic.AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

        self.supported_models = [
            f"anthropic/{model.id}" for model in self.sync_client.models.list(limit=30)
        ]

    def supports_extended_thinking(self, model):
        """Check if a model supports extended thinking (reasoning capabilities)"""
        return model in self.extended_thinking_models

    def _is_reasoning_model(self, model):
        """Check if the model supports extended thinking - alias for backwards compatibility"""
        return self.supports_extended_thinking(model)

    def _prepare_messages(self, messages):
        """Prepare messages for Claude API, handling both text, images, and caching."""
        system_content = []
        user_messages = []

        for message in messages:
            if message["role"] == "system":
                system_msg = {"type": "text", "text": message["content"]}
                # Add cache_control if present
                if "cache_control" in message:
                    system_msg["cache_control"] = message["cache_control"]
                system_content.append(system_msg)
            else:
                if isinstance(message["content"], str):
                    user_messages.append(
                        {"role": message["role"], "content": message["content"]}
                    )
                elif isinstance(message["content"], list):
                    prepared_content = []
                    for item in message["content"]:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                prepared_content.append(
                                    {"type": "text", "text": item["text"]}
                                )
                            elif item.get("type") == "image_url":
                                # Handle both local files and URLs
                                image_url = item["image_url"]["url"]
                                if image_url.startswith(("http://", "https://")):
                                    prepared_content.append(
                                        {
                                            "type": "image",
                                            "source": {"type": "url", "url": image_url},
                                        }
                                    )
                                else:
                                    # For local files, use base64
                                    image_data = self._process_image(image_url)
                                    prepared_content.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": self._get_media_type(
                                                    image_url
                                                ),
                                                "data": image_data,
                                            },
                                        }
                                    )
                    user_messages.append(
                        {"role": message["role"], "content": prepared_content}
                    )

        return system_content, user_messages

    def _process_image(self, image_path):
        MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB in bytes

        if isinstance(image_path, (str, Path)):
            path = Path(image_path)
            if path.is_file():
                # Read the image file
                with open(path, "rb") as image_file:
                    image_data = image_file.read()

                # Check if image needs resizing
                if len(image_data) > MAX_IMAGE_SIZE:
                    # Open with PIL and resize
                    img = Image.open(io.BytesIO(image_data))

                    # Calculate scaling factor to get under the limit
                    # Start with 0.5 scaling as suggested
                    scale = 0.5
                    img_resized = img.resize(
                        (int(img.width * scale), int(img.height * scale))
                    )

                    # Keep resizing if still too large
                    buffer = io.BytesIO()
                    img_format = img.format or "JPEG"
                    img_resized.save(buffer, format=img_format)
                    resized_data = buffer.getvalue()

                    while len(resized_data) > MAX_IMAGE_SIZE and scale > 0.1:
                        # Reduce scale further if still too large
                        scale *= 0.8
                        img_resized = img.resize(
                            (int(img.width * scale), int(img.height * scale))
                        )
                        buffer = io.BytesIO()
                        img_resized.save(buffer, format=img_format)
                        resized_data = buffer.getvalue()

                    return base64.b64encode(resized_data).decode("utf-8")

                # If image is already small enough, just return the encoded data
                return base64.b64encode(image_data).decode("utf-8")
            # Add URL handling
            elif str(image_path).startswith(("http://", "https://")):
                response = requests.get(str(image_path))
                response.raise_for_status()
                image_data = response.content

                # Check if image needs resizing
                if len(image_data) > MAX_IMAGE_SIZE:
                    # Open with PIL and resize
                    img = Image.open(io.BytesIO(image_data))

                    # Start with 0.5 scaling
                    scale = 0.5
                    img_resized = img.resize(
                        (int(img.width * scale), int(img.height * scale))
                    )

                    # Keep resizing if still too large
                    buffer = io.BytesIO()
                    img_format = img.format or "JPEG"
                    img_resized.save(buffer, format=img_format)
                    resized_data = buffer.getvalue()

                    while len(resized_data) > MAX_IMAGE_SIZE and scale > 0.1:
                        # Reduce scale further if still too large
                        scale *= 0.8
                        img_resized = img.resize(
                            (int(img.width * scale), int(img.height * scale))
                        )
                        buffer = io.BytesIO()
                        img_resized.save(buffer, format=img_format)
                        resized_data = buffer.getvalue()

                    return base64.b64encode(resized_data).decode("utf-8")

                return base64.b64encode(image_data).decode("utf-8")
            else:
                raise ValueError(f"File not found: {image_path}")
        elif isinstance(image_path, bytes):
            image_data = image_path

            # Check if image needs resizing
            if len(image_data) > MAX_IMAGE_SIZE:
                # Open with PIL and resize
                img = Image.open(io.BytesIO(image_data))

                # Start with 0.5 scaling
                scale = 0.5
                img_resized = img.resize(
                    (int(img.width * scale), int(img.height * scale))
                )

                # Keep resizing if still too large
                buffer = io.BytesIO()
                img_format = img.format or "JPEG"
                img_resized.save(buffer, format=img_format)
                resized_data = buffer.getvalue()

                while len(resized_data) > MAX_IMAGE_SIZE and scale > 0.1:
                    # Reduce scale further if still too large
                    scale *= 0.8
                    img_resized = img.resize(
                        (int(img.width * scale), int(img.height * scale))
                    )
                    buffer = io.BytesIO()
                    img_resized.save(buffer, format=img_format)
                    resized_data = buffer.getvalue()

                return base64.b64encode(resized_data).decode("utf-8")

            return base64.b64encode(image_data).decode("utf-8")
        elif isinstance(image_path, Image.Image):
            img = image_path
            # Preserve original format if possible, fallback to PNG
            img_format = getattr(img, "format", "PNG") or "PNG"

            # First try with original size
            buffer = io.BytesIO()
            img.save(buffer, format=img_format)
            image_data = buffer.getvalue()

            # Check if image needs resizing
            if len(image_data) > MAX_IMAGE_SIZE:
                # Start with 0.5 scaling
                scale = 0.5
                img_resized = img.resize(
                    (int(img.width * scale), int(img.height * scale))
                )

                # Keep resizing if still too large
                buffer = io.BytesIO()
                img_resized.save(buffer, format=img_format)
                resized_data = buffer.getvalue()

                while len(resized_data) > MAX_IMAGE_SIZE and scale > 0.1:
                    # Reduce scale further if still too large
                    scale *= 0.8
                    img_resized = img.resize(
                        (int(img.width * scale), int(img.height * scale))
                    )
                    buffer = io.BytesIO()
                    img_resized.save(buffer, format=img_format)
                    resized_data = buffer.getvalue()

                return base64.b64encode(resized_data).decode("utf-8")

            return base64.b64encode(image_data).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image format: {type(image_path)}")

    def _get_media_type(self, file_path):
        if isinstance(file_path, str) and file_path.startswith(("http://", "https://")):
            response = requests.head(file_path)
            return response.headers.get("content-type", "image/jpeg")
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "image/jpeg"

    def generate(self, messages, temperature, max_tokens, model=None, **kwargs):
        raise NotImplementedError("This provider does not support image generation")

    async def generate_async(
        self, messages, temperature, max_tokens, model=None, **kwargs
    ):
        raise NotImplementedError("This provider does not support image generation")

    def call_ai(
        self,
        messages,
        temperature,
        max_tokens,
        model="anthropic/claude-3-5-sonnet-20240620",
        **kwargs,
    ):
        try:
            system_content, user_messages = self._prepare_messages(messages)

            # Extract thinking parameter if provided in kwargs
            thinking = kwargs.pop("thinking", None)

            # If thinking is True (boolean), convert to proper format
            if thinking is True:
                # According to docs: minimum budget is 1,024 tokens
                # Budget MUST be less than max_tokens
                min_budget = 1024

                # Ensure we have room for both thinking and response
                if max_tokens <= min_budget:
                    # If max_tokens is too small, use a smaller budget
                    budget_tokens = max(256, max_tokens - 100)
                else:
                    # Standard calculation with safety margin
                    max_budget = max_tokens - 100  # Leave room for response
                    budget_tokens = max(min_budget, min(max_budget, max_tokens // 2))

                thinking = {"type": "enabled", "budget_tokens": budget_tokens}

            # Handle thinking parameter compatibility constraints BEFORE creating api_params
            if thinking:
                # Adjust top_p if needed
                if "top_p" in kwargs and kwargs["top_p"] < 0.95:
                    # Docs say top_p can be set between 0.95 and 1 when thinking is enabled
                    kwargs["top_p"] = max(kwargs["top_p"], 0.95)

                # Remove unsupported parameters for thinking
                thinking_incompatible = ["top_k"]
                for param in thinking_incompatible:
                    if param in kwargs:
                        kwargs.pop(param)

            # Create API call parameters
            api_params = {
                "model": model.split("/")[-1],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_content,
                "messages": user_messages
                if user_messages
                else [{"role": "user", "content": "Follow the system prompt."}],
                **kwargs,
            }

            # Add thinking parameter only if it's provided
            if thinking:
                # Temperature MUST be set to 1 when thinking is enabled
                api_params["temperature"] = 1
                api_params["thinking"] = thinking

            response = self.sync_client.messages.create(**api_params)

            # Handle thinking content if present
            if hasattr(response, "content") and len(response.content) > 1:
                # Check if any content block is of type "thinking" or "redacted_thinking"
                thinking_blocks = [
                    block
                    for block in response.content
                    if getattr(block, "type", None) in ["thinking", "redacted_thinking"]
                ]
                if thinking_blocks:
                    # You can log or process thinking blocks separately if needed
                    # For now, we'll just return the final text response
                    text_blocks = [
                        block
                        for block in response.content
                        if getattr(block, "type", None) == "text"
                    ]
                    if text_blocks:
                        return text_blocks[0].text

            # Default return for normal responses
            return response.content[0].text
        except Exception as e:
            raise e

    async def call_ai_async(
        self,
        messages,
        temperature,
        max_tokens,
        model="anthropic/claude-3-5-sonnet-20240620",
        **kwargs,
    ):
        try:
            system_content, user_messages = self._prepare_messages(messages)

            # Extract thinking parameter if provided in kwargs
            thinking = kwargs.pop("thinking", None)

            # If thinking is True (boolean), convert to proper format
            if thinking is True:
                # According to docs: minimum budget is 1,024 tokens
                # Budget MUST be less than max_tokens
                min_budget = 1024

                # Ensure we have room for both thinking and response
                if max_tokens <= min_budget:
                    # If max_tokens is too small, use a smaller budget
                    budget_tokens = max(256, max_tokens - 100)
                else:
                    # Standard calculation with safety margin
                    max_budget = max_tokens - 100  # Leave room for response
                    budget_tokens = max(min_budget, min(max_budget, max_tokens // 2))

                thinking = {"type": "enabled", "budget_tokens": budget_tokens}

            # Handle thinking parameter compatibility constraints BEFORE creating api_params
            if thinking:
                # Adjust top_p if needed
                if "top_p" in kwargs and kwargs["top_p"] < 0.95:
                    # Docs say top_p can be set between 0.95 and 1 when thinking is enabled
                    kwargs["top_p"] = max(kwargs["top_p"], 0.95)

                # Remove unsupported parameters for thinking
                thinking_incompatible = ["top_k"]
                for param in thinking_incompatible:
                    if param in kwargs:
                        kwargs.pop(param)

            # Create API call parameters
            api_params = {
                "model": model.split("/")[-1],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_content,
                "messages": user_messages
                if user_messages
                else [{"role": "user", "content": "Follow the system prompt."}],
                **kwargs,
            }

            # Add thinking parameter only if it's provided
            if thinking:
                # Temperature MUST be set to 1 when thinking is enabled
                api_params["temperature"] = 1
                api_params["thinking"] = thinking

            response = await self.async_client.messages.create(**api_params)

            # Handle thinking content if present
            if hasattr(response, "content") and len(response.content) > 1:
                # Check if any content block is of type "thinking" or "redacted_thinking"
                thinking_blocks = [
                    block
                    for block in response.content
                    if getattr(block, "type", None) in ["thinking", "redacted_thinking"]
                ]
                if thinking_blocks:
                    # You can log or process thinking blocks separately if needed
                    # For now, we'll just return the final text response
                    text_blocks = [
                        block
                        for block in response.content
                        if getattr(block, "type", None) == "text"
                    ]
                    if text_blocks:
                        return text_blocks[0].text

            # Default return for normal responses
            return response.content[0].text
        except Exception as e:
            raise e


class VertexAIProvider(AIProvider):
    """
    Vertex AI provider for Anthropic Claude models running on Google Cloud Vertex AI.

    This provider uses the Anthropic Vertex AI client to access Claude models through
    Google Cloud's Vertex AI platform. It supports all the same features as the
    regular Anthropic provider but routes through Vertex AI infrastructure.

    Required environment variables:
    - VERTEX_PROJECT_ID: Your Google Cloud project ID
    - VERTEX_LOCATION: The region where your Vertex AI resources are located
    - GOOGLE_APPLICATION_CREDENTIALS: Path to your service account key file

    Supported models are the same as available in Vertex AI Model Garden.
    """

    supported_models = [
        "vertex/claude-opus-4@20250514",
        "vertex/claude-sonnet-4@20250514",
        "vertex/claude-opus-4-5@20251101",
    ]

    def __init__(self, project_id=None, location=None):
        if not VERTEX_AVAILABLE:
            raise ImportError(
                "anthropic[vertex] is required for VertexAIProvider. Install with: pip install 'anthropic[vertex]'"
            )

        self.project_id = project_id or os.getenv("VERTEX_PROJECT_ID")
        self.location = location or os.getenv("VERTEX_LOCATION", "us-east5")

        if not self.project_id:
            raise ValueError("VERTEX_PROJECT_ID environment variable is required")

        try:
            self.sync_client = AnthropicVertex(
                project_id=self.project_id, region=self.location
            )
            self.async_client = AnthropicVertex(
                project_id=self.project_id, region=self.location
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Vertex AI client: {e}")

    def _prepare_messages(self, messages):
        """Prepare messages for Vertex AI Claude API, handling both text and images."""
        system_content = []
        user_messages = []

        for message in messages:
            if message["role"] == "system":
                system_msg = {"type": "text", "text": message["content"]}
                # Add cache_control if present
                if "cache_control" in message:
                    system_msg["cache_control"] = message["cache_control"]
                system_content.append(system_msg)
            else:
                if isinstance(message["content"], str):
                    user_messages.append(
                        {"role": message["role"], "content": message["content"]}
                    )
                elif isinstance(message["content"], list):
                    prepared_content = []
                    for item in message["content"]:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                prepared_content.append(
                                    {"type": "text", "text": item["text"]}
                                )
                            elif item.get("type") == "image_url":
                                # Handle both local files and URLs
                                image_url = item["image_url"]["url"]
                                if image_url.startswith(("http://", "https://")):
                                    prepared_content.append(
                                        {
                                            "type": "image",
                                            "source": {"type": "url", "url": image_url},
                                        }
                                    )
                                else:
                                    # For local files, use base64
                                    image_data = self._process_image(image_url)
                                    prepared_content.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": self._get_media_type(
                                                    image_url
                                                ),
                                                "data": image_data,
                                            },
                                        }
                                    )
                    user_messages.append(
                        {"role": message["role"], "content": prepared_content}
                    )

        return system_content, user_messages

    def _process_image(self, image_path):
        """Process image with size constraints for Vertex AI."""
        MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5MB in bytes

        if isinstance(image_path, (str, Path)):
            path = Path(image_path)
            if path.is_file():
                # Read the image file
                with open(path, "rb") as image_file:
                    image_data = image_file.read()

                # Check if image needs resizing
                if len(image_data) > MAX_IMAGE_SIZE:
                    # Open with PIL and resize
                    img = Image.open(io.BytesIO(image_data))

                    # Calculate scaling factor to get under the limit
                    scale = 0.5
                    img_resized = img.resize(
                        (int(img.width * scale), int(img.height * scale))
                    )

                    # Keep resizing if still too large
                    buffer = io.BytesIO()
                    img_format = img.format or "JPEG"
                    img_resized.save(buffer, format=img_format)
                    resized_data = buffer.getvalue()

                    while len(resized_data) > MAX_IMAGE_SIZE and scale > 0.1:
                        scale *= 0.8
                        img_resized = img.resize(
                            (int(img.width * scale), int(img.height * scale))
                        )
                        buffer = io.BytesIO()
                        img_resized.save(buffer, format=img_format)
                        resized_data = buffer.getvalue()

                    return base64.b64encode(resized_data).decode("utf-8")

                return base64.b64encode(image_data).decode("utf-8")
            elif str(image_path).startswith(("http://", "https://")):
                response = requests.get(str(image_path))
                response.raise_for_status()
                image_data = response.content

                # Check if image needs resizing
                if len(image_data) > MAX_IMAGE_SIZE:
                    img = Image.open(io.BytesIO(image_data))
                    scale = 0.5
                    img_resized = img.resize(
                        (int(img.width * scale), int(img.height * scale))
                    )

                    buffer = io.BytesIO()
                    img_format = img.format or "JPEG"
                    img_resized.save(buffer, format=img_format)
                    resized_data = buffer.getvalue()

                    while len(resized_data) > MAX_IMAGE_SIZE and scale > 0.1:
                        scale *= 0.8
                        img_resized = img.resize(
                            (int(img.width * scale), int(img.height * scale))
                        )
                        buffer = io.BytesIO()
                        img_resized.save(buffer, format=img_format)
                        resized_data = buffer.getvalue()

                    return base64.b64encode(resized_data).decode("utf-8")

                return base64.b64encode(image_data).decode("utf-8")
            else:
                raise ValueError(f"File not found: {image_path}")
        elif isinstance(image_path, bytes):
            image_data = image_path

            # Check if image needs resizing
            if len(image_data) > MAX_IMAGE_SIZE:
                img = Image.open(io.BytesIO(image_data))
                scale = 0.5
                img_resized = img.resize(
                    (int(img.width * scale), int(img.height * scale))
                )

                buffer = io.BytesIO()
                img_format = img.format or "JPEG"
                img_resized.save(buffer, format=img_format)
                resized_data = buffer.getvalue()

                while len(resized_data) > MAX_IMAGE_SIZE and scale > 0.1:
                    scale *= 0.8
                    img_resized = img.resize(
                        (int(img.width * scale), int(img.height * scale))
                    )
                    buffer = io.BytesIO()
                    img_resized.save(buffer, format=img_format)
                    resized_data = buffer.getvalue()

                return base64.b64encode(resized_data).decode("utf-8")

            return base64.b64encode(image_data).decode("utf-8")
        elif isinstance(image_path, Image.Image):
            img = image_path
            img_format = getattr(img, "format", "PNG") or "PNG"

            # First try with original size
            buffer = io.BytesIO()
            img.save(buffer, format=img_format)
            image_data = buffer.getvalue()

            # Check if image needs resizing
            if len(image_data) > MAX_IMAGE_SIZE:
                scale = 0.5
                img_resized = img.resize(
                    (int(img.width * scale), int(img.height * scale))
                )

                buffer = io.BytesIO()
                img_resized.save(buffer, format=img_format)
                resized_data = buffer.getvalue()

                while len(resized_data) > MAX_IMAGE_SIZE and scale > 0.1:
                    scale *= 0.8
                    img_resized = img.resize(
                        (int(img.width * scale), int(img.height * scale))
                    )
                    buffer = io.BytesIO()
                    img_resized.save(buffer, format=img_format)
                    resized_data = buffer.getvalue()

                return base64.b64encode(resized_data).decode("utf-8")

            return base64.b64encode(image_data).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image format: {type(image_path)}")

    def _get_media_type(self, file_path):
        """Get media type for image files."""
        if isinstance(file_path, str) and file_path.startswith(("http://", "https://")):
            response = requests.head(file_path)
            return response.headers.get("content-type", "image/jpeg")
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "image/jpeg"

    def call_ai(
        self,
        messages,
        temperature,
        max_tokens,
        model="vertex/claude-sonnet-4@20250514",
        **kwargs,
    ):
        try:
            system_content, user_messages = self._prepare_messages(messages)

            # Create API call parameters
            api_params = {
                "model": model.split("/")[-1],  # Remove vertex/ prefix
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_content,
                "messages": user_messages
                if user_messages
                else [{"role": "user", "content": "Follow the system prompt."}],
                **kwargs,
            }

            response = self.sync_client.messages.create(**api_params)
            return response.content[0].text
        except Exception as e:
            raise e

    async def call_ai_async(
        self,
        messages,
        temperature,
        max_tokens,
        model="vertex/claude-sonnet-4@20250514",
        **kwargs,
    ):
        try:
            system_content, user_messages = self._prepare_messages(messages)

            # Create API call parameters
            api_params = {
                "model": model.split("/")[-1],  # Remove vertex/ prefix
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_content,
                "messages": user_messages
                if user_messages
                else [{"role": "user", "content": "Follow the system prompt."}],
                **kwargs,
            }

            response = await self.async_client.messages.create(**api_params)
            return response.content[0].text
        except Exception as e:
            raise e

    def generate(self, messages, temperature, max_tokens, model=None, **kwargs):
        raise NotImplementedError("This provider does not support image generation")

    async def generate_async(
        self, messages, temperature, max_tokens, model=None, **kwargs
    ):
        raise NotImplementedError("This provider does not support image generation")


class GeminiProvider(AIProvider):
    supported_models = [
        "gemini/gemini-1.0-pro-vision-latest",
        "gemini/gemini-1.5-flash",
        "gemini/gemini-1.5-flash-001",
        "gemini/gemini-1.5-flash-001-tuning",
        "gemini/gemini-1.5-flash-002",
        "gemini/gemini-1.5-flash-8b",
        "gemini/gemini-1.5-flash-8b-001",
        "gemini/gemini-1.5-flash-8b-exp-0827",
        "gemini/gemini-1.5-flash-8b-exp-0924",
        "gemini/gemini-1.5-flash-8b-latest",
        "gemini/gemini-1.5-flash-latest",
        "gemini/gemini-1.5-pro",
        "gemini/gemini-1.5-pro-001",
        "gemini/gemini-1.5-pro-002",
        "gemini/gemini-1.5-pro-latest",
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.0-flash-001",
        "gemini/gemini-2.0-flash-exp",
        "gemini/gemini-2.0-flash-exp-image-generation",
        "gemini/gemini-2.0-flash-lite",
        "gemini/gemini-2.0-flash-lite-001",
        "gemini/gemini-2.0-flash-lite-preview",
        "gemini/gemini-2.0-flash-lite-preview-02-05",
        "gemini/gemini-2.0-flash-thinking-exp",
        "gemini/gemini-2.0-flash-thinking-exp-01-21",
        "gemini/gemini-2.0-flash-thinking-exp-1219",
        "gemini/gemini-2.0-pro-exp",
        "gemini/gemini-2.0-pro-exp-02-05",
        "gemini/gemini-exp-1206",
        "gemini/gemini-pro-vision",
        "gemini/gemma-3-27b-it",
        "gemini/learnlm-1.5-pro-experimental",
        "gemini/models/gemini-1.0-pro-vision-latest",
        "gemini/models/gemini-1.5-flash",
        "gemini/models/gemini-1.5-flash-001",
        "gemini/models/gemini-1.5-flash-001-tuning",
        "gemini/models/gemini-1.5-flash-002",
        "gemini/models/gemini-1.5-flash-8b",
        "gemini/models/gemini-1.5-flash-8b-001",
        "gemini/models/gemini-1.5-flash-8b-exp-0827",
        "gemini/models/gemini-1.5-flash-8b-exp-0924",
        "gemini/models/gemini-1.5-flash-8b-latest",
        "gemini/models/gemini-1.5-flash-latest",
        "gemini/models/gemini-1.5-pro",
        "gemini/models/gemini-1.5-pro-001",
        "gemini/models/gemini-1.5-pro-002",
        "gemini/models/gemini-1.5-pro-latest",
        "gemini/models/gemini-2.0-flash",
        "gemini/models/gemini-2.0-flash-001",
        "gemini/models/gemini-2.0-flash-exp",
        "gemini/models/gemini-2.0-flash-exp-image-generation",
        "gemini/models/gemini-2.0-flash-lite",
        "gemini/models/gemini-2.0-flash-lite-001",
        "gemini/models/gemini-2.0-flash-lite-preview",
        "gemini/models/gemini-2.0-flash-lite-preview-02-05",
        "gemini/models/gemini-2.0-flash-preview-image-generation",
        "gemini/models/gemini-2.0-flash-thinking-exp",
        "gemini/models/gemini-2.0-flash-thinking-exp-01-21",
        "gemini/models/gemini-2.0-flash-thinking-exp-1219",
        "gemini/models/gemini-2.0-pro-exp",
        "gemini/models/gemini-2.0-pro-exp-02-05",
        "gemini/models/gemini-2.5-flash-preview-04-17",
        "gemini/models/gemini-2.5-flash-preview-04-17-thinking",
        "gemini/models/gemini-2.5-flash-preview-05-20",
        "gemini/models/gemini-2.5-flash-preview-tts",
        "gemini/models/gemini-2.5-pro-exp-03-25",
        "gemini/models/gemini-2.5-pro-preview-03-25",
        "gemini/models/gemini-2.5-pro-preview-05-06",
        "gemini/models/gemini-2.5-pro-preview-tts",
        "gemini/models/gemini-exp-1206",
        "gemini/models/gemini-pro-vision",
        "gemini/models/gemma-3-12b-it",
        "gemini/models/gemma-3-1b-it",
        "gemini/models/gemma-3-27b-it",
        "gemini/models/gemma-3-4b-it",
        "gemini/models/gemma-3n-e4b-it",
        "genai/models/gemini-1.0-pro-vision-latest",
        "genai/models/gemini-1.5-flash",
        "genai/models/gemini-1.5-flash-001",
        "genai/models/gemini-1.5-flash-001-tuning",
        "genai/models/gemini-1.5-flash-002",
        "genai/models/gemini-1.5-flash-8b",
        "genai/models/gemini-1.5-flash-8b-001",
        "genai/models/gemini-1.5-flash-8b-exp-0827",
        "genai/models/gemini-1.5-flash-8b-exp-0924",
        "genai/models/gemini-1.5-flash-8b-latest",
        "genai/models/gemini-1.5-flash-latest",
        "genai/models/gemini-1.5-pro",
        "genai/models/gemini-1.5-pro-001",
        "genai/models/gemini-1.5-pro-002",
        "genai/models/gemini-1.5-pro-latest",
        "genai/models/gemini-2.0-flash",
        "genai/models/gemini-2.0-flash-001",
        "genai/models/gemini-2.0-flash-exp",
        "genai/models/gemini-2.0-flash-exp-image-generation",
        "genai/models/gemini-2.0-flash-lite",
        "genai/models/gemini-2.0-flash-lite-001",
        "genai/models/gemini-2.0-flash-lite-preview",
        "genai/models/gemini-2.0-flash-lite-preview-02-05",
        "genai/models/gemini-2.0-flash-preview-image-generation",
        "genai/models/gemini-2.0-flash-thinking-exp",
        "genai/models/gemini-2.0-flash-thinking-exp-01-21",
        "genai/models/gemini-2.0-flash-thinking-exp-1219",
        "genai/models/gemini-2.0-pro-exp",
        "genai/models/gemini-2.0-pro-exp-02-05",
        "genai/models/gemini-2.5-flash-preview-04-17",
        "genai/models/gemini-2.5-flash-preview-04-17-thinking",
        "genai/models/gemini-2.5-flash-preview-05-20",
        "genai/models/gemini-2.5-flash-preview-tts",
        "genai/models/gemini-2.5-pro-exp-03-25",
        "genai/models/gemini-2.5-pro-preview-03-25",
        "genai/models/gemini-2.5-pro-preview-05-06",
        "genai/models/gemini-2.5-pro-preview-tts",
        "genai/models/gemini-exp-1206",
        "genai/models/gemini-pro-vision",
        "genai/models/gemma-3-12b-it",
        "genai/models/gemma-3-1b-it",
        "genai/models/gemma-3-27b-it",
        "genai/models/gemma-3-4b-it",
        "genai/models/gemma-3n-e4b-it",
        "genai/models/learnlm-2.0-flash-experimental",
        "gemini/gemini-2.5-pro",
        "gemini/gemini-2.5-flash",
        "genai/gemini-2.5-pro",
        "genai/gemini-2.5-flash",
        "gemini/gemini-3-pro-preview",
        "genai/gemini-3-pro-preview",
        "gemini/gemini-3.0-pro",
        "gemini/gemini-3.0-pro-001",
        "gemini/gemini-3.0-deep-think",
        "genai/gemini-3.0-pro",
        "genai/gemini-3.0-pro-001",
        "genai/gemini-3.0-deep-think",
    ]

    def __init__(self, api_key=None):
        self.client = genai.Client(api_key=api_key or os.getenv("GOOGLE_API_KEY"))
        try:
            # Get models from API
            api_models = []
            for m in self.client.models.list():
                if (
                    hasattr(m, "supported_actions")
                    and "generateContent" in m.supported_actions
                ):
                    # Handle both old and new naming conventions
                    model_name = m.name

                    # Remove "models/" prefix if present
                    if model_name.startswith("models/"):
                        model_name = model_name[7:]  # Remove "models/" prefix

                    # Create both gemini/ and genai/ prefixed versions
                    genai_name = f"genai/{model_name}"
                    gemini_name = f"gemini/{model_name}"

                    # Only add if not already in the list
                    if genai_name not in self.supported_models:
                        api_models.append(genai_name)
                    if gemini_name not in self.supported_models:
                        api_models.append(gemini_name)

            # Add the API models to our supported models
            if api_models:
                self.supported_models.extend(api_models)
        except Exception as e:
            print(f"Error initializing GeminiProvider: {e}")

    def call_ai(
        self,
        messages,
        temperature,
        max_tokens,
        model="gemini/gemini-2.0-flash-exp",
        **kwargs,
    ):
        try:
            # Handle different model name formats
            if "/" in model:
                model_parts = model.split("/")
                if model_parts[0] in ["gemini", "genai", "models"]:
                    # Extract just the model name without prefixes
                    model_name = model_parts[-1]
                else:
                    model_name = model
            else:
                model_name = model

            # Extract system message if present
            system_instruction = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), None
            )

            # Get the user messages
            user_messages = [msg for msg in messages if msg["role"] != "system"]

            # Convert messages to content
            if not user_messages:
                contents = "Follow the system instructions."
            elif len(user_messages) == 1 and isinstance(
                user_messages[0]["content"], str
            ):
                contents = user_messages[0]["content"]
            else:
                # Handle multiple messages or messages with images and videos
                contents = []
                for message in user_messages:
                    if isinstance(message["content"], str):
                        contents.append(message["content"])
                    elif isinstance(message["content"], list):
                        text_parts = []
                        media_parts = []
                        for item in message["content"]:
                            if item.get("type") == "text":
                                text_parts.append(item["text"])
                            elif item.get("type") == "image_url":
                                image_path = item["image_url"]["url"]
                                with open(image_path, "rb") as f:
                                    image_data = f.read()
                                media_parts.append(
                                    types.Part.from_bytes(
                                        data=image_data, mime_type="image/jpeg"
                                    )
                                )
                            elif item.get("type") == "video_url":
                                # Support video processing
                                video_path = item["video_url"]["url"]
                                video_file = self.process_video(video_path)
                                media_parts.append(video_file)

                        # Always ensure there's text content
                        if not text_parts:
                            text_parts.append("Consider this media in your response.")

                        # Combine text parts into a single string
                        contents.append(" ".join(text_parts))
                        # Add media parts after text
                        contents.extend(media_parts)

            # Extract thinking configuration from kwargs
            thinking_config = kwargs.pop("thinking_config", None)
            thinking_budget = kwargs.pop("thinking_budget", None)

            # Handle thinking configuration
            config_kwargs = {}
            if thinking_config:
                config_kwargs["thinking_config"] = thinking_config
            elif thinking_budget:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )

            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=system_instruction,
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_CIVIC_INTEGRITY",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_NONE",
                        ),
                    ],
                    **config_kwargs,
                ),
            )
            if response.text:
                return response.text

            # If no text, check why
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.finish_reason:
                    # Raise a helpful error about why it stopped
                    raise ValueError(
                        f"Gemini refused response. Finish Reason: {candidate.finish_reason}"
                    )

            return ""  # Or raise an error if empty is unacceptable
        except Exception as e:
            raise e

    async def call_ai_async(
        self,
        messages,
        temperature,
        max_tokens,
        model="gemini/gemini-2.0-flash-exp",
        **kwargs,
    ):
        try:
            if "/" in model:
                model_parts = model.split("/")
                if model_parts[0] in ["gemini", "genai", "models"]:
                    # Extract just the model name without prefixes
                    model_name = model_parts[-1]
                else:
                    model_name = model
            else:
                model_name = model

            # Extract system message if present
            system_instruction = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), None
            )

            # Get the user messages
            user_messages = [msg for msg in messages if msg["role"] != "system"]

            if not user_messages:
                contents = "Follow the system instructions."
            elif len(user_messages) == 1 and isinstance(
                user_messages[0]["content"], str
            ):
                contents = user_messages[0]["content"]
            else:
                # Handle multiple messages or messages with images and videos
                contents = []
                for message in user_messages:
                    if isinstance(message["content"], str):
                        contents.append(message["content"])
                    elif isinstance(message["content"], list):
                        text_parts = []
                        media_parts = []
                        for item in message["content"]:
                            if item.get("type") == "text":
                                text_parts.append(item["text"])
                            elif item.get("type") == "image_url":
                                image_path = item["image_url"]["url"]
                                with open(image_path, "rb") as f:
                                    image_data = f.read()
                                media_parts.append(
                                    types.Part.from_bytes(
                                        data=image_data, mime_type="image/jpeg"
                                    )
                                )
                            elif item.get("type") == "video_url":
                                # Support video processing
                                video_path = item["video_url"]["url"]
                                video_file = self.process_video(video_path)
                                media_parts.append(video_file)

                        # Always ensure there's text content
                        if not text_parts:
                            text_parts.append("Consider this media in your response.")

                        # Combine text parts into a single string
                        contents.append(" ".join(text_parts))
                        # Add media parts after text
                        contents.extend(media_parts)

            # Extract thinking configuration from kwargs
            thinking_config = kwargs.pop("thinking_config", None)
            thinking_budget = kwargs.pop("thinking_budget", None)

            # Handle thinking configuration
            config_kwargs = {}
            if thinking_config:
                config_kwargs["thinking_config"] = thinking_config
            elif thinking_budget:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )

            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    system_instruction=system_instruction,
                    safety_settings=[
                        types.SafetySetting(
                            category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_CIVIC_INTEGRITY",
                            threshold="BLOCK_NONE",
                        ),
                        types.SafetySetting(
                            category="HARM_CATEGORY_DANGEROUS_CONTENT",
                            threshold="BLOCK_NONE",
                        ),
                    ],
                    **config_kwargs,
                ),
            )

            if response.text:
                return response.text

            # If no text, check why
            if response.candidates:
                candidate = response.candidates[0]
                if candidate.finish_reason:
                    # Raise a helpful error about why it stopped
                    raise ValueError(
                        f"Gemini refused response. Finish Reason: {candidate.finish_reason}"
                    )

            return ""  # Or raise an error if empty is unacceptable
        except Exception as e:
            raise e

    def process_video(self, video_path):
        """
        Public method to process and upload video files to Gemini.

        Args:
            video_path: Path to the video file

        Returns:
            Uploaded video file object that can be used in generate_content
        """
        video_file = self.client.files.upload(file=video_path)

        # Wait until the uploaded video is available
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = self.client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state.name}")

        return video_file

    def _process_video(self, video_path):
        # Keep the old private method for backward compatibility
        return self.process_video(video_path)

    def generate(
        self,
        messages,
        temperature,
        max_tokens,
        model="gemini/gemini-2.0-flash-exp-image-generation",
        **kwargs,
    ):
        try:
            # Extract system message if present
            system_instruction = next(
                (msg["content"] for msg in messages if msg["role"] == "system"), None
            )

            # Get the user messages
            user_messages = [msg for msg in messages if msg["role"] != "system"]

            # Convert messages to content
            if not user_messages:
                contents = "Follow the system instructions."
            elif len(user_messages) == 1 and isinstance(
                user_messages[0]["content"], str
            ):
                contents = [user_messages[0]["content"]]
            else:
                # Handle multiple messages or messages with images and videos
                contents = []
                for message in user_messages:
                    if isinstance(message["content"], str):
                        contents.append(message["content"])
                    elif isinstance(message["content"], list):
                        text_parts = []
                        media_parts = []
                        for item in message["content"]:
                            if item.get("type") == "text":
                                text_parts.append(item["text"])
                            elif item.get("type") == "image_url":
                                image_path = item["image_url"]["url"]
                                with open(image_path, "rb") as f:
                                    image_data = f.read()
                                media_parts.append(
                                    types.Part.from_bytes(
                                        data=image_data, mime_type="image/jpeg"
                                    )
                                )
                            elif item.get("type") == "video_url":
                                # Support video processing
                                video_path = item["video_url"]["url"]
                                video_file = self.process_video(video_path)
                                media_parts.append(video_file)

                        # Always ensure there's text content
                        if not text_parts:
                            text_parts.append("Consider this media in your response.")

                        # Combine text parts into a single string
                        contents.append(" ".join(text_parts))
                        # Add media parts after text
                        contents.extend(media_parts)

            # Configure response modalities to include both text and image
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction,
                response_modalities=["Text", "Image"],
                safety_settings=[
                    types.SafetySetting(
                        category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_CIVIC_INTEGRITY",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        threshold="BLOCK_NONE",
                    ),
                    types.SafetySetting(
                        category="HARM_CATEGORY_DANGEROUS_CONTENT",
                        threshold="BLOCK_NONE",
                    ),
                ],
            )

            response = self.client.models.generate_content(
                model=model.split("/")[-1],
                contents=contents,
                config=config,
            )

            # Process response to extract both text and images
            result = {"text": "", "images": []}

            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.text is not None:
                        result["text"] += part.text
                    elif part.inline_data is not None:
                        # Convert byte data to PIL Image
                        image = Image.open(io.BytesIO(part.inline_data.data))
                        result["images"].append(image)

            return result
        except Exception as e:
            raise e

    async def generate_async(
        self,
        messages,
        temperature,
        max_tokens,
        model="gemini/gemini-2.0-flash-exp-image-generation",
        **kwargs,
    ):
        # For simplicity, we'll use the synchronous version for now
        # In a real-world scenario, you might want to use an async implementation
        return self.generate(messages, temperature, max_tokens, model, **kwargs)


class DeepSeekProvider(AIProvider):
    supported_models = ["deepseek-chat", "deepseek-reasoner"]

    def __init__(self, api_key=None):
        self.sync_client = OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.async_client = AsyncOpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

    def generate(self, messages, temperature, max_tokens, model=None, **kwargs):
        raise NotImplementedError("This provider does not support image generation")

    async def generate_async(
        self, messages, temperature, max_tokens, model=None, **kwargs
    ):
        raise NotImplementedError("This provider does not support image generation")

    def call_ai(
        self, messages, temperature, max_tokens, model="deepseek-chat", **kwargs
    ):
        try:
            response = self.sync_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise e

    async def call_ai_async(
        self, messages, temperature, max_tokens, model="deepseek-chat", **kwargs
    ):
        try:
            response = await self.async_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise e


class BedrockProvider(AIProvider):
    """AWS Bedrock provider supporting multiple model families"""

    supported_models = [
        # Anthropic Claude Models - Direct IDs
        "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
        "bedrock/anthropic.claude-3-opus-20240229-v1:0",
        "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        "bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
        "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        "bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0",
        "bedrock/anthropic.claude-opus-4-20250514-v1:0",
        "bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
        "bedrock/anthropic.claude-opus-4-5-20251101-v1:0",
        "bedrock/anthropic.claude-v2:1",
        "bedrock/anthropic.claude-v2",
        "bedrock/anthropic.claude-instant-v1",
        # Anthropic Claude Models - APAC Inference Profiles
        "bedrock/apac.anthropic.claude-3-haiku-20240307-v1:0",
        "bedrock/apac.anthropic.claude-3-sonnet-20240229-v1:0",
        "bedrock/apac.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "bedrock/apac.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "bedrock/apac.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "bedrock/apac.anthropic.claude-sonnet-4-20250514-v1:0",
        "bedrock/apac.anthropic.claude-opus-4-5-20251101-v1:0",
        # Anthropic Claude Models - US Inference Profiles
        "bedrock/us.anthropic.claude-3-haiku-20240307-v1:0",
        "bedrock/us.anthropic.claude-3-opus-20240229-v1:0",
        "bedrock/us.anthropic.claude-3-sonnet-20240229-v1:0",
        "bedrock/us.anthropic.claude-3-5-haiku-20241022-v1:0",
        "bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "bedrock/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "bedrock/us.anthropic.claude-opus-4-20250514-v1:0",
        "bedrock/us.anthropic.claude-sonnet-4-20250514-v1:0",
        "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0",
        # Anthropic Claude Models - EU Inference Profiles
        "bedrock/eu.anthropic.claude-3-haiku-20240307-v1:0",
        "bedrock/eu.anthropic.claude-3-opus-20240229-v1:0",
        "bedrock/eu.anthropic.claude-3-sonnet-20240229-v1:0",
        "bedrock/eu.anthropic.claude-3-5-haiku-20241022-v1:0",
        "bedrock/eu.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "bedrock/eu.anthropic.claude-3-5-sonnet-20240620-v1:0",
        "bedrock/eu.anthropic.claude-3-7-sonnet-20250219-v1:0",
        "bedrock/eu.anthropic.claude-opus-4-20250514-v1:0",
        "bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0",
        "bedrock/eu.anthropic.claude-opus-4-5-20251101-v1:0",
        # Amazon Nova Models
        "bedrock/amazon.nova-lite-v1:0",
        "bedrock/amazon.nova-micro-v1:0",
        "bedrock/amazon.nova-pro-v1:0",
        "bedrock/amazon.nova-premier-v1:0",
        # Amazon Titan Models
        "bedrock/amazon.titan-text-express-v1",
        "bedrock/amazon.titan-text-lite-v1",
        "bedrock/amazon.titan-text-premier-v1:0",
        # Meta Llama Models
        "bedrock/meta.llama3-8b-instruct-v1:0",
        "bedrock/meta.llama3-70b-instruct-v1:0",
        "bedrock/meta.llama3-1-8b-instruct-v1:0",
        "bedrock/meta.llama3-1-70b-instruct-v1:0",
        "bedrock/meta.llama3-1-405b-instruct-v1:0",
        "bedrock/meta.llama3-2-1b-instruct-v1:0",
        "bedrock/meta.llama3-2-3b-instruct-v1:0",
        "bedrock/meta.llama3-2-11b-instruct-v1:0",
        "bedrock/meta.llama3-2-90b-instruct-v1:0",
        "bedrock/meta.llama3-3-70b-instruct-v1:0",
        "bedrock/meta.llama4-scout-17b-instruct-v1:0",
        "bedrock/meta.llama4-maverick-17b-instruct-v1:0",
        # Cohere Models
        "bedrock/cohere.command-r-v1:0",
        "bedrock/cohere.command-r-plus-v1:0",
        "bedrock/cohere.command-text-v14",
        "bedrock/cohere.command-light-text-v14",
        # Mistral Models
        "bedrock/mistral.mistral-7b-instruct-v0:2",
        "bedrock/mistral.mistral-large-2402-v1:0",
        "bedrock/mistral.mistral-large-2407-v1:0",
        "bedrock/mistral.mistral-small-2402-v1:0",
        "bedrock/mistral.mixtral-8x7b-instruct-v0:1",
        "bedrock/mistral.pixtral-large-2502-v1:0",
        # AI21 Labs Models
        "bedrock/ai21.jamba-1-5-large-v1:0",
        "bedrock/ai21.jamba-1-5-mini-v1:0",
        "bedrock/ai21.jamba-instruct-v1:0",
        # DeepSeek Models
        "bedrock/deepseek.r1-v1:0",
        # Writer Models
        "bedrock/writer.palmyra-x4-v1:0",
        "bedrock/writer.palmyra-x5-v1:0",
        # Support for ARN-based inference profiles (pattern matching)
        # This allows for any ARN format: arn:aws:bedrock:region:account:inference-profile/model
    ]

    def __init__(
        self,
        region_name="us-east-1",
        aws_access_key_id=None,
        aws_secret_access_key=None,
    ):
        if not BEDROCK_AVAILABLE:
            raise ImportError(
                "boto3 is required for BedrockProvider. Install with: pip install boto3"
            )

        # Set up AWS credentials and region
        kwargs = {"region_name": region_name}
        if aws_access_key_id:
            kwargs["aws_access_key_id"] = aws_access_key_id
        if aws_secret_access_key:
            kwargs["aws_secret_access_key"] = aws_secret_access_key

        # Use environment variables or default credential chain if not provided
        if not aws_access_key_id and not aws_secret_access_key:
            if os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"):
                kwargs["aws_access_key_id"] = os.getenv("AWS_ACCESS_KEY_ID")
                kwargs["aws_secret_access_key"] = os.getenv("AWS_SECRET_ACCESS_KEY")

        self.bedrock_runtime = boto3.client("bedrock-runtime", **kwargs)
        self.region_name = region_name

    def _extract_model_from_arn(self, arn):
        """Extract the model identifier from an ARN"""
        if not arn.startswith("arn:aws:bedrock:"):
            return arn

        # ARN format: arn:aws:bedrock:region:account:inference-profile/model-id
        parts = arn.split("/")
        if len(parts) >= 2:
            return parts[-1]  # Get the model-id part
        return arn

    def _get_model_family(self, model_id):
        """Determine the model family from the model ID"""
        # Handle ARN format
        if model_id.startswith("arn:aws:bedrock:"):
            base_model = self._extract_model_from_arn(model_id)
        # Handle inference profiles (strip region prefix)
        elif model_id.startswith(("apac.", "us.", "eu.")):
            base_model = model_id.split(".", 1)[1]  # Remove region prefix
        else:
            base_model = model_id

        if "anthropic" in base_model:
            return "anthropic"
        elif "amazon.nova" in base_model:
            return "nova"
        elif "amazon.titan" in base_model:
            return "titan"
        elif "meta.llama" in base_model:
            return "llama"
        elif "cohere" in base_model:
            return "cohere"
        elif "mistral" in base_model:
            return "mistral"
        elif "ai21" in base_model:
            return "ai21"
        elif "deepseek" in base_model:
            return "deepseek"
        elif "writer" in base_model:
            return "writer"
        else:
            return "unknown"

    def _is_inference_profile(self, model_id):
        """Check if the model ID is an inference profile"""
        return (
            model_id.startswith(("apac.", "us.", "eu."))
            or "arn:aws:bedrock" in model_id
        )

    def _prepare_anthropic_request(self, messages, max_tokens, temperature, **kwargs):
        """Prepare request for Anthropic Claude models using Messages API"""
        system_content = ""
        user_messages = []

        for message in messages:
            if message["role"] == "system":
                # For Bedrock, system should be a simple string, not an array
                if system_content:
                    system_content += "\n" + message["content"]
                else:
                    system_content = message["content"]
            else:
                if isinstance(message["content"], str):
                    user_messages.append(
                        {
                            "role": message["role"],
                            "content": [{"type": "text", "text": message["content"]}],
                        }
                    )
                elif isinstance(message["content"], list):
                    prepared_content = []
                    for item in message["content"]:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                prepared_content.append(
                                    {"type": "text", "text": item["text"]}
                                )
                            elif item.get("type") == "image_url":
                                # Handle image for Bedrock
                                image_url = item["image_url"]["url"]
                                if image_url.startswith(("http://", "https://")):
                                    # Download image for processing
                                    response = requests.get(image_url)
                                    response.raise_for_status()
                                    image_data = base64.b64encode(
                                        response.content
                                    ).decode("utf-8")
                                else:
                                    # Local file
                                    with open(image_url, "rb") as img_file:
                                        image_data = base64.b64encode(
                                            img_file.read()
                                        ).decode("utf-8")

                                prepared_content.append(
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/jpeg",
                                            "data": image_data,
                                        },
                                    }
                                )
                    user_messages.append(
                        {"role": message["role"], "content": prepared_content}
                    )

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": user_messages
            or [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        }

        if system_content:
            request_body["system"] = system_content

        # Add any additional kwargs
        request_body.update(
            {k: v for k, v in kwargs.items() if k not in ["anthropic_version"]}
        )

        return request_body

    def _prepare_titan_request(self, messages, max_tokens, temperature, **kwargs):
        """Prepare request for Amazon Titan models"""
        # Combine all messages into a single prompt for Titan
        prompt = ""
        for message in messages:
            role = message["role"].capitalize()
            content = message["content"]
            if isinstance(content, list):
                # Extract text from content list
                text_parts = [
                    item["text"] for item in content if item.get("type") == "text"
                ]
                content = " ".join(text_parts)
            prompt += f"{role}: {content}\n"

        request_body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": max_tokens,
                "temperature": temperature,
                "topP": kwargs.get("top_p", 0.9),
                "stopSequences": kwargs.get("stop_sequences", []),
            },
        }
        return request_body

    def _prepare_llama_request(self, messages, max_tokens, temperature, **kwargs):
        """Prepare request for Meta Llama models"""
        # Format prompt for Llama
        prompt = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            if isinstance(content, list):
                text_parts = [
                    item["text"] for item in content if item.get("type") == "text"
                ]
                content = " ".join(text_parts)

            if role == "system":
                prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{content}<|eot_id|>"
            elif role == "user":
                prompt += (
                    f"<|start_header_id|>user<|end_header_id|>\n{content}<|eot_id|>"
                )
            elif role == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n{content}<|eot_id|>"

        prompt += "<|start_header_id|>assistant<|end_header_id|>\n"

        request_body = {
            "prompt": prompt,
            "max_gen_len": max_tokens,
            "temperature": temperature,
            "top_p": kwargs.get("top_p", 0.9),
        }
        return request_body

    def _prepare_cohere_request(self, messages, max_tokens, temperature, **kwargs):
        """Prepare request for Cohere models"""
        # Extract the last user message as the prompt
        prompt = ""
        chat_history = []

        for i, message in enumerate(messages):
            content = message["content"]
            if isinstance(content, list):
                text_parts = [
                    item["text"] for item in content if item.get("type") == "text"
                ]
                content = " ".join(text_parts)

            if message["role"] == "user" and i == len(messages) - 1:
                prompt = content
            elif message["role"] in ["user", "assistant"]:
                chat_history.append(
                    {
                        "role": "USER" if message["role"] == "user" else "CHATBOT",
                        "message": content,
                    }
                )

        request_body = {
            "message": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_history": chat_history,
            "p": kwargs.get("top_p", 0.9),
            "k": kwargs.get("top_k", 0),
        }
        return request_body

    def _prepare_generic_request(self, messages, max_tokens, temperature, **kwargs):
        """Generic request preparation for other model families"""
        # Simple prompt-based approach
        prompt = ""
        for message in messages:
            role = message["role"].capitalize()
            content = message["content"]
            if isinstance(content, list):
                text_parts = [
                    item["text"] for item in content if item.get("type") == "text"
                ]
                content = " ".join(text_parts)
            prompt += f"{role}: {content}\n"

        prompt += "Assistant:"

        request_body = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        request_body.update(kwargs)
        return request_body

    def call_ai(
        self,
        messages,
        temperature,
        max_tokens,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        **kwargs,
    ):
        try:
            # Extract model ID from prefixed format
            model_id = model.replace("bedrock/", "")
            model_family = self._get_model_family(model_id)

            # Prepare request based on model family
            if model_family == "anthropic":
                request_body = self._prepare_anthropic_request(
                    messages, max_tokens, temperature, **kwargs
                )
            elif model_family in ["nova", "titan"]:
                request_body = self._prepare_titan_request(
                    messages, max_tokens, temperature, **kwargs
                )
            elif model_family == "llama":
                request_body = self._prepare_llama_request(
                    messages, max_tokens, temperature, **kwargs
                )
            elif model_family == "cohere":
                request_body = self._prepare_cohere_request(
                    messages, max_tokens, temperature, **kwargs
                )
            else:
                request_body = self._prepare_generic_request(
                    messages, max_tokens, temperature, **kwargs
                )

            # Make the API call
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )

            # Parse response based on model family
            response_body = json.loads(response["body"].read())

            if model_family == "anthropic":
                return response_body["content"][0]["text"]
            elif model_family in ["nova", "titan"]:
                return response_body["results"][0]["outputText"]
            elif model_family == "llama":
                return response_body["generation"]
            elif model_family == "cohere":
                return response_body["text"]
            else:
                # Try common response formats
                if "text" in response_body:
                    return response_body["text"]
                elif "content" in response_body:
                    return response_body["content"]
                elif "generation" in response_body:
                    return response_body["generation"]
                else:
                    return str(response_body)

        except (ClientError, BotoCoreError) as e:
            raise e
        except Exception as e:
            raise e

    async def call_ai_async(
        self,
        messages,
        temperature,
        max_tokens,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        **kwargs,
    ):
        # AWS Bedrock doesn't have native async support in boto3, so we'll use the sync version
        # In a production environment, you might want to use asyncio.run_in_executor
        return self.call_ai(messages, temperature, max_tokens, model, **kwargs)

    def generate(self, messages, temperature, max_tokens, model=None, **kwargs):
        # Check if model supports image generation
        if model and any(
            x in model for x in ["nova-canvas", "titan-image", "stable-diffusion"]
        ):
            # Handle image generation models
            raise NotImplementedError(
                "Image generation through Bedrock not yet implemented"
            )
        else:
            raise NotImplementedError("This provider does not support image generation")

    async def generate_async(
        self, messages, temperature, max_tokens, model=None, **kwargs
    ):
        return self.generate(messages, temperature, max_tokens, model, **kwargs)


class AIManager:
    def __init__(self):
        self.providers = {}

    def add_provider(self, provider):
        self.providers[provider.__class__.__name__] = provider

    def get_provider(self, model):
        for provider in self.providers.values():
            if model in provider.supported_models:
                return provider

            # Special handling for dynamic model providers
            if hasattr(provider, "__class__"):
                # Handle Azure OpenAI with dynamic deployments
                if (
                    provider.__class__.__name__ == "AzureOpenAIProvider"
                    and model.startswith("azure/")
                ):
                    return provider
                # Handle BedrockProvider with ARN-based inference profiles
                elif (
                    provider.__class__.__name__ == "BedrockProvider"
                    and model.startswith("bedrock/arn:aws:bedrock:")
                ):
                    return provider

        raise ValueError(f"No provider found for model: {model}")

    def call_ai(self, messages, temperature, max_tokens, model, **kwargs):
        provider = self.get_provider(model)
        if provider:
            """
            text_messages = [
                {
                    "role": msg["role"],
                    "content": msg["content"]
                    if isinstance(msg["content"], str)
                    else "",
                }
                for msg in messages
            ]
            """
            # Calculate prompt cost estimate
            try:
                # prompt_cost = float(calculate_prompt_cost(text_messages, model))
                prompt_cost = 0.0
            except Exception:
                prompt_cost = 0.0

            response = provider.call_ai(
                messages, temperature, max_tokens, model, **kwargs
            )

            # Calculate completion cost estimate
            try:
                # completion_cost = float(calculate_completion_cost(response, model))
                completion_cost = 0.0
            except Exception:
                completion_cost = 0.0

            total_cost = prompt_cost + completion_cost

            return response, total_cost
        else:
            raise ValueError(f"No provider found for model: {model}")

    # TODO shouldn't be 0 when no model...
    async def call_ai_async(self, messages, temperature, max_tokens, model, **kwargs):
        provider = self.get_provider(model)
        if provider:
            """
            text_messages = [
                {
                    "role": msg["role"],
                    "content": msg["content"]
                    if isinstance(msg["content"], str)
                    else "",
                }
                for msg in messages
            ]
            """
            # Calculate cost estimate
            try:
                # prompt_cost = float(calculate_prompt_cost(text_messages, model))
                prompt_cost = 0.0
            except Exception:
                prompt_cost = 0.0

            response = await provider.call_ai_async(
                messages, temperature, max_tokens, model, **kwargs
            )

            try:
                # completion_cost = float(calculate_completion_cost(response, model))
                completion_cost = 0.0
            except Exception:
                completion_cost = 0.0

            total_cost = prompt_cost + completion_cost

            return response, total_cost
        else:
            raise ValueError(f"No provider found for model: {model}")

    def generate(self, messages, temperature, max_tokens, model, **kwargs):
        provider = self.get_provider(model)
        if provider:
            try:
                prompt_cost = 0.0  # Cost calculation could be implemented later

                response = provider.generate(
                    messages, temperature, max_tokens, model, **kwargs
                )

                completion_cost = 0.0  # Cost calculation could be implemented later
                total_cost = prompt_cost + completion_cost

                return response, total_cost
            except NotImplementedError:
                raise ValueError(f"Model {model} does not support image generation")
        else:
            raise ValueError(f"No provider found for model: {model}")

    async def generate_async(self, messages, temperature, max_tokens, model, **kwargs):
        provider = self.get_provider(model)
        if provider:
            try:
                prompt_cost = 0.0  # Cost calculation could be implemented later

                response = await provider.generate_async(
                    messages, temperature, max_tokens, model, **kwargs
                )

                completion_cost = 0.0  # Cost calculation could be implemented later
                total_cost = prompt_cost + completion_cost

                return response, total_cost
            except NotImplementedError:
                raise ValueError(f"Model {model} does not support image generation")
        else:
            raise ValueError(f"No provider found for model: {model}")


@retry(
    retry=(
        retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIStatusError)
        | retry_if_exception_type(google_exceptions.DeadlineExceeded)
        | retry_if_exception_type(google_exceptions.ServiceUnavailable)
        | retry_if_exception_type(google_exceptions.ResourceExhausted)
        | retry_if_exception_type(anthropic.InternalServerError)
        | retry_if_exception_type(
            anthropic.APIStatusError
        )  # Catches HTTP 529 overloaded errors
        | retry_if_exception_type(anthropic.APITimeoutError)
        | retry_if_exception_type(anthropic.APIConnectionError)
        | retry_if_exception_type(anthropic.RateLimitError)
        | retry_if_exception_type(
            ClientError
        )  # AWS Bedrock errors including ThrottlingException
        | retry_if_exception_type(BotoCoreError)
    ),
    wait=wait_exponential(multiplier=2, min=2, max=120),
    stop=stop_after_attempt(3),
    reraise=True,
)
def call_ai_with_retry(ai_manager, messages, temperature, max_tokens, model, **kwargs):
    response, cost = ai_manager.call_ai(
        messages, temperature, max_tokens, model=model, **kwargs
    )
    return response, cost


@retry(
    retry=(
        retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIStatusError)
        | retry_if_exception_type(google_exceptions.DeadlineExceeded)
        | retry_if_exception_type(google_exceptions.ServiceUnavailable)
        | retry_if_exception_type(google_exceptions.ResourceExhausted)
        | retry_if_exception_type(anthropic.InternalServerError)
        | retry_if_exception_type(
            anthropic.APIStatusError
        )  # Catches HTTP 529 overloaded errors
        | retry_if_exception_type(anthropic.APITimeoutError)
        | retry_if_exception_type(anthropic.APIConnectionError)
        | retry_if_exception_type(anthropic.RateLimitError)
        | retry_if_exception_type(
            ClientError
        )  # AWS Bedrock errors including ThrottlingException
        | retry_if_exception_type(BotoCoreError)
    ),
    wait=wait_exponential(multiplier=2, min=2, max=120),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def call_ai_async_with_retry(
    ai_manager, messages, temperature, max_tokens, model, **kwargs
):
    response, cost = await ai_manager.call_ai_async(
        messages, temperature, max_tokens, model=model, **kwargs
    )
    return response, cost


# Add retry wrapper functions
@retry(
    retry=(
        retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIStatusError)
        | retry_if_exception_type(google_exceptions.DeadlineExceeded)
        | retry_if_exception_type(google_exceptions.ServiceUnavailable)
        | retry_if_exception_type(google_exceptions.ResourceExhausted)
        | retry_if_exception_type(anthropic.InternalServerError)
        | retry_if_exception_type(
            anthropic.APIStatusError
        )  # Catches HTTP 529 overloaded errors
        | retry_if_exception_type(anthropic.APITimeoutError)
        | retry_if_exception_type(anthropic.APIConnectionError)
        | retry_if_exception_type(anthropic.RateLimitError)
        | retry_if_exception_type(
            ClientError
        )  # AWS Bedrock errors including ThrottlingException
        | retry_if_exception_type(BotoCoreError)
    ),
    wait=wait_exponential(multiplier=2, min=2, max=120),
    stop=stop_after_attempt(3),
    reraise=True,
)
def generate_with_retry(ai_manager, messages, temperature, max_tokens, model, **kwargs):
    response, cost = ai_manager.generate(
        messages, temperature, max_tokens, model=model, **kwargs
    )
    return response, cost


@retry(
    retry=(
        retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIStatusError)
        | retry_if_exception_type(google_exceptions.DeadlineExceeded)
        | retry_if_exception_type(google_exceptions.ServiceUnavailable)
        | retry_if_exception_type(google_exceptions.ResourceExhausted)
        | retry_if_exception_type(anthropic.InternalServerError)
        | retry_if_exception_type(
            anthropic.APIStatusError
        )  # Catches HTTP 529 overloaded errors
        | retry_if_exception_type(anthropic.APITimeoutError)
        | retry_if_exception_type(anthropic.APIConnectionError)
        | retry_if_exception_type(anthropic.RateLimitError)
        | retry_if_exception_type(
            ClientError
        )  # AWS Bedrock errors including ThrottlingException
        | retry_if_exception_type(BotoCoreError)
    ),
    wait=wait_exponential(multiplier=2, min=2, max=120),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def generate_async_with_retry(
    ai_manager, messages, temperature, max_tokens, model, **kwargs
):
    response, cost = await ai_manager.generate_async(
        messages, temperature, max_tokens, model=model, **kwargs
    )
    return response, cost


class AIManagerSingleton:
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = AIManager()
                    # Initialize providers only when needed
                    if os.getenv("OPENAI_API_KEY"):
                        try:
                            instance.add_provider(OpenAIProvider())
                        except Exception as e:
                            print(f"Error adding OpenAI provider: {e}")

                    # Add Azure OpenAI provider
                    if os.getenv("AZURE_OPENAI_ENDPOINT") and os.getenv(
                        "AZURE_OPENAI_API_KEY"
                    ):
                        try:
                            instance.add_provider(AzureOpenAIProvider())
                        except Exception as e:
                            print(f"Error adding Azure OpenAI provider: {e}")

                    if os.getenv("ANTHROPIC_API_KEY"):
                        try:
                            instance.add_provider(AnthropicProvider())
                        except Exception as e:
                            print(f"Error adding Anthropic provider: {e}")

                    # Add Vertex AI provider if required environment variables are set
                    if (
                        VERTEX_AVAILABLE
                        and os.getenv("VERTEX_PROJECT_ID")
                        and os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                    ):
                        try:
                            instance.add_provider(VertexAIProvider())
                        except Exception as e:
                            print(f"Error adding Vertex AI provider: {e}")

                    if os.getenv("GOOGLE_API_KEY"):
                        try:
                            instance.add_provider(GeminiProvider())
                        except Exception as e:
                            print(f"Error adding Gemini provider: {e}")
                    if os.getenv("DEEPSEEK_API_KEY"):
                        try:
                            instance.add_provider(DeepSeekProvider())
                        except Exception as e:
                            print(f"Error adding DeepSeek provider: {e}")

                    # Add Bedrock provider if boto3 is available and AWS credentials are set
                    if BEDROCK_AVAILABLE and (
                        (
                            os.getenv("AWS_ACCESS_KEY_ID")
                            and os.getenv("AWS_SECRET_ACCESS_KEY")
                        )
                        or os.getenv("AWS_PROFILE")
                        or os.getenv("AWS_DEFAULT_REGION")
                    ):
                        try:
                            bedrock_region = os.getenv(
                                "AWS_DEFAULT_REGION", "us-east-1"
                            )
                            instance.add_provider(
                                BedrockProvider(region_name=bedrock_region)
                            )
                        except Exception as e:
                            print(f"Error adding Bedrock provider: {e}")

                    if os.getenv("LMSTUDIO_IP") and os.getenv("LMSTUDIO_PORT"):
                        try:
                            instance.add_provider(
                                LMStudioProvider(
                                    ip_address=os.getenv(
                                        "LMSTUDIO_IP", "192.168.11.34"
                                    ),
                                    port=int(os.getenv("LMSTUDIO_PORT", "1234")),
                                )
                            )
                        except Exception as e:
                            print(f"Error adding LMStudio provider: {e}")

                    cls._instance = instance

        return cls._instance


def call_ai(
    model: str,
    messages: List[Message],
    temperature: float = 0.1,
    max_tokens: int = 4096,
    **kwargs: Dict[str, Any],
) -> Tuple[str, float]:
    ai_manager = AIManagerSingleton.get_instance()
    return call_ai_with_retry(
        ai_manager, messages, temperature, max_tokens, model, **kwargs
    )


def call_ai_async(
    model: str,
    messages: List[Message],
    temperature: float = 0.1,
    max_tokens: int = 4096,
    **kwargs: Dict[str, Any],
) -> Tuple[str, float]:
    ai_manager = AIManagerSingleton.get_instance()
    return call_ai_async_with_retry(
        ai_manager, messages, temperature, max_tokens, model, **kwargs
    )


def generate(
    model: str,
    messages: List[Message],
    temperature: float = 0.1,
    max_tokens: int = 4096,
    **kwargs: Dict[str, Any],
) -> Tuple[str, float]:
    ai_manager = AIManagerSingleton.get_instance()
    return generate_with_retry(
        ai_manager, messages, temperature, max_tokens, model, **kwargs
    )


async def generate_async(
    model: str,
    messages: List[Message],
    temperature: float = 0.1,
    max_tokens: int = 4096,
    **kwargs: Dict[str, Any],
) -> Tuple[str, float]:
    ai_manager = AIManagerSingleton.get_instance()
    return generate_async_with_retry(
        ai_manager, messages, temperature, max_tokens, model, **kwargs
    )
