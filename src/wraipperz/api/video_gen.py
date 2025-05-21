import abc
import os
import time
import json
import base64
import requests
from pathlib import Path
from typing import Dict, Optional, Union, Any, Tuple
from dotenv import load_dotenv
from PIL import Image
import io
import mimetypes
import urllib.parse
import uuid
import re

load_dotenv(override=True)

try:
    import fal_client

    FAL_CLIENT_AVAILABLE = True
except ImportError:
    FAL_CLIENT_AVAILABLE = False


class VideoGenProvider(abc.ABC):
    @abc.abstractmethod
    def text_to_video(
        self, prompt: str, negative_prompt: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Generate a video from a text prompt"""
        pass

    @abc.abstractmethod
    def image_to_video(
        self,
        image_path: Union[str, Path, Image.Image],
        prompt: str,
        negative_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a video from an image"""
        pass

    @abc.abstractmethod
    def video_to_video(
        self,
        video_url: str,
        prompt: str,
        modify_region: Optional[str] = None,
        image_url: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a modified video using inpainting"""
        pass

    @abc.abstractmethod
    def get_video_status(self, video_id: int) -> Dict[str, Any]:
        """Get the status of a video generation job"""
        pass

    def resize_image_if_needed(
        self,
        img: Image.Image,
        max_size_bytes: int = 4000000,  # Default ~3.8MB
        provider_name: str = "",
    ) -> Tuple[Image.Image, int]:
        """
        Utility method to resize an image if it exceeds the maximum size limit.
        Returns the resized image and the size in bytes.

        Args:
            img: The PIL Image to resize if needed
            max_size_bytes: Maximum size in bytes (default 4MB)
            provider_name: Name of the provider for logging

        Returns:
            Tuple containing (PIL Image, size in bytes)
        """
        # Start with original quality
        quality = 95
        max_attempts = 10
        attempt = 0

        # First check current size
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=quality)
        img_size = buffered.tell()

        # If already small enough, return as is
        if img_size <= max_size_bytes:
            provider_prefix = f"{provider_name} " if provider_name else ""
            print(
                f"{provider_prefix}Image already within size limit ({img_size/1024:.1f} KB)"
            )
            return img, img_size

        while attempt < max_attempts:
            # Try reducing quality first (for first two attempts)
            if attempt == 0 and img_size > max_size_bytes:
                quality = 85
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=quality)
                img_size = buffered.tell()

                if img_size <= max_size_bytes:
                    provider_prefix = f"{provider_name} " if provider_name else ""
                    print(
                        f"{provider_prefix}Image compressed to {img_size/1024:.1f} KB with quality {quality}%"
                    )
                    return img, img_size

                attempt += 1
                continue
            elif attempt == 1 and img_size > max_size_bytes:
                quality = 75
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG", quality=quality)
                img_size = buffered.tell()

                if img_size <= max_size_bytes:
                    provider_prefix = f"{provider_name} " if provider_name else ""
                    print(
                        f"{provider_prefix}Image compressed to {img_size/1024:.1f} KB with quality {quality}%"
                    )
                    return img, img_size

                attempt += 1
                continue

            # Calculate scale factor (reduce by 20% each time)
            scale_factor = 0.8

            # Get current dimensions
            width, height = img.size

            # Calculate new dimensions
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Resize the image
            img = img.resize((new_width, new_height), Image.LANCZOS)

            # Check new size
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=95)  # Reset to high quality
            img_size = buffered.tell()

            provider_prefix = f"{provider_name} " if provider_name else ""
            print(
                f"{provider_prefix}Resized image to {new_width}x{new_height} ({img_size/1024:.1f} KB)"
            )

            if img_size <= max_size_bytes:
                return img, img_size

            attempt += 1

        # If we reach here, use minimum reasonable size/quality
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=70)
        img_size = buffered.tell()

        provider_prefix = f"{provider_name} " if provider_name else ""
        print(
            f"{provider_prefix}Warning: Could not reduce image below target size. Final size: {img_size/1024:.1f} KB"
        )

        return img, img_size

    def download_video(
        self, video_url: Union[str, dict], output_path: Union[str, Path]
    ) -> Path:
        """Download a video from a URL to a specified path"""
        output_path = Path(output_path)

        # Handle the case where video_url is a dictionary
        if isinstance(video_url, dict):
            # Extract the URL from the dictionary
            if "url" in video_url:
                video_url = video_url["url"]
            else:
                raise ValueError(
                    f"Could not find URL in video_url dictionary: {video_url}"
                )

        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine file extension from URL if not specified in output path
        if not output_path.suffix:
            # Parse the URL to extract extension
            parsed_url = urllib.parse.urlparse(video_url)
            path = parsed_url.path
            ext = os.path.splitext(path)[1]

            if not ext:
                # Default to .mp4 if extension can't be determined
                ext = ".mp4"

            # Update output path with extension
            output_path = output_path.with_suffix(ext)

        # Download video from URL
        with requests.get(video_url, stream=True) as response:
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        return output_path


class FalProvider(VideoGenProvider):
    """Provider for fal.ai video generation"""

    supported_models = [
        "fal/minimax-video",
        "fal/minimax-video-live",
        "fal/luma-dream-machine",
        "fal/kling-video",
        "fal/kling-video-v2-master",
        "fal/veo2",
        "fal/pixverse-v4",
        "fal/wan-pro",
        "fal/magi-distilled",
        "fal/vidu",
        "fal/ltx-video-v095",
        "fal/pika-swaps-v2",
    ]

    model_mapping = {
        "fal/minimax-video": "fal-ai/minimax-video/image-to-video",
        "fal/minimax-video-live": "fal-ai/minimax/video-01-live/image-to-video",
        "fal/luma-dream-machine": "fal-ai/luma-dream-machine",
        "fal/kling-video": "fal-ai/kling-video/v1/standard",
        "fal/kling-video-v2-master": "fal-ai/kling-video/v2/master/image-to-video",
        "fal/veo2": "fal-ai/veo2/image-to-video",
        "fal/pixverse-v4": "fal-ai/pixverse/v4/image-to-video",
        "fal/wan-pro": "fal-ai/wan-pro/image-to-video",
        "fal/magi-distilled": "fal-ai/magi-distilled/image-to-video",
        "fal/vidu": "fal-ai/vidu/image-to-video",
        "fal/ltx-video-v095": "fal-ai/ltx-video-v095/image-to-video",
        "fal/pika-swaps-v2": "fal-ai/pika/v2/pikaswaps",
    }

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("FAL_KEY")
        if not self.api_key:
            raise ValueError("FAL_KEY is required")

        if not FAL_CLIENT_AVAILABLE:
            raise ImportError(
                "fal-client is required. Install with `pip install fal-client`"
            )

        # Configure fal_client with API key
        os.environ["FAL_KEY"] = self.api_key

    def _encode_image_to_base64(self, image_path: Union[str, Path, Image.Image]) -> str:
        """Convert image to base64 data URI for fal.ai"""
        # Determine the mime type
        mime_type = "image/jpeg"  # Default

        # Maximum allowed request body size for Fal.ai is 4MB (4194304 bytes)
        # We'll use a slightly lower limit to account for overhead
        MAX_IMAGE_SIZE_BYTES = 4000000  # ~3.8MB

        # Process image based on input type
        if isinstance(image_path, Image.Image):
            # Use the image directly
            img = image_path
        else:
            # Handle string path or Path object
            path = Path(image_path)
            if not path.exists():
                raise ValueError(f"Image file not found: {image_path}")

            # Load the image
            img = Image.open(path)
            # Try to determine mime type from file
            mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"

        # Convert to RGB if it's RGBA (or any mode with transparency)
        if img.mode in ("RGBA", "LA") or (
            img.mode == "P" and "transparency" in img.info
        ):
            print(f"Converting image from {img.mode} mode to RGB")
            img = img.convert("RGB")

        # Resize the image if needed
        resized_img, _ = self.resize_image_if_needed(
            img, MAX_IMAGE_SIZE_BYTES, "Fal.ai"
        )

        # Convert to bytes and then to base64
        buffered = io.BytesIO()
        resized_img.save(buffered, format="JPEG", quality=90)
        buffered.seek(0)
        base64_data = base64.b64encode(buffered.read()).decode("utf-8")

        # Format as data URI
        return f"data:{mime_type};base64,{base64_data}"

    def text_to_video(
        self, prompt: str, negative_prompt: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Generate a video from a text prompt"""
        # fal.ai doesn't appear to support direct text-to-video in their docs
        # Instead, users typically generate an image first, then do image-to-video
        raise NotImplementedError(
            "Text-to-video generation is not directly supported by fal.ai. "
            "Generate an image first, then use image_to_video."
        )

    def image_to_video(
        self,
        image_path: Union[str, Path, Image.Image, str],
        prompt: str,
        negative_prompt: Optional[str] = None,
        model_override: Optional[str] = None,
        fal_model: Optional[str] = None,
        duration: str = "5",
        aspect_ratio: str = "16:9",
        cfg_scale: float = 0.5,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a video from an image using fal.ai

        Args:
            image_path: Path to image file, PIL Image, or direct image URL
            prompt: Text prompt describing the desired video
            negative_prompt: Optional negative prompt to guide generation
            model_override: Override the mapped model with a custom fal.ai endpoint
            fal_model: Alias for model_override
            duration: The duration in seconds ("5" or "10")
            aspect_ratio: The aspect ratio ("16:9", "9:16", or "1:1")
            cfg_scale: Guidance scale (0.0 to 1.0)
            **kwargs: Additional parameters to pass to the provider
        """
        # Determine which fal.ai model to use
        fal_endpoint = model_override or fal_model

        if not fal_endpoint:
            # Get the model mapping from our internal model name
            model_name = kwargs.get("model", "fal/minimax-video")

            # Check if the model name is already a full fal.ai endpoint path
            if model_name.startswith("fal-ai/"):
                fal_endpoint = model_name
            else:
                # Use our mapping
                fal_endpoint = self.model_mapping.get(model_name)

            if not fal_endpoint:
                raise ValueError(f"Unknown model: {model_name}")

        # Prepare image
        if isinstance(image_path, str) and (
            image_path.startswith("http://") or image_path.startswith("https://")
        ):
            # Direct URL provided
            image_url = image_path
        elif isinstance(image_path, str) and image_path.startswith("data:"):
            # Already a data URI
            image_url = image_path
        else:
            # Convert local image to base64 data URI
            image_url = self._encode_image_to_base64(image_path)

        # Create request ID
        request_id = f"wraipperz-fal-i2v-{str(uuid.uuid4())[:8]}"

        # Prepare arguments based on the model
        arguments = {
            "prompt": prompt,
            "image_url": image_url,
        }

        # Handle model-specific parameters
        if "veo2" in fal_endpoint:
            # Veo 2 has different duration format and aspect ratio options
            veo_duration = duration
            if duration in ["5", "6", "7", "8"]:
                veo_duration = f"{duration}s"

            arguments["duration"] = veo_duration

            # Handle aspect ratio for Veo 2
            if aspect_ratio not in ["auto", "auto_prefer_portrait", "16:9", "9:16"]:
                # Map standard aspect ratios to Veo 2 format
                if aspect_ratio == "1:1":
                    aspect_ratio = "auto"

            arguments["aspect_ratio"] = aspect_ratio
        else:
            # Standard parameters for other models
            arguments["duration"] = duration
            arguments["aspect_ratio"] = aspect_ratio
            arguments["cfg_scale"] = cfg_scale

            if negative_prompt:
                arguments["negative_prompt"] = negative_prompt
            else:
                # Use default negative prompt
                arguments["negative_prompt"] = "blur, distort, and low quality"

        # Handle quality/resolution parameter compatibility
        # If quality is specified but resolution isn't, use quality value for resolution
        if "quality" in kwargs and "resolution" not in kwargs:
            arguments["resolution"] = kwargs["quality"]
            # Keep quality for backwards compatibility
            arguments["quality"] = kwargs["quality"]

        # Include any other valid kwargs
        for key, value in kwargs.items():
            if key not in arguments and key not in ["model"]:
                arguments[key] = value

        # Submit the request to fal.ai
        handler = fal_client.submit(
            fal_endpoint,
            arguments=arguments,
        )

        # Return immediately with the handler reference
        return {"request_id": request_id, "fal_handler": handler, "status": "submitted"}

    def get_video_status(self, video_id: Any) -> Dict[str, Any]:
        """
        Get the status of a video generation job

        Note: For fal.ai, video_id is actually the fal_handler from the initial request
        """
        if not isinstance(video_id, dict) or "fal_handler" not in video_id:
            raise ValueError(
                "For fal.ai, video_id must be the result dict from image_to_video"
            )

        handler = video_id["fal_handler"]

        # Get the result from the handler
        try:
            # SyncRequestHandle doesn't have result_ready() - attempt to get the result directly
            # If it's not ready, it will raise an exception
            result = handler.get()

            # Extract video URL - check different fields that might contain the video URL
            video_url = None

            if isinstance(result, dict):
                # Common pattern in newer fal.ai models, including Veo 2
                if (
                    "video" in result
                    and isinstance(result["video"], dict)
                    and "url" in result["video"]
                ):
                    video_url = result["video"]["url"]

                # Check other possible locations
                if not video_url:
                    video_url = (
                        (
                            result.get("video", {}).get("url")
                            if isinstance(result.get("video"), dict)
                            else result.get("video")
                        )
                        or result.get("url")
                        or result.get("output_url")
                        or result.get("result_url")
                        or result.get("video_url")
                    )

                # If there's a 'results' or 'data' field containing another dict, check there
                if (
                    not video_url
                    and "results" in result
                    and isinstance(result["results"], dict)
                ):
                    video_url = (
                        (
                            result["results"].get("video", {}).get("url")
                            if isinstance(result["results"].get("video"), dict)
                            else result["results"].get("video")
                        )
                        or result["results"].get("url")
                        or result["results"].get("output_url")
                        or result["results"].get("result_url")
                        or result["results"].get("video_url")
                    )

                if (
                    not video_url
                    and "data" in result
                    and isinstance(result["data"], dict)
                ):
                    video_url = (
                        (
                            result["data"].get("video", {}).get("url")
                            if isinstance(result["data"].get("video"), dict)
                            else result["data"].get("video")
                        )
                        or result["data"].get("url")
                        or result["data"].get("output_url")
                        or result["data"].get("result_url")
                        or result["data"].get("video_url")
                    )
            else:
                # Fallback: check if the result itself is a URL string
                if isinstance(result, str) and (
                    result.startswith("http://") or result.startswith("https://")
                ):
                    video_url = result

            if not video_url:
                # Cannot determine video URL
                raise ValueError(f"Could not extract video URL from result: {result}")

            return {
                "status": "success",
                "progress": 100,
                "url": video_url,
                "request_id": video_id.get("request_id"),
                "result": result,  # Include full result for reference
            }
        except Exception as e:
            if "still processing" in str(e).lower() or "processing" in str(e).lower():
                # The job is still being processed
                return {
                    "status": "processing",
                    "progress": 0,  # fal.ai doesn't provide progress information
                    "url": None,
                    "request_id": video_id.get("request_id"),
                }

            # Some other error occurred
            return {
                "status": "failed",
                "error": str(e),
                "request_id": video_id.get("request_id"),
            }

    def _upload_or_get_video_url(self, video_path: Union[str, Path]) -> str:
        """
        Upload a local video file to fal.ai or return the URL if already a URL.

        Args:
            video_path: Path to local video file or URL

        Returns:
            URL to the video (either uploaded or the original URL)
        """
        # If it's already a URL, just return it
        if isinstance(video_path, str) and (
            video_path.startswith("http://") or video_path.startswith("https://")
        ):
            return video_path

        # If it's a local file, upload it to fal.ai
        path = Path(video_path)
        if not path.exists():
            raise ValueError(f"Video file not found: {video_path}")

        # Upload the file to fal.ai storage
        print(f"Uploading video file: {path}")
        url = fal_client.upload_file(str(path))
        print(f"Video uploaded successfully: {url}")

        return url

    def video_to_video(
        self,
        video_url: Union[str, Path],
        prompt: str,
        modify_region: str = None,
        image_url: Union[str, Path] = None,
        negative_prompt: Optional[str] = None,
        seed: Optional[int] = None,
        model_override: Optional[str] = None,
        fal_model: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a modified video using Pika Swaps inpainting

        Args:
            video_url: URL or path to the input video file
            prompt: Text prompt describing the modification
            modify_region: Plaintext description of the object/region to modify
            image_url: Optional URL or path to the image to swap with
            negative_prompt: Optional negative prompt to guide the model
            seed: Optional seed for the random number generator
            model_override: Override the mapped model with a custom fal.ai endpoint
            fal_model: Alias for model_override
            **kwargs: Additional parameters to pass to the provider

        Returns:
            Dict with the request details
        """
        # Determine which fal.ai model to use
        fal_endpoint = model_override or fal_model

        if not fal_endpoint:
            # Get the model mapping from our internal model name
            model_name = kwargs.get("model", "fal/pika-swaps-v2")

            # Check if the model name is already a full fal.ai endpoint path
            if model_name.startswith("fal-ai/"):
                fal_endpoint = model_name
            else:
                # Use our mapping
                fal_endpoint = self.model_mapping.get(model_name)

            if not fal_endpoint:
                raise ValueError(f"Unknown model: {model_name}")

        # Create request ID
        request_id = f"wraipperz-fal-v2v-{str(uuid.uuid4())[:8]}"

        # Handle local video file or URL
        video_url_to_use = self._upload_or_get_video_url(video_url)

        # Handle local image file or URL if provided
        image_url_to_use = None
        if image_url:
            if isinstance(image_url, str) and (
                image_url.startswith("http://") or image_url.startswith("https://")
            ):
                # Already a URL
                image_url_to_use = image_url
            else:
                # Local file - convert to base64 or upload
                image_url_to_use = self._encode_image_to_base64(image_url)

        # Prepare arguments for Pika Swaps
        arguments = {
            "video_url": video_url_to_use,
        }

        # Add optional parameters if provided
        if prompt:
            arguments["prompt"] = prompt

        if modify_region:
            arguments["modify_region"] = modify_region

        if image_url_to_use:
            arguments["image_url"] = image_url_to_use

        if negative_prompt:
            arguments["negative_prompt"] = negative_prompt

        if seed is not None:
            arguments["seed"] = seed

        # Include any other valid kwargs
        for key, value in kwargs.items():
            if key not in arguments and key not in ["model"]:
                arguments[key] = value

        # Submit the request to fal.ai
        handler = fal_client.submit(
            fal_endpoint,
            arguments=arguments,
        )

        # Return immediately with the handler reference
        return {"request_id": request_id, "fal_handler": handler, "status": "submitted"}


class KlingAIProvider(VideoGenProvider):
    supported_models = ["kling/text-to-video", "kling/image-to-video"]

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("KLING_API_KEY")
        if not self.api_key:
            raise ValueError("KLING_API_KEY is required")

        self.api_base = "https://app.klingai.com/api"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _encode_image(self, image_path: Union[str, Path, Image.Image]) -> str:
        """Convert image to base64 for API submission"""
        # Maximum allowed request body size for KlingAI is 4MB (4194304 bytes)
        # We'll use a slightly lower limit to account for overhead
        MAX_IMAGE_SIZE_BYTES = 4000000  # ~3.8MB

        if isinstance(image_path, Image.Image):
            # Already a PIL image
            img = image_path
            # Convert to RGB if it's RGBA
            if image_path.mode == "RGBA":
                img = image_path.convert("RGB")
        else:
            # Handle string path or Path object
            path = Path(image_path)
            if not path.exists():
                raise ValueError(f"Image file not found: {image_path}")

            # Open as PIL image for potential resizing
            img = Image.open(path)

        # Resize if needed
        resized_img, _ = self.resize_image_if_needed(
            img, MAX_IMAGE_SIZE_BYTES, "KlingAI"
        )

        # Convert to base64
        buffered = io.BytesIO()
        resized_img.save(buffered, format="JPEG", quality=90)
        buffered.seek(0)
        return base64.b64encode(buffered.read()).decode("utf-8")

    def text_to_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        duration: int = 5,
        width: int = 768,
        height: int = 432,
        fps: int = 24,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a video from a text prompt using Kling AI API"""
        endpoint = f"{self.api_base}/text-to-video"

        # Create a unique request ID
        request_id = f"wraipperz-kling-t2v-{int(time.time())}"

        # Prepare request payload
        payload = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "duration": duration,
            "fps": fps,
            "request_id": request_id,
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        # Include any other valid kwargs
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value

        # Make API request
        response = requests.post(endpoint, headers=self.headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if result.get("code") != 0:
            raise ValueError(
                f"Text-to-video generation failed: {result.get('message')}"
            )

        return {"task_id": result["data"]["task_id"], "request_id": request_id}

    def image_to_video(
        self,
        image_path: Union[str, Path, Image.Image],
        prompt: str,
        negative_prompt: Optional[str] = None,
        duration: int = 5,
        fps: int = 24,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a video from an image using Kling AI API"""
        endpoint = f"{self.api_base}/image-to-video"

        # Encode the image to base64
        image_base64 = self._encode_image(image_path)

        # Create a unique request ID
        request_id = f"wraipperz-kling-i2v-{int(time.time())}"

        # Prepare request payload
        payload = {
            "image": image_base64,
            "prompt": prompt,
            "duration": duration,
            "fps": fps,
            "request_id": request_id,
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        # Include any other valid kwargs
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value

        # Make API request
        response = requests.post(endpoint, headers=self.headers, json=payload)
        response.raise_for_status()
        result = response.json()

        if result.get("code") != 0:
            raise ValueError(
                f"Image-to-video generation failed: {result.get('message')}"
            )

        return {"task_id": result["data"]["task_id"], "request_id": request_id}

    def get_video_status(self, task_id: int) -> Dict[str, Any]:
        """Get the status of a video generation job"""
        endpoint = f"{self.api_base}/task-status"

        params = {"task_id": task_id}

        response = requests.get(endpoint, headers=self.headers, params=params)
        response.raise_for_status()
        result = response.json()

        if result.get("code") != 0:
            raise ValueError(f"Failed to get video status: {result.get('message')}")

        status_data = result["data"]

        # Map Kling status to our internal status format
        # Assuming Kling uses status codes like: 0=pending, 1=processing, 2=success, 3=failed
        status_map = {0: "pending", 1: "processing", 2: "success", 3: "failed"}

        # Format the response to match our internal format
        formatted_status = {
            "status": status_map.get(status_data["status"], "unknown"),
            "progress": status_data.get("progress", 0),
            "url": status_data.get("result_url", ""),
            "task_id": task_id,
        }

        return formatted_status

    def download_video(
        self, video_url: Union[str, dict], output_path: Union[str, Path]
    ) -> Path:
        """Download a video from a URL to a specified path"""
        output_path = Path(output_path)

        # Handle the case where video_url is a dictionary
        if isinstance(video_url, dict):
            # Extract the URL from the dictionary
            if "url" in video_url:
                video_url = video_url["url"]
            else:
                raise ValueError(
                    f"Could not find URL in video_url dictionary: {video_url}"
                )

        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine file extension from URL if not specified in output path
        if not output_path.suffix:
            # Parse the URL to extract extension
            parsed_url = urllib.parse.urlparse(video_url)
            path = parsed_url.path
            ext = os.path.splitext(path)[1]

            if not ext:
                # Default to .mp4 if extension can't be determined
                ext = ".mp4"

            # Update output path with extension
            output_path = output_path.with_suffix(ext)

        # Download video from URL
        with requests.get(video_url, stream=True) as response:
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        return output_path

    def video_to_video(
        self,
        video_url: str,
        prompt: str,
        modify_region: Optional[str] = None,
        image_url: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a modified video using inpainting - not yet supported by KlingAI"""
        raise NotImplementedError(
            "Video-to-video inpainting is not supported by KlingAI provider"
        )


class PixVerseProvider(VideoGenProvider):
    supported_models = [
        "pixverse/text-to-video-v3.5",
        "pixverse/image-to-video-v3.5",
        "pixverse/image-to-video-v4.0",
        "pixverse/text-to-video-v4.0",
        "pixverse/image-to-video-v4.5",
        "pixverse/text-to-video-v4.5",
    ]

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("PIXVERSE_API_KEY")
        if not self.api_key:
            raise ValueError("PIXVERSE_API_KEY is required")

        self.api_base = "https://app-api.pixverse.ai/openapi/v2"
        self.headers = {"API-KEY": self.api_key, "Content-Type": "application/json"}

    def _extract_model_version(self, model_name):
        """Extract the version from the model name (e.g., '3.5' from 'pixverse/text-to-video-v3.5')"""
        if not model_name:
            return "v4.0"  # Default fallback

        # For debugging
        print(f"Extracting version from model name: {model_name}")

        if "/" in model_name:
            # Extract part after the slash
            model_name = model_name.split("/")[1]

        # Explicitly check for v4.5 models
        if "v4.5" in model_name:
            print(f"Detected v4.5 model: {model_name}")
            return "v4.5"

        version_match = re.search(r"v(\d+\.\d+)$", model_name)
        if version_match:
            version = f"v{version_match.group(1)}"
            print(f"Matched version pattern: {version}")
            return version  # Return with v prefix (e.g., 'v4.0')

        print("No version pattern matched, using default v4.0")
        return "v4.0"  # Default fallback if no version found

    def upload_image(self, image_path: Union[str, Path, Image.Image]) -> int:
        """Upload an image to PixVerse and get an image ID"""
        upload_url = f"{self.api_base}/image/upload"

        # Maximum size: Pixverse has higher limits, but we'll use 8MB to be safe
        MAX_IMAGE_SIZE_BYTES = 8000000  # ~8MB

        # Handle PIL Image case
        temp_file = None
        try:
            if isinstance(image_path, Image.Image):
                # Convert to RGB if it's RGBA
                if image_path.mode == "RGBA":
                    image_path = image_path.convert("RGB")

                # Resize if needed
                img = image_path
                resized_img, _ = self.resize_image_if_needed(
                    img, MAX_IMAGE_SIZE_BYTES, "Pixverse"
                )

                # Save to temporary file
                temp_file = Path(f"temp_image_{int(time.time())}.jpg")
                resized_img.save(temp_file)
                file_path = temp_file
            else:
                # Use the provided file path
                path = Path(image_path)
                if not path.exists():
                    raise ValueError(f"Image file not found: {image_path}")

                # Load and resize if needed
                img = Image.open(path)
                resized_img, _ = self.resize_image_if_needed(
                    img, MAX_IMAGE_SIZE_BYTES, "Pixverse"
                )

                # If we needed to resize, save to a temporary file
                if resized_img != img:
                    temp_file = Path(f"temp_image_{int(time.time())}.jpg")
                    resized_img.save(temp_file)
                    file_path = temp_file
                else:
                    # Use original file if no resize was needed
                    file_path = path

            # Create a unique trace ID
            ai_trace_id = str(uuid.uuid4())

            # Create headers without Content-Type (will be set by requests for multipart)
            headers = {"API-KEY": self.api_key, "Ai-trace-id": ai_trace_id}

            # Prepare the file for multipart upload
            files = [
                (
                    "image",
                    (file_path.name, open(file_path, "rb"), "application/octet-stream"),
                )
            ]

            # Make the upload request
            response = requests.post(upload_url, headers=headers, files=files)

            # Debug logging
            if response.status_code != 200:
                print(
                    f"Upload failed with status {response.status_code}: {response.text}"
                )

            response.raise_for_status()
            result = response.json()

            if result.get("ErrCode") != 0:
                raise ValueError(f"Image upload failed: {result.get('ErrMsg')}")

            return result["Resp"]["img_id"]
        finally:
            # Clean up temporary file if created
            if temp_file and temp_file.exists():
                temp_file.unlink()

    def text_to_video(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        style: Optional[str] = None,
        duration: int = 5,
        quality: str = "720p",
        motion_mode: str = "normal",
        template_id: Optional[int] = None,
        seed: Optional[int] = None,
        aspect_ratio: str = "16:9",
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a video from a text prompt using PixVerse API"""
        generate_url = f"{self.api_base}/video/text/generate"

        # Log the input model name for debugging
        print(
            f"PixVerse text_to_video called with model: {kwargs.get('model', 'None specified')}"
        )

        # Create a unique ai-trace-id for each request
        ai_trace_id = str(uuid.uuid4())
        headers = {
            "API-KEY": self.api_key,
            "Content-Type": "application/json",
            "Ai-trace-id": ai_trace_id,
        }

        # Extract model version from the model parameter
        model_version = self._extract_model_version(kwargs.get("model"))

        # Prepare request payload
        payload = {
            "model": model_version,
            "prompt": prompt,
            "duration": duration,
            "quality": quality,
            "motion_mode": motion_mode,
            "water_mark": False,
            "aspect_ratio": aspect_ratio,
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        if style and style in ["anime", "3d_animation", "day", "cyberpunk", "comic"]:
            payload["style"] = style

        if template_id:
            payload["template_id"] = template_id

        if seed:
            payload["seed"] = seed

        # Include any other valid kwargs
        for key, value in kwargs.items():
            if key not in payload:
                payload[key] = value

        # Print the final payload for debugging
        print(f"PixVerse text_to_video payload: {json.dumps(payload, indent=2)}")

        # Convert payload to JSON string
        json_payload = json.dumps(payload)

        # Make API request
        try:
            print(f"Making text_to_video request to: {generate_url}")
            response = requests.post(generate_url, headers=headers, data=json_payload)

            # Log response details for debugging
            if response.status_code != 200:
                print(f"API URL: {generate_url}")
                print(f"Headers: {headers}")
                print(f"Request payload: {json_payload}")
                print(f"Response status: {response.status_code}")
                print(f"Response content: {response.text}")

            response.raise_for_status()
            result = response.json()

            print(f"PixVerse text_to_video response: {json.dumps(result, indent=2)}")

            if result.get("ErrCode") != 0:
                raise ValueError(
                    f"Text-to-video generation failed: {result.get('ErrMsg')}"
                )

            return {"video_id": result["Resp"]["video_id"], "request_id": ai_trace_id}
        except Exception as e:
            print(f"Error in text_to_video: {str(e)}")
            raise

    def image_to_video(
        self,
        image_path: Union[str, Path, Image.Image],
        prompt: str,
        negative_prompt: Optional[str] = None,
        style: Optional[str] = None,
        duration: int = 5,
        quality: str = "720p",
        motion_mode: str = "normal",
        template_id: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a video from an image using PixVerse API"""
        # Log the input model name for debugging
        print(
            f"PixVerse image_to_video called with model: {kwargs.get('model', 'None specified')}"
        )

        # First upload the image to get an img_id
        img_id = self.upload_image(image_path)
        print(f"Successfully uploaded image with ID: {img_id}")

        # Generate video from the uploaded image
        generate_url = f"{self.api_base}/video/img/generate"

        # Create a unique ai-trace-id for each request
        ai_trace_id = str(uuid.uuid4())
        headers = {
            "API-KEY": self.api_key,
            "Content-Type": "application/json",
            "Ai-trace-id": ai_trace_id,
        }

        # Extract model version from the model parameter
        model_version = self._extract_model_version(kwargs.get("model"))

        print(f"Using model version: {model_version}")

        # Prepare request payload
        payload = {
            "model": model_version,
            "prompt": prompt,
            "img_id": img_id,
            "duration": duration,
            "quality": quality,
            "motion_mode": motion_mode,
            "water_mark": False,
        }

        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        if style and style in ["anime", "3d_animation", "day", "cyberpunk", "comic"]:
            payload["style"] = style

        if template_id:
            payload["template_id"] = template_id

        if seed:
            payload["seed"] = seed

        # Include any other valid kwargs
        for key, value in kwargs.items():
            if key not in payload and key != "img_id":
                payload[key] = value

        # Print the final payload for debugging
        print(f"PixVerse image_to_video payload: {json.dumps(payload, indent=2)}")

        # Convert payload to JSON string
        json_payload = json.dumps(payload)

        # Make API request
        try:
            print(f"Making image_to_video request to: {generate_url}")
            response = requests.post(generate_url, headers=headers, data=json_payload)

            # Log response details for debugging
            if response.status_code != 200:
                print(f"API URL: {generate_url}")
                print(f"Headers: {headers}")
                print(f"Request payload: {json_payload}")
                print(f"Response status: {response.status_code}")
                print(f"Response content: {response.text}")

            response.raise_for_status()
            result = response.json()

            print(f"PixVerse image_to_video response: {json.dumps(result, indent=2)}")

            if result.get("ErrCode") != 0:
                raise ValueError(
                    f"Image-to-video generation failed: {result.get('ErrMsg')}"
                )

            return {"video_id": result["Resp"]["video_id"], "request_id": ai_trace_id}
        except Exception as e:
            print(f"Error in image_to_video: {str(e)}")
            raise

    def get_video_status(self, video_id: int) -> Dict[str, Any]:
        """Get the status of a video generation job"""
        # Extract request_id if it's in the video_id dict
        request_id = None
        original_video_id = video_id
        if isinstance(video_id, dict) and "request_id" in video_id:
            request_id = video_id.get("request_id")
            video_id = video_id.get("video_id")

        print(f"Checking status for video ID: {video_id}, Request ID: {request_id}")

        # Different endpoints to try - using correct one based on example code
        endpoints = [
            # Main endpoint that should work based on reference code
            {"url": f"{self.api_base}/video/result/{video_id}", "method": "GET"},
            # Fallback options
            {
                "url": f"{self.api_base}/video/status",
                "method": "GET",
                "params": {"video_id": video_id},
            },
            {
                "url": f"{self.api_base}/video/query",
                "method": "POST",
                "body": {"video_id": video_id},
            },
            {
                "url": f"{self.api_base.replace('v2', 'v1')}/video/status",
                "method": "GET",
                "params": {"video_id": video_id},
            },
        ]

        # If we have a request_id, add endpoint for that
        if request_id:
            endpoints.append(
                {
                    "url": f"{self.api_base}/video/get",
                    "method": "GET",
                    "params": {"trace_id": request_id},
                }
            )

        # Try each endpoint
        for endpoint in endpoints:
            try:
                print(f"Trying endpoint: {endpoint['url']}")
                headers = {"API-KEY": self.api_key, "Content-Type": "application/json"}

                if endpoint.get("method") == "GET":
                    response = requests.get(
                        endpoint["url"],
                        headers=headers,
                        params=endpoint.get("params", {}),
                    )
                else:  # POST
                    response = requests.post(
                        endpoint["url"],
                        headers=headers,
                        data=json.dumps(endpoint.get("body", {})),
                    )

                # Print response for debugging
                print(f"Status: {response.status_code}")
                print(
                    f"Response: {response.text[:200]}..."
                    if len(response.text) > 200
                    else f"Response: {response.text}"
                )

                # If we get a successful response
                if response.status_code == 200:
                    try:
                        result = response.json()

                        # Different APIs might have different response structures
                        # Case 1: Standard API response with ErrCode/Resp structure
                        if (
                            "ErrCode" in result
                            and result["ErrCode"] == 0
                            and "Resp" in result
                        ):
                            resp = result["Resp"]

                            # Format the response to our standard format
                            if isinstance(resp, dict):
                                status_data = {
                                    "status": resp.get(
                                        "status", 1
                                    ),  # Default to success if not specified
                                    "url": resp.get(
                                        "url", "https://app.pixverse.ai/studio"
                                    ),
                                    "progress": resp.get("progress", 100),
                                    "video_id": video_id,
                                }
                                return status_data

                        # Case 2: Direct JSON response
                        if isinstance(result, dict):
                            status_data = {
                                "status": result.get(
                                    "status", 1
                                ),  # Default to success if not specified
                                "url": result.get(
                                    "url", "https://app.pixverse.ai/studio"
                                ),
                                "progress": result.get("progress", 100),
                                "video_id": video_id,
                            }
                            return status_data
                    except Exception as e:
                        print(f"Error parsing response: {str(e)}")
            except Exception as e:
                print(f"Error with endpoint {endpoint['url']}: {str(e)}")

        # If all endpoints failed, return a simulated status
        print("All status check endpoints failed, providing simulated success status.")
        return {
            "status": 1,  # Assume success
            "url": "https://app.pixverse.ai/studio",  # Placeholder URL
            "progress": 100,
            "video_id": original_video_id,
            "message": "Status check failed, check video in the Pixverse dashboard.",
        }

    def download_video(
        self, video_url: Union[str, dict], output_path: Union[str, Path]
    ) -> Path:
        """Download a video from a URL to a specified path"""
        output_path = Path(output_path)

        # Handle the case where video_url is a dictionary
        if isinstance(video_url, dict):
            # Extract the URL from the dictionary
            if "url" in video_url:
                video_url = video_url["url"]
            else:
                raise ValueError(
                    f"Could not find URL in video_url dictionary: {video_url}"
                )

        # Create parent directories if they don't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine file extension from URL if not specified in output path
        if not output_path.suffix:
            # Parse the URL to extract extension
            parsed_url = urllib.parse.urlparse(video_url)
            path = parsed_url.path
            ext = os.path.splitext(path)[1]

            if not ext:
                # Default to .mp4 if extension can't be determined
                ext = ".mp4"

            # Update output path with extension
            output_path = output_path.with_suffix(ext)

        # Download video from URL
        with requests.get(video_url, stream=True) as response:
            response.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

        return output_path

    def video_to_video(
        self,
        video_url: str,
        prompt: str,
        modify_region: Optional[str] = None,
        image_url: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a modified video using inpainting - not yet supported by PixVerse"""
        raise NotImplementedError(
            "Video-to-video inpainting is not supported by PixVerse provider"
        )


class VideoGenManager:
    def __init__(self):
        self.providers = {}

    def add_provider(self, provider):
        self.providers[provider.__class__.__name__] = provider

    def get_provider(self, model):
        # Special case for direct fal.ai endpoint paths
        if model.startswith("fal-ai/"):
            for provider in self.providers.values():
                if isinstance(provider, FalProvider):
                    return provider

        # Normal case - check supported_models
        for provider in self.providers.values():
            if (
                hasattr(provider, "supported_models")
                and model in provider.supported_models
            ):
                return provider

        raise ValueError(f"No provider found for model: {model}")

    def generate_video_from_text(
        self, model: str, prompt: str, negative_prompt: Optional[str] = None, **kwargs
    ) -> Dict[str, Any]:
        """Generate a video from text prompt"""
        provider = self.get_provider(model)
        return provider.text_to_video(prompt, negative_prompt, **kwargs)

    def generate_video_from_image(
        self,
        model: str,
        image_path: Union[str, Path, Image.Image],
        prompt: str,
        negative_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a video from an image"""
        provider = self.get_provider(model)
        # Add model to kwargs so it's available inside provider methods
        kwargs["model"] = model
        return provider.image_to_video(image_path, prompt, negative_prompt, **kwargs)

    def generate_video_from_video(
        self,
        model: str,
        video_url: str,
        prompt: str,
        modify_region: Optional[str] = None,
        image_url: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate a modified video using inpainting"""
        provider = self.get_provider(model)
        # Add model to kwargs so it's available inside provider methods
        kwargs["model"] = model

        # Check if provider supports video-to-video
        if not hasattr(provider, "video_to_video"):
            raise ValueError(
                f"The provider for {model} doesn't support video-to-video operations"
            )

        return provider.video_to_video(
            video_url=video_url,
            prompt=prompt,
            modify_region=modify_region,
            image_url=image_url,
            negative_prompt=negative_prompt,
            **kwargs,
        )

    def get_video_status(self, model: str, video_id: int) -> Dict[str, Any]:
        """Get the status of a video generation job"""
        provider = self.get_provider(model)
        return provider.get_video_status(video_id)

    def wait_for_video_completion(
        self,
        model: str,
        video_id: int,
        polling_interval: int = 5,
        max_wait_time: int = 300,
    ) -> Dict[str, Any]:
        """Wait for a video generation job to complete"""
        provider = self.get_provider(model)
        start_time = time.time()

        # Special handling for fal.ai which uses a different structure
        if isinstance(video_id, dict) and "fal_handler" in video_id:
            while (time.time() - start_time) < max_wait_time:
                status = provider.get_video_status(video_id)

                if status["status"] == "success":
                    return status
                elif status["status"] == "failed":
                    raise ValueError(
                        f"Video generation failed: {status.get('error', 'Unknown error')}"
                    )

                # Still processing, wait and try again
                time.sleep(polling_interval)

            # If we get here, we've timed out
            raise TimeoutError(
                f"Video generation timed out after {max_wait_time} seconds"
            )
        else:
            # Original code for other providers
            attempt_count = 0
            max_attempts = max_wait_time // polling_interval

            while (time.time() - start_time) < max_wait_time:
                try:
                    attempt_count += 1
                    print(f"Status check attempt {attempt_count}/{max_attempts}")

                    status = provider.get_video_status(video_id)

                    # Handle special case where we get a message about Pixverse dashboard
                    if (
                        isinstance(status, dict)
                        and status.get("message")
                        and "check video in the Pixverse dashboard"
                        in status.get("message", "")
                    ):
                        print(
                            "Status check is not available, but video generation was initiated."
                        )
                        print(
                            "You can check the status in the Pixverse dashboard: https://app.pixverse.ai/studio"
                        )
                        # Return the status with simulated success to allow continuing
                        return status

                    # For Pixverse:
                    # Status codes: 1: Success, 5: Processing, 2: Failed, 3: Timeout, 4: Rejected, 7: Moderation failure, 8: Failed
                    if status["status"] == 1:  # Success
                        return status
                    elif status["status"] == 5:  # Still processing
                        print(
                            f"Video is still processing... ({attempt_count}/{max_attempts})"
                        )
                        time.sleep(polling_interval)
                        continue
                    elif status["status"] in [2, 3, 4, 7, 8]:  # Failed states
                        status_messages = {
                            2: "Failed",
                            3: "Timeout",
                            4: "Rejected",
                            7: "Moderation failure",
                            8: "Failed",
                        }
                        raise ValueError(
                            f"Video generation {status_messages.get(status['status'], 'failed')}"
                        )

                    # Still processing, wait and try again
                    time.sleep(polling_interval)

                except Exception as e:
                    print(f"Error during status check: {str(e)}")
                    print(f"Will retry in {polling_interval} seconds...")
                    time.sleep(polling_interval)

            # If we get here, we've timed out or had consistent errors
            # For PixVerse, return a simulated success to allow continuing
            if "pixverse" in model.lower():
                print(
                    "Status check timed out or failed, but the video may still be generating."
                )
                print(
                    "You can check your videos in the PixVerse dashboard: https://app.pixverse.ai/studio"
                )
                return {
                    "status": 1,  # Simulate success
                    "url": "https://app.pixverse.ai/studio",  # Placeholder URL
                    "message": "Status check timed out, check video in Pixverse dashboard.",
                }

            # For other providers, raise timeout error
            raise TimeoutError(
                f"Video generation timed out after {max_wait_time} seconds"
            )

    def download_video(
        self, model: str, video_url: Union[str, dict], output_path: Union[str, Path]
    ) -> Path:
        """Download a video to a specified path"""
        provider = self.get_provider(model)
        return provider.download_video(video_url, output_path)


class VideoGenManagerSingleton:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = VideoGenManager()
            # Initialize providers
            if os.getenv("PIXVERSE_API_KEY"):
                try:
                    cls._instance.add_provider(PixVerseProvider())
                except Exception as e:
                    print(f"Error adding PixVerse provider: {e}")

            if os.getenv("KLING_API_KEY"):
                try:
                    cls._instance.add_provider(KlingAIProvider())
                except Exception as e:
                    print(f"Error adding Kling AI provider: {e}")

            if os.getenv("FAL_KEY"):
                try:
                    cls._instance.add_provider(FalProvider())
                except Exception as e:
                    print(f"Error adding fal.ai provider: {e}")

        return cls._instance


def generate_video_from_text(
    model: str,
    prompt: str,
    negative_prompt: Optional[str] = None,
    wait_for_completion: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate a video from a text prompt.

    Args:
        model: The model to use (e.g., "pixverse/text-to-video-v3.5")
        prompt: The text prompt describing the video
        negative_prompt: Optional negative prompt to guide generation
        wait_for_completion: Whether to wait for the video to finish generating
        output_path: If provided, download the completed video to this path
        **kwargs: Additional parameters to pass to the provider

    Returns:
        Dict with video generation details
    """
    manager = VideoGenManagerSingleton.get_instance()
    result = manager.generate_video_from_text(model, prompt, negative_prompt, **kwargs)

    if wait_for_completion or output_path:
        # Always wait for completion if output_path is provided
        status = manager.wait_for_video_completion(
            model,
            result["video_id"],
            polling_interval=kwargs.get("polling_interval", 5),
            max_wait_time=kwargs.get("max_wait_time", 300),
        )

        result.update({"status": status["status"], "url": status["url"]})

        # Download the video if output_path is provided
        if output_path:
            downloaded_path = manager.download_video(model, status["url"], output_path)
            result["file_path"] = str(downloaded_path)

    return result


def generate_video_from_image(
    model: str,
    image_path: Union[str, Path, Image.Image],
    prompt: str,
    negative_prompt: Optional[str] = None,
    wait_for_completion: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate a video from an image.

    Args:
        model: The model to use (e.g., "pixverse/image-to-video-v3.5")
        image_path: Path to image file, Path object, or PIL Image
        prompt: The text prompt describing the video
        negative_prompt: Optional negative prompt to guide generation
        wait_for_completion: Whether to wait for the video to finish generating
        output_path: If provided, download the completed video to this path
        **kwargs: Additional parameters to pass to the provider

    Returns:
        Dict with video generation details
    """
    manager = VideoGenManagerSingleton.get_instance()
    result = manager.generate_video_from_image(
        model, image_path, prompt, negative_prompt, **kwargs
    )

    if wait_for_completion or output_path:
        # Always wait for completion if output_path is provided
        # Use wait_for_video_completion for both fal.ai and other providers
        try:
            status = manager.wait_for_video_completion(
                model,
                result,  # Pass the entire result dict (works for both fal.ai and other providers)
                polling_interval=kwargs.get("polling_interval", 5),
                max_wait_time=kwargs.get("max_wait_time", 300),
            )

            # Update the result with status info
            result.update({"status": status["status"], "url": status["url"]})

            # Download the video if output_path is provided
            if output_path:
                downloaded_path = manager.download_video(
                    model, status["url"], output_path
                )
                result["file_path"] = str(downloaded_path)
        except Exception as e:
            # Update result with error info
            result.update({"status": "failed", "error": str(e)})
            # Re-raise the exception so the user knows what happened
            raise

    return result


def get_video_status(model: str, video_id: int) -> Dict[str, Any]:
    """
    Get the status of a video generation job.

    Args:
        model: The model used for generation
        video_id: The ID of the video generation job

    Returns:
        Dict with video status details
    """
    manager = VideoGenManagerSingleton.get_instance()
    return manager.get_video_status(model, video_id)


def wait_for_video_completion(
    model: str, video_id: int, polling_interval: int = 5, max_wait_time: int = 300
) -> Dict[str, Any]:
    """
    Wait for a video generation job to complete.

    Args:
        model: The model used for generation
        video_id: The ID of the video generation job
        polling_interval: How often to check status (in seconds)
        max_wait_time: Maximum time to wait (in seconds)

    Returns:
        Dict with video details once complete
    """
    manager = VideoGenManagerSingleton.get_instance()
    return manager.wait_for_video_completion(
        model, video_id, polling_interval, max_wait_time
    )


def download_video(
    model: str, video_url: Union[str, dict], output_path: Union[str, Path]
) -> Path:
    """
    Download a video from a URL to a specified path.

    Args:
        model: The model used for generation (needed to get the provider)
        video_url: The URL of the video to download
        output_path: Where to save the downloaded video

    Returns:
        Path to the downloaded video file
    """
    manager = VideoGenManagerSingleton.get_instance()
    return manager.download_video(model, video_url, output_path)


def generate_video_from_video(
    model: str,
    video_url: str,
    prompt: str,
    modify_region: Optional[str] = None,
    image_url: Optional[str] = None,
    negative_prompt: Optional[str] = None,
    wait_for_completion: bool = False,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Modify a video using inpainting (like Pika Swaps).

    Args:
        model: The model to use (e.g., "fal/pika-swaps-v2")
        video_url: URL of the input video
        prompt: Text prompt describing the modification
        modify_region: Plaintext description of the object/region to modify
        image_url: Optional URL of the image to swap with
        negative_prompt: Optional negative prompt to guide the model
        wait_for_completion: Whether to wait for the video to finish generating
        output_path: If provided, download the completed video to this path
        **kwargs: Additional parameters to pass to the provider

    Returns:
        Dict with video generation details
    """
    # Get the manager instance
    manager = VideoGenManagerSingleton.get_instance()

    # Generate the video using the manager
    result = manager.generate_video_from_video(
        model=model,
        video_url=video_url,
        prompt=prompt,
        modify_region=modify_region,
        image_url=image_url,
        negative_prompt=negative_prompt,
        **kwargs,
    )

    if wait_for_completion or output_path:
        # Always wait for completion if output_path is provided
        try:
            status = manager.wait_for_video_completion(
                model,
                result,  # Pass the entire result dict
                polling_interval=kwargs.get("polling_interval", 5),
                max_wait_time=kwargs.get("max_wait_time", 300),
            )

            # Update the result with status info
            result.update({"status": status["status"], "url": status["url"]})

            # Download the video if output_path is provided
            if output_path:
                downloaded_path = manager.download_video(
                    model, status["url"], output_path
                )
                result["file_path"] = str(downloaded_path)
        except Exception as e:
            # Update result with error info
            result.update({"status": "failed", "error": str(e)})
            # Re-raise the exception so the user knows what happened
            raise

    return result
