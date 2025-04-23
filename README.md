# wraipperz

Easy wrapper for various AI APIs including LLMs, ASR, TTS, and Video Generation.

## Installation

Basic installation:

```bash
pip install wraipperz
uv add wraipperz
```

With optional dependencies for specific providers:

```bash
# For fal.ai video generation
pip install wraipperz fal-client

# For all supported providers
pip install wraipperz "wraipperz[all]"
```

## Features

- **LLM API Wrappers**: Unified interface for OpenAI, Anthropic, Google, and other LLM providers
- **ASR (Automatic Speech Recognition)**: Convert speech to text
- **TTS (Text-to-Speech)**: Convert text to speech
- **Video Generation**: Text-to-video and image-to-video generation
- **Async Support**: Asynchronous API calls for improved performance

## Quick Start

### LLM

```python
import os
from wraipperz import call_ai, MessageBuilder

os.environ["OPENAI_API_KEY"] = "your_openai_key" # if not defined in environment variables
messages = MessageBuilder().add_system("You are a helpful assistant.").add_user("What's 1+1?")

# Call an LLM with a simple interface
response, cost = call_ai(
    model="openai/gpt-4o",
    messages=messages
)
```

Parsing LLM output to pydantic object.

```python
from pydantic import BaseModel, Field
from wraipperz import pydantic_to_yaml_example, find_yaml, MessageBuilder, call_ai
import yaml


class User(BaseModel):
    name: str = Field(json_schema_extra={"example": "Bob", "comment": "The name of the character."})
    age: int = Field(json_schema_extra={"example": 12, "comment": "The age of the character."})


template = pydantic_to_yaml_example(User)
prompt = f"""Extract the user's name and age from the unstructured text provided below and output your answer following the provided example.
Text: "John is a well respected 31 years old pirate who really likes mooncakes."
Exampe output:
\`\`\`yaml
{template}
\`\`\`
"""
messages = MessageBuilder().add_system(prompt).build()
response, cost = call_ai(model="openai/gpt-4o-mini", messages=messages)

yaml_content = find_yaml(response)
user = User(**yaml.safe_load(yaml_content))
print(user)  # prints name='John' age=31
```

### Image Generation and Modification (todo check readme)

```python
from wraipperz import generate, MessageBuilder
from PIL import Image

# Text-to-image generation
messages = MessageBuilder().add_user("Generate an image of a futuristic city skyline at sunset.").build()

result, cost = generate(
    model="gemini/gemini-2.0-flash-exp-image-generation",
    messages=messages,
    temperature=0.7,
    max_tokens=4096
)

# The result contains both text and images
print(result["text"])  # Text description/commentary from the model

# Save the generated images
for i, image in enumerate(result["images"]):
    image.save(f"generated_city_{i}.png")
    # image.show()  # Uncomment to display the image

# Image modification with input image
input_image = Image.open("input_photo.jpg")  # Replace with your image path

image_messages = MessageBuilder().add_user("Add a futuristic flying car to this image.").add_image(input_image).build()

result, cost = generate(
    model="gemini/gemini-2.0-flash-exp-image-generation",
    messages=image_messages,
    temperature=0.7,
    max_tokens=4096
)

# Save the modified images
for i, image in enumerate(result["images"]):
    image.save(f"modified_image_{i}.png")
```

The `generate` function returns a dictionary containing both textual response and generated images, enabling multimodal AI capabilities in your applications.

### Video Generation

```python
import os
from wraipperz import generate_video_from_text, generate_video_from_image, wait_for_video_completion
from PIL import Image

# Set your API key
os.environ["PIXVERSE_API_KEY"] = "your_pixverse_key"

# Text-to-Video Generation with automatic download
result = generate_video_from_text(
    model="pixverse/text-to-video-v3.5",
    prompt="A serene mountain lake at sunrise, with mist rising from the water.",
    negative_prompt="blurry, distorted, low quality, text, watermark",
    duration=5,  # 5 seconds
    quality="720p",
    style="3d_animation",  # Optional: "anime", "3d_animation", "day", "cyberpunk", "comic"
    wait_for_completion=True,  # Wait for the video to complete
    output_path="videos/mountain_lake"  # Extension (.mp4) will be added automatically
)

print(f"Video downloaded to: {result['file_path']}")
print(f"Video URL: {result['url']}")

# Image-to-Video Generation
# Load an image
image = Image.open("your_image.jpg")

# Convert the image to a video with motion and download automatically
result = generate_video_from_image(
    model="pixverse/image-to-video-v3.5",
    image_path=image,  # Can also be a file path string
    prompt="Add gentle motion and waves to this image",
    duration=5,
    quality="720p",
    output_path="videos/animated_image.mp4"  # Specify full path with extension
)

print(f"Video downloaded to: {result['file_path']}")
```


#### Using fal.ai for Video Generation

```python
import os
from wraipperz import generate_video_from_image
from PIL import Image

# Set your API key
os.environ["FAL_KEY"] = "your_fal_key"

# Works with local image paths (auto-encoded as base64)
result = generate_video_from_image(
    model="fal/kling-video-v2-master",  # Using Kling 2.0 Master
    image_path="path/to/your/local/image.jpg",  # Local image path
    prompt="A beautiful mountain scene with gentle motion in the clouds and water",
    duration="5",  # "5" or "10" seconds
    aspect_ratio="16:9",  # "16:9", "9:16", or "1:1"
    wait_for_completion=True,
    output_path="videos/fal_mountain_scene.mp4"
)

print(f"Video downloaded to: {result['file_path']}")

# Works directly with PIL Image objects
pil_image = Image.open("path/to/your/image.jpg")
result = generate_video_from_image(
    model="fal/minimax-video",  # Options: fal/minimax-video, fal/luma-dream-machine, fal/kling-video
    image_path=pil_image,  # PIL Image object
    prompt="Gentle ocean waves with clouds moving in the sky",
    wait_for_completion=True,
    output_path="videos/fal_ocean_scene"  # Extension will be added automatically
)

print(f"Video downloaded to: {result['file_path']}")

# You can also still use image URLs if you prefer
result = generate_video_from_image(
    model="fal/kling-video-v2-master",
    image_path="https://example.com/your-image.jpg",  # Web URL
    prompt="A colorful autumn scene with leaves gently falling",
    wait_for_completion=True,
    output_path="videos/fal_autumn_scene"
)

print(f"Video downloaded to: {result['file_path']}")
```

**Note**: fal.ai requires the `fal-client` package. Install it with `pip install fal-client`.

### TTS

```python
from wraipperz.api.tts import create_tts_manager

tts_manager = create_tts_manager()

# Generate speech using OpenAI Realtime TTS
response = tts_manager.generate_speech(
    "openai_realtime",
    text="This is a demonstration of my voice capabilities!",
    output_path="realtime_output.mp3",
    voice="ballad",
    context="Speak in a extremelly calm, soft, and relaxed voice.",
    return_alignment=True,
    speed=1.1,
)

# Convert speech using ElevenLabs
# TODO add example

```

## Environment Variables

Set up your API keys in environment variables to enable providers.

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
PIXVERSE_API_KEY=your_pixverse_key
KLING_API_KEY=your_kling_key
FAL_KEY=your_fal_key
# ...  todo add all
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
