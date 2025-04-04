# wraipperz

Easy wrapper for various AI APIs including LLMs, ASR, and TTS.

## Installation

```bash
pip install wraipperz
uv add wraipperz
```

## Features

- **LLM API Wrappers**: Unified interface for OpenAI, Anthropic, Google, and other LLM providers
- **ASR (Automatic Speech Recognition)**: Convert speech to text
- **TTS (Text-to-Speech)**: Convert text to speech
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
# ...  todo add all
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
