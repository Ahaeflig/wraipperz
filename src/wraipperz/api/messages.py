from __future__ import annotations
from typing import List, Dict, Union, Optional
from pathlib import Path


class Message:
    def __init__(
        self, role: str = "user", content: Union[str, List[Dict]] = ""
    ) -> None:
        """Initialize a message with a role and content.

        Args:
            role: The role of the message sender ("system", "user", or "assistant")
            content: Either a string for simple text messages or a list of content items
        """
        self.role: str = role

        if isinstance(content, list):
            self.content = content
        elif content:  # Only create a text block if content is non-empty
            self.content = [{"type": "text", "text": content}]
        else:
            # Initialize with empty list if no content provided
            self.content = []

    def add_text(self, text: str) -> Message:
        """Add a text content item to the message.

        Args:
            text: The text content to add

        Returns:
            self for method chaining
        """
        self.content.append({"type": "text", "text": text})
        return self

    def add_image(self, image_path: Union[str, Path]) -> Message:
        """Add an image content item to the message.

        Args:
            image_path: Path to the image file or URL

        Returns:
            self for method chaining
        """
        self.content.append(
            {"type": "image_url", "image_url": {"url": str(image_path)}}
        )
        return self

    def to_dict(self) -> Dict:
        """Convert the message to a dictionary format expected by AI providers.

        Returns:
            Dict containing the role and content in the expected format
        """
        if len(self.content) == 1 and self.content[0]["type"] == "text":
            # If only one text message, simplify to string content
            return {"role": self.role, "content": self.content[0]["text"]}
        return {"role": self.role, "content": self.content}


class MessageBuilder:
    def __init__(self) -> None:
        """Initialize an empty message builder."""
        self.messages: List[Message] = []

    def add_system(self, content: str) -> MessageBuilder:
        """Add a system message.

        Args:
            content: The system message content

        Returns:
            self for method chaining
        """
        self.messages.append(Message("system", content))
        return self

    def add_user(self, content: str = "") -> MessageBuilder:
        """Add a user message.

        Args:
            content: Optional user message content

        Returns:
            self for method chaining
        """
        self.messages.append(Message("user", content))
        return self

    def add_assistant(self, content: str = "") -> MessageBuilder:
        """Add an assistant message.

        Args:
            content: Optional assistant message content

        Returns:
            self for method chaining
        """
        self.messages.append(Message("assistant", content))
        return self

    def add_image(
        self, image_path: Union[str, Path], text: Optional[str] = None
    ) -> MessageBuilder:
        """Add an image with optional text to the current or new user message.

        Args:
            image_path: Path to the image file or URL
            text: Optional text to accompany the image

        Returns:
            self for method chaining
        """
        if not self.messages or self.messages[-1].role != "user":
            self.add_user()
        self.messages[-1].add_image(image_path)
        if text:
            self.messages[-1].add_text(text)
        return self

    def build(self) -> List[Dict]:
        """Convert all messages to the format expected by AI providers.

        Returns:
            List of message dictionaries in the format expected by AI providers
        """
        return [msg.to_dict() for msg in self.messages]


# Example usage:
"""
from agency.messages import MessageBuilder

# Create a conversation
messages = (
    MessageBuilder()
    .add_system("You are a helpful AI assistant.")
    .add_user("Hello! Can you help me analyze some images?")
    .add_assistant("Of course! I'd be happy to help analyze any images you share.")
    .add_image(
        "path/to/image.jpg",
        "What can you tell me about this picture?"
    )
    .build()
)

# Use with AI provider
response, cost = await call_ai_async(
    messages,
    temperature=0.7,
    max_tokens=150,
    model="gemini-1.5-flash"
)
"""
