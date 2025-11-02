from typing import List, Optional

import pytest
from pydantic import BaseModel, Field

from wraipperz.parsing.yaml_fix import yaml_extract_validate_repair


# Define simple test models
class Person(BaseModel):
    name: str
    age: int
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")


class Team(BaseModel):
    team_name: str
    members: List[Person]
    budget: float
    is_active: bool = True


class Config(BaseModel):
    database_host: str
    database_port: int
    username: str
    password: Optional[str] = None
    max_connections: int = Field(ge=1, le=100)


def test_heal_missing_required_fields():
    """Test healing YAML with missing required fields"""
    # YAML wrong email
    malformed_yaml = """
    Here's a person config:
    ```yaml
    name: John Smith
    age: 22
    email: cacaca#gmail.com
    ```
    """

    # The healing function should be able to add missing fields
    result = yaml_extract_validate_repair(
        model="gemini/gemini-2.5-flash",
        text=malformed_yaml,
        model_class=Person,
        max_retries=3,
    )

    # Verify we got a valid Person object
    assert isinstance(result, Person)
    assert result.name == "John Smith"
    assert isinstance(result.age, int)
    assert "@" in result.email  # Should have generated a valid email


def test_heal_wrong_data_types():
    """Test healing YAML with incorrect data types"""
    # YAML with wrong types: age as string, members not as list
    malformed_yaml = """
    Team configuration:
    ```yaml
    team_name: Engineering Team
    members: John Doe
    budget: "fifty thousand"
    is_active: yes
    ```
    """

    # The healing function should fix type issues
    result = yaml_extract_validate_repair(
        model="gemini/gemini-2.5-flash",
        text=malformed_yaml,
        model_class=Team,
        max_retries=3,
    )

    # Verify we got a valid Team object
    assert isinstance(result, Team)
    assert result.team_name == "Engineering Team"
    assert isinstance(result.members, list)
    assert isinstance(result.budget, float)
    assert isinstance(result.is_active, bool)


def test_heal_invalid_constraints():
    """Test healing YAML with values that violate field constraints"""
    # YAML with invalid values: port as string, max_connections out of range
    malformed_yaml = """
    Database configuration:
    ```yaml
    database_host: localhost
    database_port: "three thousand three hundred and six"
    username: admin
    max_connections: 500
    ```
    """

    # The healing function should fix constraint violations
    result = yaml_extract_validate_repair(
        model="gemini/gemini-2.5-flash",
        text=malformed_yaml,
        model_class=Config,
        max_retries=3,
    )

    # Verify we got a valid Config object
    assert isinstance(result, Config)
    assert result.database_host == "localhost"
    assert isinstance(result.database_port, int)
    assert result.username == "admin"
    assert 1 <= result.max_connections <= 100  # Should be within valid range


# Optional: Test that truly unfixable YAML raises an error after max retries
@pytest.mark.xfail(reason="This should fail after max retries")
def test_unfixable_yaml_raises_error():
    """Test that completely garbage input raises ValueError after retries"""
    garbage_yaml = """
    This is not YAML at all!
    {{{[[[///\\\\
    ```yaml
    !!!@#$%^&*()
    ```
    """

    with pytest.raises(ValueError, match="Failed to validate YAML"):
        yaml_extract_validate_repair(
            model="gemini/gemini-2.5-flash",
            text=garbage_yaml,
            model_class=Person,
            max_retries=2,  # Use fewer retries for faster failure
        )


def test_heal_unquoted_special_characters_in_list():
    """Test healing YAML with unquoted special characters in list items"""

    # Define the model for the anime scene
    class AnimeSceneMetadata(BaseModel):
        start_seconds: float
        end_seconds: float
        dialogues: List[dict]  # Simplified for this test
        unique_characters_japanese: List[str]
        key_events: str
        visuals: str
        predicted_user_queries: List[str]
        technicals: str
        mood: str
        locations: List[str]

    # The malformed YAML with unquoted special characters causing parsing errors
    malformed_yaml = """
    Anime scene metadata:
    ```yaml
    start_seconds: 0.0
    end_seconds: 5.0
    dialogues:
      - character_name_japanese: "L"
        character_name_kana: "エル"
        character_name_english: "L"
        line_japanese: "キラ、必ずお前を捕まえる。"
        line_english: "Kira, I will definitely catch you."
        start_time: 1.0
        end_time: 3.5
    unique_characters_japanese:
      - L
      - 夜神 月
    key_events: "In the preview for the next episode, the master detective L makes his first appearance (as a voice and a symbol) and publicly declares his intention to hunt down and capture Kira, establishing the central cat-and-mouse conflict of the series."
    visuals: "The scene opens with an extreme close-up of Light Yagami's glowing red eye, then cuts to a black screen displaying a large, white, gothic-style letter 'L' in the center."
    predicted_user_queries:
      - L's first appearance in Death Note
      - L vows to catch Kira
      - Death Note episode 1 ending preview
      - "Kira, I will definitely catch you" scene
      - The introduction of the detective L
    technicals: "An extreme close-up on an eye with a red glow effect is followed by a hard cut to a static, high-contrast graphic. L's voice is digitally altered and synthesized to conceal his identity, creating an anonymous and imposing presence."
    mood: "Ominous, tense, suspenseful, and confrontational."
    locations:
      - Interpol Conference (via broadcast)
    ```
    """

    # The healing function should fix the unquoted special characters
    result = yaml_extract_validate_repair(
        model="gemini/gemini-2.5-flash",
        text=malformed_yaml,
        model_class=AnimeSceneMetadata,
        max_retries=3,
    )

    # Verify we got a valid AnimeSceneMetadata object
    assert isinstance(result, AnimeSceneMetadata)
    assert result.start_seconds == 0.0
    assert result.end_seconds == 5.0

    # Check that the problematic list items were properly parsed
    assert isinstance(result.predicted_user_queries, list)
    assert len(result.predicted_user_queries) == 5

    # The items with special characters should be present
    # (they might be slightly modified by the AI to be valid)
    queries_text = " ".join(result.predicted_user_queries).lower()
    assert "first appearance" in queries_text or "l" in queries_text
    assert "catch" in queries_text or "kira" in queries_text

    # Check other fields are present
    assert len(result.dialogues) > 0
    assert len(result.unique_characters_japanese) == 2
    assert result.mood == "Ominous, tense, suspenseful, and confrontational."


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
