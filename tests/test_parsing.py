import pytest
import yaml
from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from datetime import date
import enum
from wraipperz.parsing.yaml_utils import pydantic_to_yaml_example


class Gender(enum.Enum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


class Address(BaseModel):
    street: str = Field(
        json_schema_extra={"example": "123 Main St", "comment": "Street address"}
    )
    city: str = Field(json_schema_extra={"example": "New York", "comment": "City name"})
    country: str = Field(json_schema_extra={"example": "USA"})


class Actor(BaseModel):
    name: str = Field(
        json_schema_extra={"example": "John Doe", "comment": "Full name of the actor"}
    )
    height: float = Field(
        json_schema_extra={"example": 1.85, "comment": "Height in meters"}
    )
    gender: Optional[Gender] = Field(
        default=None,
        json_schema_extra={"example": "male", "comment": "Gender identity"},
    )
    weight: Optional[float] = Field(default=None, json_schema_extra={"example": 75.5})
    birth_date: Optional[date] = Field(
        default=None,
        json_schema_extra={"example": "1990-01-15", "comment": "Date of birth"},
    )
    languages: List[str] = Field(
        default_factory=list,
        json_schema_extra={
            "example": ["English", "French"],
            "comment": "Languages spoken",
        },
    )
    address: Optional[Address] = Field(
        default=None,
        json_schema_extra={
            "example": {
                "street": "123 Broadway",
                "city": "Los Angeles",
                "country": "USA",
            },
            "comment": "Current address",
        },
    )
    filmography: Dict[int, str] = Field(
        default_factory=dict,
        json_schema_extra={
            "example": {2020: "Movie A", 2021: "Movie B"},
            "comment": "Movies by year",
        },
    )


class EmptyExample(BaseModel):
    name: str  # no example provided
    tags: List[str]  # no example provided


class ComplexNesting(BaseModel):
    nested_dict: Dict[str, Dict[str, List[str]]] = Field(
        json_schema_extra={
            "example": {
                "category1": {
                    "subcategory1": ["item1", "item2"],
                    "subcategory2": ["item3", "item4"],
                },
                "category2": {"subcategory3": ["item5", "item6"]},
            },
            "comment": "Complex nested structure",
        }
    )


def test_basic_model_yaml_generation():
    """Test YAML generation for a basic model with examples."""
    yaml_example = pydantic_to_yaml_example(Actor)
    assert yaml_example is not None
    assert isinstance(yaml_example, str)

    # Verify the YAML can be parsed
    parsed_data = yaml.safe_load(yaml_example)
    assert parsed_data is not None

    # Verify expected fields are present
    assert "name" in parsed_data
    assert parsed_data["name"] == "John Doe"
    assert "height" in parsed_data
    assert parsed_data["height"] == 1.85
    assert "languages" in parsed_data
    assert isinstance(parsed_data["languages"], list)


def test_model_without_examples():
    """Test YAML generation for a model without examples."""
    yaml_example = pydantic_to_yaml_example(EmptyExample)
    assert yaml_example is not None

    # Verify the YAML can be parsed
    parsed_data = yaml.safe_load(yaml_example)
    assert parsed_data is not None

    # Verify expected fields are present but with default values
    assert "name" in parsed_data
    assert "tags" in parsed_data
    assert isinstance(parsed_data["tags"], list)


def test_nested_model_yaml_generation():
    """Test YAML generation for a model with nested fields."""
    yaml_example = pydantic_to_yaml_example(Actor)
    parsed_data = yaml.safe_load(yaml_example)

    # Verify nested address field
    assert "address" in parsed_data
    assert isinstance(parsed_data["address"], dict)
    assert "street" in parsed_data["address"]
    assert "city" in parsed_data["address"]
    assert "country" in parsed_data["address"]


def test_complex_nesting_yaml_generation():
    """Test YAML generation for a model with complex nested structures."""
    yaml_example = pydantic_to_yaml_example(ComplexNesting)
    print(f"\nGenerated YAML:\n{yaml_example}")

    parsed_data = yaml.safe_load(yaml_example)
    print(f"\nParsed data:\n{parsed_data}")

    # Verify complex nested structure
    assert "nested_dict" in parsed_data
    assert isinstance(parsed_data["nested_dict"], dict)

    # Debug what's actually in the nested_dict
    print(f"\nNested dict content: {parsed_data['nested_dict']}")

    # Check if category1 exists before asserting its contents
    if "category1" in parsed_data["nested_dict"]:
        category1 = parsed_data["nested_dict"]["category1"]
        print(f"Category1 content: {category1}")

        if isinstance(category1, dict) and "subcategory1" in category1:
            subcategory1 = category1["subcategory1"]
            print(f"Subcategory1 content: {subcategory1}")
            assert isinstance(subcategory1, list)
            assert "item1" in subcategory1
        else:
            pytest.fail(
                f"subcategory1 not found in category1 or category1 is not a dict: {category1}"
            )
    else:
        pytest.fail(f"category1 not found in nested_dict: {parsed_data['nested_dict']}")


def test_enum_handling():
    """Test YAML generation for a model with enum fields."""
    yaml_example = pydantic_to_yaml_example(Actor)
    parsed_data = yaml.safe_load(yaml_example)

    # Verify enum field is handled correctly
    assert "gender" in parsed_data
    assert parsed_data["gender"] == "male"


def test_date_handling():
    """Test YAML generation for a model with date fields."""
    yaml_example = pydantic_to_yaml_example(Actor)
    parsed_data = yaml.safe_load(yaml_example)

    # Verify date field is handled correctly
    assert "birth_date" in parsed_data
    assert parsed_data["birth_date"] == "1990-01-15"


def test_comments_in_yaml():
    """Test that comments from json_schema_extra are included in the YAML."""
    yaml_example = pydantic_to_yaml_example(Actor)

    # Check for comments in the generated YAML
    assert "# Full name of the actor" in yaml_example
    assert "# Height in meters" in yaml_example
    assert "# Gender identity" in yaml_example
    assert "# Date of birth" in yaml_example
    assert "# Languages spoken" in yaml_example
    assert "# Current address" in yaml_example
    assert "# Movies by year" in yaml_example


def test_list_nested_object():
    """Test YAML generation for a model with a list of nested objects."""

    class Line(BaseModel):
        name: str = Field(
            json_schema_extra={"example": "Bob", "comment": "The name of the character"}
        )
        text: str = Field(
            json_schema_extra={
                "example": "Hello, how are you?",
                "comment": "The text of the line",
            }
        )
        # extra_info: str = Field(json_schema_extra={"example": "Hello, how are you?", "comment": "The text of the line"})

    class Lines(BaseModel):
        lines: List[Line] = Field(
            json_schema_extra={
                "example": [Line(name="Bob", text="Hello, how are you?")],
                "comment": "Lines of each entry in the script in order.",
            }
        )

    yaml_example = pydantic_to_yaml_example(Lines)
    parsed_data = yaml.safe_load(yaml_example)

    # Verify the structure of the generated YAML
    assert "lines" in parsed_data
    assert isinstance(parsed_data["lines"], list)
    assert len(parsed_data["lines"]) > 0

    # Verify the nested object properties
    first_line = parsed_data["lines"][0]
    assert "name" in first_line
    assert "text" in first_line
    assert first_line["name"] == "Bob"
    assert first_line["text"] == "Hello, how are you?"

    # Check for comments in the generated YAML
    assert "# Lines of each entry in the script in order." in yaml_example
    # assert "# The name of the character" in yaml_example
    # assert "# The text of the line" in yaml_example
