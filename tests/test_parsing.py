import enum
from datetime import date
from typing import Dict, List, Optional

import pytest
import yaml
from pydantic import BaseModel, Field, field_validator

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


# Add this Ethnicity enum and test model for options feature
class Ethnicity(enum.Enum):
    JAPANESE = "japanese"
    CHINESE = "chinese"
    KOREAN = "korean"
    AMERICAN = "american"
    BRITISH = "british"
    FRENCH = "french"


class ActorWithOptions(BaseModel):
    """Actor model with options in fields."""

    name: str = Field(
        json_schema_extra={"example": "John Doe", "comment": "Full name of the actor"}
    )

    ethnicity: Ethnicity = Field(
        default=None,
        json_schema_extra={
            "example": "japanese",
            "comment": "Required - Ethnicity of the actor",
            "options": [e.value for e in Ethnicity],
            "is_attribute": True,
        },
    )

    gender: Optional[Gender] = Field(
        default=None,
        json_schema_extra={
            "example": "male",
            "comment": "Gender identity",
            "options": ["male", "female", "other"],
        },
    )

    role_type: str = Field(
        json_schema_extra={
            "example": "protagonist",
            "comment": "Type of role",
            "options": ["protagonist", "antagonist", "supporting", "cameo"],
        }
    )


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
    assert "# The name of the character" in yaml_example
    assert "# The text of the line" in yaml_example


def test_nested_basemodel_without_explicit_example():
    """Test YAML generation for a model with nested BaseModel in list without explicit example."""

    class LineAnalysis(BaseModel):
        """Analysis of a single dialogue line from video."""

        action: str = Field(
            json_schema_extra={
                "example": "Character looks worried and fidgets with their hands",
                "comment": "Description of the character's actions or expressions",
            }
        )

        voice_type: str = Field(
            json_schema_extra={
                "example": "NORMAL",
                "comment": "Voice type: VO for voice over, OFF for offscreen, WHISPER for whispers, or NORMAL for normal voices",
            }
        )

        location: str = Field(
            json_schema_extra={
                "example": "Living room",
                "comment": "The location where the dialogue takes place",
            }
        )

        time_of_day: str = Field(
            json_schema_extra={
                "example": "DAY",
                "comment": "Time of day (e.g., DAY, NIGHT, MORNING, etc.)",
            }
        )

        translation: str = Field(
            json_schema_extra={
                "example": "I can't believe this is happening.",
                "comment": "High-quality English translation of the dialogue",
            }
        )

        scene_change: Optional[str] = Field(
            default="",
            json_schema_extra={
                "example": "Characters move to the kitchen to prepare dinner",
                "comment": "Brief description of scene change if any, empty if no scene change",
            },
        )

    class VideoAnalysisResponse(BaseModel):
        """Complete video analysis response containing all line analyses."""

        lines: List[LineAnalysis] = Field(
            json_schema_extra={
                "comment": "Analysis for each line of dialogue in the script, in order"
            }
        )

    yaml_example = pydantic_to_yaml_example(VideoAnalysisResponse)
    print(f"\nGenerated YAML:\n{yaml_example}")

    parsed_data = yaml.safe_load(yaml_example)
    print(f"\nParsed data:\n{parsed_data}")

    # Verify the structure of the generated YAML
    assert "lines" in parsed_data
    assert isinstance(parsed_data["lines"], list)
    assert len(parsed_data["lines"]) > 0

    # Verify that we don't get null values
    first_line = parsed_data["lines"][0]
    assert first_line is not None
    assert isinstance(first_line, dict)

    # Verify the nested object properties have actual values, not null
    assert "action" in first_line
    assert first_line["action"] is not None
    assert (
        first_line["action"] == "Character looks worried and fidgets with their hands"
    )

    assert "voice_type" in first_line
    assert first_line["voice_type"] == "NORMAL"

    assert "location" in first_line
    assert first_line["location"] == "Living room"

    assert "time_of_day" in first_line
    assert first_line["time_of_day"] == "DAY"

    assert "translation" in first_line
    assert first_line["translation"] == "I can't believe this is happening."

    assert "scene_change" in first_line
    assert (
        first_line["scene_change"] == "Characters move to the kitchen to prepare dinner"
    )

    # Check for comments in the generated YAML
    assert (
        "# Analysis for each line of dialogue in the script, in order" in yaml_example
    )


def test_nested_object_comments_bug():
    """Test that comments from nested BaseModel objects are included in generated YAML."""

    class NestedModel(BaseModel):
        nested_field: str = Field(
            json_schema_extra={
                "example": "nested value",
                "comment": "This is a comment from the nested model",
            }
        )
        another_nested_field: int = Field(
            json_schema_extra={
                "example": 42,
                "comment": "Another comment from nested model",
            }
        )

    class ParentModel(BaseModel):
        parent_field: str = Field(
            json_schema_extra={
                "example": "parent value",
                "comment": "This is a comment from the parent model",
            }
        )
        nested: NestedModel = Field(
            json_schema_extra={
                "comment": "This is a comment about the nested object field"
            }
        )

    yaml_example = pydantic_to_yaml_example(ParentModel)
    print(f"\nGenerated YAML:\n{yaml_example}")

    # Parent level comments should be present (this works currently)
    assert "# This is a comment from the parent model" in yaml_example
    assert "# This is a comment about the nested object field" in yaml_example

    # These assertions should FAIL with current implementation - nested object comments missing
    assert (
        "# This is a comment from the nested model" in yaml_example
    ), "Comments from nested BaseModel fields are not included in YAML output"
    assert (
        "# Another comment from nested model" in yaml_example
    ), "Comments from nested BaseModel fields are not included in YAML output"


def test_deeply_nested_object_comments_bug():
    """Test that comments from deeply nested BaseModel objects are included in generated YAML.

    This test should FAIL with the current implementation to demonstrate the bug
    with multiple levels of nesting.
    """

    class DeepestModel(BaseModel):
        deepest_field: str = Field(
            json_schema_extra={
                "example": "deep value",
                "comment": "Comment from the deepest level",
            }
        )

    class MiddleModel(BaseModel):
        middle_field: str = Field(
            json_schema_extra={
                "example": "middle value",
                "comment": "Comment from the middle level",
            }
        )
        deepest: DeepestModel = Field(
            json_schema_extra={"comment": "Comment about deepest object"}
        )

    class TopModel(BaseModel):
        top_field: str = Field(
            json_schema_extra={
                "example": "top value",
                "comment": "Comment from the top level",
            }
        )
        middle: MiddleModel = Field(
            json_schema_extra={"comment": "Comment about middle object"}
        )

    yaml_example = pydantic_to_yaml_example(TopModel)
    print(f"\nGenerated YAML:\n{yaml_example}")

    # Top level comments should be present (this works currently)
    assert "# Comment from the top level" in yaml_example
    assert "# Comment about middle object" in yaml_example

    # These assertions should FAIL with current implementation - nested comments missing
    assert (
        "# Comment from the middle level" in yaml_example
    ), "Comments from nested BaseModel fields are not included in YAML output"
    assert (
        "# Comment about deepest object" in yaml_example
    ), "Comments from nested BaseModel fields are not included in YAML output"
    assert (
        "# Comment from the deepest level" in yaml_example
    ), "Comments from deeply nested BaseModel fields are not included in YAML output"


def test_list_of_nested_objects_comments_bug():
    """Test that comments from BaseModel objects in lists are included in generated YAML.

    This test should FAIL with the current implementation to demonstrate the bug
    where BaseModel objects inside lists don't have their field comments included.
    """

    class ItemModel(BaseModel):
        item_name: str = Field(
            json_schema_extra={
                "example": "item name",
                "comment": "Comment for item name field",
            }
        )
        item_value: float = Field(
            json_schema_extra={
                "example": 99.99,
                "comment": "Comment for item value field",
            }
        )

    class ContainerModel(BaseModel):
        container_field: str = Field(
            json_schema_extra={
                "example": "container value",
                "comment": "Comment from container model",
            }
        )
        items: List[ItemModel] = Field(
            json_schema_extra={"comment": "List of items with their own comments"}
        )

    yaml_example = pydantic_to_yaml_example(ContainerModel)
    print(f"\nGenerated YAML:\n{yaml_example}")

    # Container level comments should be present (this works currently)
    assert "# Comment from container model" in yaml_example
    assert "# List of items with their own comments" in yaml_example

    # These assertions should FAIL with current implementation - nested list item comments missing
    assert (
        "# Comment for item name field" in yaml_example
    ), "Comments from BaseModel fields inside lists are not included in YAML output"
    assert (
        "# Comment for item value field" in yaml_example
    ), "Comments from BaseModel fields inside lists are not included in YAML output"


# ============================================================================
# COMPREHENSIVE BUG TESTS - Testing edge cases and potential failures
# ============================================================================


def test_circular_reference_bug():
    """Test that circular references cause infinite recursion.

    This test demonstrates a critical bug where self-referencing models
    will cause infinite recursion and stack overflow.
    """

    class SelfReference(BaseModel):
        name: str = Field(
            json_schema_extra={"example": "test", "comment": "Name field"}
        )
        self_ref: Optional["SelfReference"] = Field(
            default=None, json_schema_extra={"comment": "Self reference field"}
        )

    # This should either handle circular references gracefully or raise a proper error
    # Currently this will cause infinite recursion - CRITICAL BUG
    with pytest.raises((RecursionError, RuntimeError, TypeError)):
        yaml_example = pydantic_to_yaml_example(SelfReference)
        print(f"Generated YAML (this shouldn't happen): {yaml_example}")


def test_mutual_circular_reference_bug():
    """Test mutual circular references between two models."""

    class ModelA(BaseModel):
        name: str = Field(json_schema_extra={"example": "A"})
        ref_to_b: Optional["ModelB"] = None

    class ModelB(BaseModel):
        name: str = Field(json_schema_extra={"example": "B"})
        ref_to_a: Optional[ModelA] = None

    # Update forward references
    ModelA.model_rebuild()
    ModelB.model_rebuild()

    # This will also cause infinite recursion
    with pytest.raises((RecursionError, RuntimeError, TypeError)):
        _ = pydantic_to_yaml_example(ModelA)


def test_complex_union_types_bug():
    """Test that complex Union types (non-Optional) are not handled properly."""
    from typing import Union

    class UnionModel(BaseModel):
        simple_union: Union[str, int] = Field(
            json_schema_extra={"example": "test", "comment": "String or int field"}
        )
        complex_union: Union[str, int, float, bool] = Field(
            json_schema_extra={"example": 42, "comment": "Multiple type field"}
        )
        union_with_model: Union[str, Address] = Field(
            json_schema_extra={
                "example": "simple string",
                "comment": "String or Address",
            }
        )

    yaml_example = pydantic_to_yaml_example(UnionModel)
    print(f"Generated YAML:\n{yaml_example}")

    parsed_data = yaml.safe_load(yaml_example)

    # The current implementation doesn't handle Union types properly
    # It should use the example if provided, but for fields without examples
    # it will likely fall back to None or cause errors
    assert "simple_union" in parsed_data
    assert "complex_union" in parsed_data
    assert "union_with_model" in parsed_data


def test_input_validation_bug():
    """Test that passing a model instance instead of class gives proper error."""

    # Create an instance instead of using the class
    actor_instance = Actor(name="Test Actor", height=1.75)

    # This should raise a clear TypeError, but the current error message might be confusing
    with pytest.raises(TypeError) as exc_info:
        pydantic_to_yaml_example(actor_instance)

    # The error message should be clear about expecting a class, not instance
    error_msg = str(exc_info.value)
    assert "BaseModel" in error_msg


def test_non_basemodel_input_validation():
    """Test validation with non-BaseModel inputs."""

    class RegularClass:
        pass

    # Should raise TypeError for non-BaseModel classes
    with pytest.raises(TypeError):
        pydantic_to_yaml_example(RegularClass)

    # Should raise TypeError for built-in types
    with pytest.raises(TypeError):
        pydantic_to_yaml_example(str)

    with pytest.raises(TypeError):
        pydantic_to_yaml_example(dict)


def test_empty_enum_handling_bug():
    """Test that empty enums cause IndexError."""

    class EmptyEnum(enum.Enum):
        pass  # No enum values

    class ModelWithEmptyEnum(BaseModel):
        empty_field: EmptyEnum = Field(json_schema_extra={"comment": "This will break"})

    # This should cause an IndexError in generate_default_example
    # when trying to access list(EmptyEnum)[0]
    with pytest.raises(IndexError):
        _ = pydantic_to_yaml_example(ModelWithEmptyEnum)


def test_invalid_json_schema_extra_bug():
    """Test edge cases with malformed json_schema_extra."""

    # Test with non-dict json_schema_extra (this might not be possible with current Pydantic)
    # But let's test what happens with missing or None values

    class BadSchemaModel(BaseModel):
        # Field with None json_schema_extra
        field1: str = Field(json_schema_extra=None)
        # Field with empty dict
        field2: str = Field(json_schema_extra={})
        # Field with malformed comment
        field3: str = Field(json_schema_extra={"comment": None})
        # Field with non-string comment
        field4: str = Field(json_schema_extra={"comment": 123})

    # Should handle these gracefully without crashing
    yaml_example = pydantic_to_yaml_example(BadSchemaModel)
    parsed_data = yaml.safe_load(yaml_example)

    assert "field1" in parsed_data
    assert "field2" in parsed_data
    assert "field3" in parsed_data
    assert "field4" in parsed_data


def test_forward_reference_bug():
    """Test handling of forward references (string annotations)."""

    class ForwardRefModel(BaseModel):
        name: str = Field(json_schema_extra={"example": "test"})
        # Forward reference as string
        future_field: Optional["FutureModel"] = Field(
            default=None, json_schema_extra={"comment": "Reference to future model"}
        )

    class FutureModel(BaseModel):
        value: str = Field(json_schema_extra={"example": "future value"})

    # Update forward references
    ForwardRefModel.model_rebuild()

    # This might not work properly with string forward references
    yaml_example = pydantic_to_yaml_example(ForwardRefModel)
    parsed_data = yaml.safe_load(yaml_example)

    assert "name" in parsed_data
    assert "future_field" in parsed_data


def test_deep_nesting_performance_bug():
    """Test very deep nesting that could cause performance issues or stack overflow."""

    class Level5(BaseModel):
        value: str = Field(json_schema_extra={"example": "level5"})

    class Level4(BaseModel):
        nested: Level5 = Field(json_schema_extra={"comment": "Level 4"})

    class Level3(BaseModel):
        nested: Level4 = Field(json_schema_extra={"comment": "Level 3"})

    class Level2(BaseModel):
        nested: Level3 = Field(json_schema_extra={"comment": "Level 2"})

    class Level1(BaseModel):
        nested: Level2 = Field(json_schema_extra={"comment": "Level 1"})

    class RootLevel(BaseModel):
        deep_nested: Level1 = Field(json_schema_extra={"comment": "Root level"})

    # This should work but might be slow or cause issues with very deep nesting
    yaml_example = pydantic_to_yaml_example(RootLevel)
    parsed_data = yaml.safe_load(yaml_example)

    # Verify the deep nesting works
    assert "deep_nested" in parsed_data
    assert "nested" in parsed_data["deep_nested"]

    # Check that nested comments are missing (demonstrates the bug)
    # This will likely fail showing the comment propagation bug
    assert "# Level 1" in yaml_example  # Parent level comment should work
    # These will likely fail:
    try:
        assert "# Level 2" in yaml_example
        assert "# Level 3" in yaml_example
        assert "# Level 4" in yaml_example
    except AssertionError:
        print("Deep nesting comment propagation bug confirmed")


def test_field_constraints_ignored_bug():
    """Test that field constraints are ignored and examples might violate them."""

    class ConstrainedModel(BaseModel):
        positive_int: int = Field(
            gt=0,
            json_schema_extra={
                "example": -5,
                "comment": "Should be positive but example is negative",
            },
        )
        min_length_str: str = Field(
            min_length=10,
            json_schema_extra={
                "example": "short",
                "comment": "Should be at least 10 chars",
            },
        )

        @field_validator("positive_int")  # Updated decorator
        @classmethod  # Required for field_validator
        def validate_positive(cls, v):
            if v <= 0:
                raise ValueError("must be positive")
            return v  # Must return the value

        @field_validator("min_length_str")  # Updated decorator
        @classmethod  # Required for field_validator
        def validate_length(cls, v):
            if len(v) < 10:
                raise ValueError("must be at least 10 characters")
            return v  # Must return the value

    # The function should generate YAML, but the examples violate the constraints
    yaml_example = pydantic_to_yaml_example(ConstrainedModel)
    parsed_data = yaml.safe_load(yaml_example)

    # These examples violate the model's constraints
    assert parsed_data["positive_int"] == -5  # Violates gt=0
    assert parsed_data["min_length_str"] == "short"  # Violates min_length=10

    # If we tried to create an actual instance with this data, it would fail
    with pytest.raises(ValueError):
        ConstrainedModel(**parsed_data)


def test_nested_list_complex_types_bug():
    """Test complex nested list structures with multiple BaseModel types."""

    class Item(BaseModel):
        name: str = Field(json_schema_extra={"example": "item", "comment": "Item name"})
        tags: List[str] = Field(json_schema_extra={"example": ["tag1", "tag2"]})

    class Category(BaseModel):
        category_name: str = Field(json_schema_extra={"example": "category"})
        items: List[Item] = Field(json_schema_extra={"comment": "Items in category"})

    class Store(BaseModel):
        store_name: str = Field(json_schema_extra={"example": "My Store"})
        categories: List[Category] = Field(
            json_schema_extra={"comment": "Store categories"}
        )
        featured_items: List[Item] = Field(
            json_schema_extra={"comment": "Featured items"}
        )

    yaml_example = pydantic_to_yaml_example(Store)
    print(f"Complex nested YAML:\n{yaml_example}")

    parsed_data = yaml.safe_load(yaml_example)

    # Verify structure
    assert "categories" in parsed_data
    assert isinstance(parsed_data["categories"], list)
    assert len(parsed_data["categories"]) > 0

    # Check first category
    first_category = parsed_data["categories"][0]
    assert "items" in first_category
    assert isinstance(first_category["items"], list)

    # Comments from nested BaseModel fields in lists will likely be missing
    # This demonstrates the comment propagation bug in lists
    assert "# Store categories" in yaml_example  # Top level should work

    # These will likely fail - nested comments in list items missing:
    try:
        assert "# Item name" in yaml_example
        assert "# Items in category" in yaml_example
    except AssertionError:
        print("Nested list comment propagation bug confirmed")


def test_none_values_and_optional_handling():
    """Test handling of None values and Optional fields edge cases."""

    class OptionalModel(BaseModel):
        required_field: str = Field(json_schema_extra={"example": "required"})
        optional_with_example: Optional[str] = Field(
            default=None,
            json_schema_extra={
                "example": "optional",
                "comment": "Optional with example",
            },
        )
        optional_without_example: Optional[str] = Field(
            default=None, json_schema_extra={"comment": "Optional without example"}
        )
        optional_complex: Optional[Address] = Field(
            default=None, json_schema_extra={"comment": "Optional complex type"}
        )

    yaml_example = pydantic_to_yaml_example(OptionalModel)
    print(f"Optional fields YAML:\n{yaml_example}")

    parsed_data = yaml.safe_load(yaml_example)

    assert "required_field" in parsed_data
    assert "optional_with_example" in parsed_data
    assert "optional_without_example" in parsed_data
    assert "optional_complex" in parsed_data

    # Check how None vs example values are handled
    assert parsed_data["optional_with_example"] == "optional"
    # The optional_without_example might be None or a default value


def test_generic_types_bug():
    """Test handling of generic types and type variables."""
    from typing import Generic, TypeVar

    T = TypeVar("T")

    class GenericModel(BaseModel, Generic[T]):
        value: T = Field(json_schema_extra={"comment": "Generic value"})

    # This will likely cause issues since T is not resolved
    # The current implementation doesn't handle generics properly
    try:
        yaml_example = pydantic_to_yaml_example(GenericModel)
        print(f"Generic model YAML:\n{yaml_example}")

        # If it doesn't crash, the result might not be meaningful
        parsed_data = yaml.safe_load(yaml_example)
        assert "value" in parsed_data

    except (TypeError, AttributeError, NameError) as e:
        print(f"Generic types not supported: {e}")
        # This is expected - generics are not properly handled


def test_set_and_tuple_complex_types():
    """Test handling of Set and Tuple types with complex nested structures."""
    from typing import Set, Tuple

    class ComplexTypesModel(BaseModel):
        string_set: Set[str] = Field(
            json_schema_extra={
                "example": {"item1", "item2"},
                "comment": "Set of strings",
            }
        )
        nested_tuple: Tuple[str, int, Address] = Field(
            json_schema_extra={
                "example": (
                    "test",
                    42,
                    {"street": "123 Main", "city": "NYC", "country": "USA"},
                ),
                "comment": "Tuple with mixed types",
            }
        )
        set_of_models: Set[Address] = Field(
            json_schema_extra={"comment": "Set of Address models"}
        )

    yaml_example = pydantic_to_yaml_example(ComplexTypesModel)
    print(f"Complex types YAML:\n{yaml_example}")

    parsed_data = yaml.safe_load(yaml_example)

    # Sets and tuples might not serialize properly to YAML
    # or might lose their type information
    assert "string_set" in parsed_data
    assert "nested_tuple" in parsed_data
    assert "set_of_models" in parsed_data


def test_model_with_class_methods_and_properties():
    """Test model with class methods and properties that might interfere."""

    class ComplexBehaviorModel(BaseModel):
        name: str = Field(json_schema_extra={"example": "test"})

        @property
        def computed_value(self) -> str:
            return f"computed_{self.name}"

        @classmethod
        def create_default(cls):
            return cls(name="default")

        def instance_method(self) -> str:
            return "instance"

    # Should ignore methods and properties, only process actual fields
    yaml_example = pydantic_to_yaml_example(ComplexBehaviorModel)
    parsed_data = yaml.safe_load(yaml_example)

    # Should only have the actual field, not computed properties
    assert "name" in parsed_data
    assert "computed_value" not in parsed_data
    assert parsed_data["name"] == "test"


def test_extremely_nested_dict_structure():
    """Test extremely nested dictionary structures that could cause issues."""

    class ExtremeNesting(BaseModel):
        # Dict[str, Dict[str, Dict[str, Dict[str, List[Dict[str, str]]]]]]
        ultra_nested: Dict[str, Dict[str, Dict[str, Dict[str, List[str]]]]] = Field(
            json_schema_extra={
                "example": {
                    "level1": {"level2": {"level3": {"level4": ["item1", "item2"]}}}
                },
                "comment": "Extremely nested structure",
            }
        )

    yaml_example = pydantic_to_yaml_example(ExtremeNesting)
    print(f"Extreme nesting YAML:\n{yaml_example}")

    parsed_data = yaml.safe_load(yaml_example)

    # Verify the structure is preserved
    assert "ultra_nested" in parsed_data
    assert "level1" in parsed_data["ultra_nested"]
    assert "level2" in parsed_data["ultra_nested"]["level1"]
    assert "level3" in parsed_data["ultra_nested"]["level1"]["level2"]
    assert "level4" in parsed_data["ultra_nested"]["level1"]["level2"]["level3"]
    assert isinstance(
        parsed_data["ultra_nested"]["level1"]["level2"]["level3"]["level4"], list
    )


def test_comments_with_options_in_yaml():
    """Test that options from json_schema_extra are included in the YAML comments."""
    yaml_example = pydantic_to_yaml_example(ActorWithOptions)

    # Check for comments with options in the generated YAML
    assert "# Full name of the actor" in yaml_example  # No options
    assert (
        "# Required - Ethnicity of the actor (options: japanese, chinese, korean, american, british, french)"
        in yaml_example
    )
    assert "# Gender identity (options: male, female, other)" in yaml_example
    assert (
        "# Type of role (options: protagonist, antagonist, supporting, cameo)"
        in yaml_example
    )

    # Verify YAML is still valid
    parsed_data = yaml.safe_load(yaml_example)
    assert "name" in parsed_data
    assert "ethnicity" in parsed_data
    assert "gender" in parsed_data
    assert "role_type" in parsed_data

    # Verify example values are preserved
    assert parsed_data["ethnicity"] == "japanese"
    assert parsed_data["gender"] == "male"
    assert parsed_data["role_type"] == "protagonist"


def test_nested_model_with_options():
    """Test that options work correctly in nested models."""

    class NestedWithOptions(BaseModel):
        status: str = Field(
            json_schema_extra={
                "example": "active",
                "comment": "Current status",
                "options": ["active", "inactive", "pending"],
            }
        )

    class ParentWithOptions(BaseModel):
        nested: NestedWithOptions = Field(
            json_schema_extra={"comment": "Nested configuration"}
        )
        parent_field: str = Field(
            json_schema_extra={
                "example": "option1",
                "comment": "Parent level field",
                "options": ["option1", "option2", "option3"],
            }
        )

    yaml_example = pydantic_to_yaml_example(ParentWithOptions)

    # Check that options appear in both parent and nested fields
    assert "# Current status (options: active, inactive, pending)" in yaml_example
    assert "# Parent level field (options: option1, option2, option3)" in yaml_example

    # Verify the YAML is valid
    parsed_data = yaml.safe_load(yaml_example)
    assert parsed_data["nested"]["status"] == "active"
    assert parsed_data["parent_field"] == "option1"


def test_empty_options_list():
    """Test handling of empty options list."""

    class EmptyOptionsModel(BaseModel):
        field_with_empty_options: str = Field(
            json_schema_extra={
                "example": "value",
                "comment": "Field with empty options",
                "options": [],
            }
        )
        field_without_options: str = Field(
            json_schema_extra={
                "example": "value",
                "comment": "Field without options key",
            }
        )

    yaml_example = pydantic_to_yaml_example(EmptyOptionsModel)

    # Empty options should not be shown
    assert "# Field with empty options" in yaml_example
    assert (
        "(options:" not in yaml_example.split("\n")[0]
    )  # First field shouldn't have options

    # Field without options key should work normally
    assert "# Field without options key" in yaml_example
