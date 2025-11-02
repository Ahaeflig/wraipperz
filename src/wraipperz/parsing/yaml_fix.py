from typing import Type, TypeVar

import yaml
from pydantic import BaseModel, ValidationError

# Import from the llm module - adjust path as needed
from wraipperz.api.llm import call_ai
from wraipperz.api.messages import MessageBuilder

from .yaml_utils import find_yaml, pydantic_to_yaml_example

T = TypeVar("T", bound=BaseModel)


def yaml_extract_validate_repair(
    model: str,
    text: str,
    model_class: Type[T],
    max_retries: int = 3,
) -> T:
    """
    Extract YAML from text, validate it against a Pydantic model, and heal if needed.

    This function will:
    1. Extract YAML content from the input text using find_yaml()
    2. Parse the YAML using yaml.safe_load()
    3. Validate against the provided Pydantic model class
    4. If validation fails, use AI to heal the YAML (up to max_retries times)

    Args:
        text: Input text containing YAML (possibly in ```yaml blocks)
        model_class: Pydantic model class to validate against
        max_retries: Maximum number of AI healing attempts (default: 3)
        ai_model: AI model to use for healing (default: Claude 3.5 Sonnet)

    Returns:
        Validated instance of the Pydantic model

    Raises:
        ValueError: If YAML cannot be extracted, parsed, or validated after all retries
    """
    # Step 1: Extract YAML content
    yaml_content = find_yaml(text)

    # Keep track of the current YAML content for healing
    current_yaml = yaml_content
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            if not current_yaml:
                raise ValueError(
                    "No YAML content found in the provided text, probably wrong YAML block usage/format"
                )

            # Step 2: Parse YAML
            yaml_data = yaml.safe_load(current_yaml)

            # Step 3: Validate with Pydantic
            validated_model = model_class.model_validate(yaml_data)
            return validated_model

        except yaml.YAMLError as e:
            last_error = e
            error_type = "YAML parsing error"
            error_message = str(e)

        except ValidationError as e:
            last_error = e
            error_type = "Pydantic validation error"
            error_message = e.json(indent=2)

        except ValueError as e:
            last_error = e
            error_type = "ValueError"
            error_message = str(e)

        except Exception as e:
            last_error = e
            error_type = "Unexpected error"
            error_message = str(e)

        # If this was the last attempt, raise the error
        if attempt == max_retries:
            raise ValueError(
                f"Failed to validate YAML after {max_retries} healing attempts. "
                f"Last error: {error_type}: {error_message}"
            ) from last_error

        # Use AI to heal the YAML
        print(f"Attempt {attempt + 1}/{max_retries}: Using AI to heal YAML...")

        YAML_FIXING_GUIDE = """
1. **Quote these ALWAYS:**
   - Strings containing `: ` (colon-space)
   - Strings containing quotes (`"` or `'`)
   - Strings starting with: `{}[]>|*&!%#@,?:-`
   - Boolean-like values when meant as strings: `yes`, `no`, `true`, `false`, `True`, `False`

2. **List items need special attention:**
   - `- "text with: colon"` ✓
   - `- 'text with "quotes"'` ✓
   - `- text with: colon` ✗ WILL FAIL

3. **Quoting methods (use appropriately):**
   - Single quotes: `'literal text, "quotes" are fine'` (no escaping)
   - Double quotes: `"text with \\n escapes"` (allows escape sequences)
   - Block scalar for complex strings:
     ```yaml
     key: |
       Multi-line text with "quotes" and: colons
       Preserves formatting exactly
     ```

4. **Common fixes:**
   - `somebody said: hello` → `"somebody said: hello"`
   - `"hello" world` → `'"hello" world'` or `"\"hello\" world"`
   - `- Scene with "quotes"` → `- 'Scene with "quotes"'`

**Remember:** Unquoted special characters are interpreted as YAML syntax, not string content!
"""

        # Create the healing prompt
        healing_prompt = f"""You are a YAML healing expert. The following YAML has an error and needs to be fixed.

**Error Type:** {error_type}
**Error Message:**
{error_message}

**Expected Pydantic Model Schema:**
```yaml
{pydantic_to_yaml_example(model_class)}
```

**Current YAML (with errors):**
```yaml
{current_yaml}
```

Guidelines:
- Make sure to follow correct YAML template and usage:
{YAML_FIXING_GUIDE}

Please fix the YAML to match the expected schema. Return the corrected YAML in a ```yaml code block.
"""

        messages = MessageBuilder().add_system(healing_prompt).build()

        try:
            # Call AI to heal the YAML
            response, _ = call_ai(
                model=model, messages=messages, temperature=0, max_tokens=40000
            )

            # Extract the healed YAML from the response
            healed_yaml = find_yaml(response)
            if healed_yaml:
                current_yaml = healed_yaml

        except Exception as ai_error:
            print(f"AI healing failed: {ai_error}")
            # Continue with the original error
            continue

    # This should never be reached due to the raise in the loop
    raise ValueError("Unexpected error in YAML validation and healing process")
