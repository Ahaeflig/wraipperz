from typing import Type, get_origin, get_args, Union, Any, Dict
from pydantic import BaseModel
from datetime import date
import enum


def find_yaml(text):

    def get_indentation(line):
        return len(line) - len(line.lstrip())

    lines = text.split("\n")
    start_index = -1
    end_index = -1

    # Find the start index
    for i, line in enumerate(lines):
        if line.strip() == "```yaml":
            start_index = i + 1
            break

    # If we didn't find the start, return the original text
    if start_index == -1:
        return ""

    end_start_index = len(lines)
    # Check if there is another yaml block starting ```yaml
    for i in range(start_index + 1, len(lines), 1):
        if lines[i] == "```yaml":
            end_start_index = i

    # Find the end index, searching from the end
    for i in range(end_start_index, start_index, -1):
        if lines[i - 1].strip() == "```":
            end_index = i
            break

    if end_index == start_index + 1:
        return ""
    elif end_index == -1:
        return ""

    # Extract YAML content
    yaml_lines = lines[start_index : end_index - 1]

    min_indent = get_indentation(lines[start_index])

    # Remove the minimum indentation from each line
    yaml_content = "\n".join(line[min_indent:] if line.strip() else "" for line in yaml_lines)

    return yaml_content.strip()

def pydantic_to_yaml_example(model_class: Type[BaseModel]) -> str:
    """
    Convert a Pydantic model class to a YAML representation with example values.
    
    Args:
        model_class: A class that inherits from pydantic.BaseModel
        
    Returns:
        A string containing the YAML representation with examples
    """
    if not issubclass(model_class, BaseModel):
        raise TypeError("Input must be a Pydantic BaseModel class")
    
    # Process each field to create the YAML with examples
    yaml_lines = []
    
    for field_name, model_field in model_class.model_fields.items():
        # Extract comment if present in json_schema_extra
        comment = ""
        if model_field.json_schema_extra and "comment" in model_field.json_schema_extra:
            comment = f" # {model_field.json_schema_extra['comment']}"
        
        # Get example from field
        example = model_field.json_schema_extra.get("example") if model_field.json_schema_extra else None
        
        # If no example is provided, generate a default one based on the type
        if example is None:
            example = generate_default_example(model_field.annotation)
        
        # Generate YAML for this field with proper indentation
        field_yaml = format_field_yaml(field_name, example, comment)
        yaml_lines.extend(field_yaml)
    
    return '\n'.join(yaml_lines)

def format_field_yaml(field_name: str, value: Any, comment: str = "", indent: int = 0) -> list:
    """
    Format a field and its value as YAML with proper indentation.
    
    Args:
        field_name: The name of the field
        value: The value to format
        comment: Optional comment to add
        indent: Current indentation level
        
    Returns:
        List of YAML lines
    """
    indent_str = " " * indent
    lines = []
    
    # Handle different value types
    if isinstance(value, list):
        lines.append(f"{indent_str}{field_name}:{comment}")
        lines.extend(format_list_yaml(value, indent + 2))
    elif isinstance(value, dict):
        lines.append(f"{indent_str}{field_name}:{comment}")
        lines.extend(format_dict_yaml(value, indent + 2))
    else:
        yaml_value = format_scalar_yaml(value)
        lines.append(f"{indent_str}{field_name}: {yaml_value}{comment}")
    
    return lines

def format_list_yaml(items: list, indent: int = 0) -> list:
    """
    Format a list as YAML with proper indentation.
    
    Args:
        items: The list to format
        indent: Current indentation level
        
    Returns:
        List of YAML lines
    """
    if not items:
        return [f"{' ' * indent}[]"]
    
    indent_str = " " * indent
    lines = []
    
    for item in items:
        if isinstance(item, dict):
            # Use a single line with dash and first item if dictionary
            dict_lines = format_dict_yaml(item, indent + 2)
            if dict_lines:
                # Add the dash to the beginning of the first line instead of as a separate line
                first_key = list(item.keys())[0] if item else "empty"
                lines.append(f"{indent_str}- {first_key}: {format_scalar_yaml(item[first_key])}")
                # Add the rest of the items with proper indentation
                for line in dict_lines[1:]:
                    lines.append(line)
            else:
                lines.append(f"{indent_str}- {{}}")
        elif isinstance(item, list):
            lines.append(f"{indent_str}- ")
            list_lines = format_list_yaml(item, indent + 2)
            lines.extend(list_lines)
        elif hasattr(item, '__dict__') and not isinstance(item, (str, int, float, bool)):
            # Handle Pydantic model with __dict__ - extract dictionary and format better
            if hasattr(item, 'model_dump'):
                obj_dict = item.model_dump()
            else:
                obj_dict = item.__dict__
                
            if obj_dict:
                # Get the first key to start the line with
                first_key = list(obj_dict.keys())[0]
                first_value = obj_dict[first_key]
                # Start with dash and the first property on the same line
                lines.append(f"{indent_str}- {first_key}: {format_scalar_yaml(first_value)}")
                
                # Add remaining properties with correct indentation
                for key, value in obj_dict.items():
                    if key != first_key:  # Skip the first one we already added
                        if isinstance(value, (dict, list)):
                            lines.append(f"{' ' * (indent+2)}{key}:")
                            if isinstance(value, dict):
                                lines.extend(format_dict_yaml(value, indent + 4))
                            else:
                                lines.extend(format_list_yaml(value, indent + 4))
                        else:
                            lines.append(f"{' ' * (indent+2)}{key}: {format_scalar_yaml(value)}")
            else:
                lines.append(f"{indent_str}- {{}}")
        else:
            scalar_value = format_scalar_yaml(item)
            # Strip quotes for list items to match YAML conventions
            if isinstance(item, str):
                scalar_value = scalar_value.strip('"')
            lines.append(f"{indent_str}- {scalar_value}")
    
    return lines

def format_dict_yaml(data: Dict, indent: int = 0) -> list:
    """
    Format a dictionary as YAML with proper indentation.
    
    Args:
        data: The dictionary to format
        indent: Current indentation level
        
    Returns:
        List of YAML lines
    """
    if not data:
        return [f"{' ' * indent}{{}}"]
    
    indent_str = " " * indent
    lines = []
    
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{indent_str}{key}:")
            lines.extend(format_dict_yaml(value, indent + 2))
        elif isinstance(value, list):
            lines.append(f"{indent_str}{key}:")
            lines.extend(format_list_yaml(value, indent + 2))
        else:
            scalar_value = format_scalar_yaml(value)
            lines.append(f"{indent_str}{key}: {scalar_value}")
    
    return lines

def format_scalar_yaml(value: Any) -> str:
    """
    Format a scalar value for YAML output.
    
    Args:
        value: The value to format
        
    Returns:
        Formatted string
    """
    if value is None:
        return "null"
    elif isinstance(value, str):
        # Quote strings
        return f'"{value}"'
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, (set, tuple)):
        return format_scalar_yaml(list(value))
    
    return str(value)

def generate_default_example(type_annotation):
    """Generate a default example value based on the type."""
    origin = get_origin(type_annotation)
    args = get_args(type_annotation)
    
    # Handle Optional types
    if origin is Union and type(None) in args:
        non_none_args = [arg for arg in args if arg is not type(None)]
        if len(non_none_args) == 1:
            return generate_default_example(non_none_args[0])
    
    # Handle List
    if origin is list:
        if args:
            return [generate_default_example(args[0]), generate_default_example(args[0])]
        return []
    
    # Handle Dict
    if origin is dict:
        if len(args) >= 2:
            key_type, value_type = args[0], args[1]
            
            # Handle nested dictionary types like Dict[str, Dict[str, List[str]]]
            if get_origin(value_type) is dict and len(get_args(value_type)) >= 2:
                sub_key_type, sub_value_type = get_args(value_type)
                
                # Create a more realistic nested example
                key1 = generate_default_example(key_type)
                key2 = f"{key1}_2" if isinstance(key1, str) else key1
                
                # Handle nested list in dictionary
                if get_origin(sub_value_type) is list:
                    list_item_type = get_args(sub_value_type)[0]
                    sub_dict = {
                        generate_default_example(sub_key_type): [
                            generate_default_example(list_item_type),
                            generate_default_example(list_item_type)
                        ],
                        f"{generate_default_example(sub_key_type)}_2": [
                            generate_default_example(list_item_type),
                            generate_default_example(list_item_type)
                        ]
                    }
                    return {key1: sub_dict, key2: sub_dict}
                else:
                    sub_dict = {
                        generate_default_example(sub_key_type): generate_default_example(sub_value_type),
                        f"{generate_default_example(sub_key_type)}_2": generate_default_example(sub_value_type)
                    }
                    return {key1: sub_dict, key2: sub_dict}
            
            # Handle Dict[str, List[str]]
            elif get_origin(value_type) is list:
                list_item_type = get_args(value_type)[0]
                key1 = generate_default_example(key_type)
                key2 = f"{key1}_2" if isinstance(key1, str) else key1
                return {
                    key1: [generate_default_example(list_item_type), generate_default_example(list_item_type)],
                    key2: [generate_default_example(list_item_type), generate_default_example(list_item_type)]
                }
            
            # Simple Dict[K, V]
            else:
                key1 = generate_default_example(key_type)
                key2 = f"{key1}_2" if isinstance(key1, str) else key1
                return {key1: generate_default_example(value_type), key2: generate_default_example(value_type)}
        return {}
    
    # Handle Set
    if origin is set:
        if args:
            return {generate_default_example(args[0])}
        return set()
    
    # Handle Tuple
    if origin is tuple:
        return tuple(generate_default_example(arg) for arg in args) if args else ()
    
    # Handle Enum
    if isinstance(type_annotation, type) and issubclass(type_annotation, enum.Enum):
        return list(type_annotation)[0].value
    
    # Handle basic types
    if type_annotation is str:
        return "example string"
    elif type_annotation is int:
        return 42
    elif type_annotation is float:
        return 3.14
    elif type_annotation is bool:
        return True
    elif type_annotation is date:
        return "2023-01-01"
    
    # Default fallback
    return None