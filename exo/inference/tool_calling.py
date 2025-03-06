from pydantic import BaseModel
from typing import Dict, Any, Union, Literal, List
import json

class ToolDefinition(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]


class SpecificToolChoice(BaseModel):
    function: str

    @classmethod
    def from_json(cls, json_dict: Dict[str, Any]) -> "SpecificToolChoice":
      if json_dict["type"] == "function":
        return cls(function=json_dict["function"]["name"])
      else:
        raise ValueError(f"Invalid tool choice type: {json_dict['type']}")

ToolChoice = Union[None, Literal["auto"], Literal["none"], SpecificToolChoice]

def generate_tool_grammar(tools: List[ToolDefinition], tool_choice: ToolChoice) -> Union[str, None]:
    """
    Generate a grammar for tool calling.
    """
    
    if tool_choice is None or tool_choice == "none":
        return None
    elif tool_choice == "auto":
        return json_schema_to_grammar(generate_tool_call_json_schema(tools))
    elif isinstance(tool_choice, SpecificToolChoice):
        tool = next((tool for tool in tools if tool.name == tool_choice.function), None)
        if tool is None:
            raise ValueError(f"Tool {tool_choice.function} not found")
        return json_schema_to_grammar(generate_tool_call_json_schema([tool]))
    else:
        raise ValueError(f"Invalid tool choice: {tool_choice}") 

def json_schema_to_grammar(json_schema: Dict[str, Any]) -> str:
    """
    Convert a JSON schema to a grammar.
    """
    return json.dumps({ "grammars": [{"json_schema": json_schema}] })
   
def generate_tool_call_json_schema(tools: List[ToolDefinition]) -> Dict[str, Any]:
    """
    Generate a JSON schema for tool calling. For a given tool name, the schema should have the rough form of:

    type ValidToolCall[name] = {
      "name": name,
      "arguments": tools[name].parameters
    }

    With the overall schema looking like:

    // For each tool in the list
    type ValidToolCall = ValidToolCall[name] | ...;

    Ie it should be a union of all the tool calls, disjoint from each other by the unqiue "name" field.
    """
    if len(tools) == 0:
      raise ValueError("No tools provided")

    schema_variants = []
    
    for tool in tools:
        # Create a schema variant for this tool
        tool_schema = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "enum": [tool.name]
                },
                "arguments": tool.parameters
            },
            "required": ["name", "arguments"],
            "additionalProperties": False
        }
        schema_variants.append(tool_schema)
    
    # Combine all tool schemas into a oneOf union
    if len(schema_variants) == 1:
        # Just return the single schema if only one tool
        return schema_variants[0]
    else:
        # Return a union of all tool schemas
        return {"oneOf": schema_variants}
