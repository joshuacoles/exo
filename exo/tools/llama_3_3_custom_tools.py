import json
import re
from typing import Any

from exo.api.chat_completion_request import ToolDefinition
from exo.api.inference_result_manager import InferenceResultChunk
from exo.helpers import DEBUG
from exo.inference.grammars import lark_grammar
from exo.tools.tool_parser import ToolParser, UnplacedToolCall


class Llama33ToolParser(ToolParser):
    def is_start_of_tool_section(self, chunk: InferenceResultChunk):
        # For Llama 3.3, tool calls begin with a JSON object after the assistant header
        # This looks for content that resembles the start of a JSON object
        text = chunk.text.strip()
        return text.startswith("{") # and ("\"name\"" in text or "'name'" in text)

    def parse_complete(self, text: str, parallel_tool_calling: bool = False) -> list[UnplacedToolCall]:
        """
        Parse tool calls from the complete text.
        For Llama 3.3, tool calls are formatted as JSON objects:
        {"name": "tool_name", "parameters": {...}}
        """
        if parallel_tool_calling:
            raise ValueError("Parallel tool calling not supported for Llama33ToolParser")

        tool_calls = []
        
        # More robust pattern to find JSON objects with proper nesting
        # This pattern tries to find complete JSON objects
        try:
            # First try to find entire valid JSON objects in the text
            for match in re.finditer(r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\})', text, re.DOTALL):
                json_str = match.group(1)
                try:
                    tool_data = json.loads(json_str)
                    
                    # Verify this is a tool call by checking for required fields
                    if "name" in tool_data and "parameters" in tool_data:
                        tool_calls.append(UnplacedToolCall(
                            name=tool_data["name"],
                            arguments=json.dumps(tool_data["parameters"])
                        ))
                except json.JSONDecodeError:
                    # If we can't parse it as JSON, just continue to the next match
                    continue
        except Exception as e:
            # If the regex approach fails, fall back to a simpler method
            if DEBUG >= 2: 
                print(f"Regex approach failed: {e}. Falling back to basic JSON extraction.")
            
            # Try to extract all text between an open brace and its matching close brace
            # This is less robust but might work as a fallback
            start_idx = text.find("{")
            if start_idx != -1:
                # Simple braces counter to find the matching closing brace
                open_braces = 0
                for i in range(start_idx, len(text)):
                    if text[i] == "{":
                        open_braces += 1
                    elif text[i] == "}":
                        open_braces -= 1
                        if open_braces == 0:
                            # Found a complete JSON object, try to parse it
                            json_str = text[start_idx:i+1]
                            try:
                                tool_data = json.loads(json_str)
                                if "name" in tool_data and "parameters" in tool_data:
                                    tool_calls.append(UnplacedToolCall(
                                        name=tool_data["name"],
                                        arguments=json.dumps(tool_data["parameters"])
                                    ))
                            except json.JSONDecodeError:
                                if DEBUG >= 2: 
                                    print(f"Failed to parse JSON: {json_str}")
                            break
                
        return tool_calls

    def to_grammar(self, tools: list[ToolDefinition], required: bool, parallel_tool_calling: bool) -> str:
        """
        Generate a grammar for tool calling in Llama 3.3 format.
        """
        if parallel_tool_calling:
            raise ValueError("Parallel tool calling not supported for Llama33ToolParser")
            
        # Create a JSON schema for the expected tool call format
        schema = self._generate_tool_call_json_schema(tools)
        
        return lark_grammar(f"""
%llguidance {{}}

start: {"tool_call" if required else "TEXT | tool_call"}
TEXT: /[^{{](.|\n)*/
tool_call: json_format
json_format: %json{json.dumps(schema)}
        """.strip())
        
    def _generate_tool_call_json_schema(self, tools: list[ToolDefinition]) -> dict[str, Any]:
        """
        Generate a JSON schema for Llama 3.3 tool calls.
        {"name": "tool_name", "parameters": {...}}
        """
        if len(tools) == 0:
            raise ValueError("No tools provided")

        schema_variants = []

        for tool in tools:
            # Create a schema variant for this tool
            tool_schema = {
                "type": "object",
                "properties": {
                    "name": {"const": tool.function.name},
                    "parameters": tool.function.parameters if getattr(tool.function, "strict", False) else {
                        "type": "object"
                    }
                },
                "required": ["name", "parameters"],
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
