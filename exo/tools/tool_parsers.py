from typing import Dict, Any, List, Optional, Literal, Type
import json
import re
from exo import DEBUG
from exo.tools import ToolDefinition, SpecificToolChoice, ToolChoice, AssistantToolCall
from typing import Protocol
from exo.inference.grammars import JSON_LARK_GRAMMAR


class Tokenizer(Protocol):
  def encode(self, text: str) -> List[int]:
    ...

  def decode(self, tokens: List[int]) -> str:
    ...


def get_parser_class(tool_call_format: Literal["tool_call", "llama_json", "watt"]) -> Type["ToolParser"]:
  if tool_call_format == "tool_call":
    return WrappedJsonToolParser
  elif tool_call_format == "llama_json":
    return LlamaPythonTag
  elif tool_call_format == "watt":
    return WattToolParser
  else:
    raise ValueError(f"Unknown tool parser format: {tool_call_format}")


class ToolParser:
  tokenizer: Tokenizer

  def __init__(self, tokenizer: Tokenizer, tools: List[ToolDefinition], tool_choice: Optional[ToolChoice]):
    self.tools = tools
    self.tokenizer = tokenizer
    if tool_choice is not None:
      self.tool_choice = tool_choice
    else:
      if tools is not None and len(tools) > 0:
        self.tool_choice = "auto"
      else:
        self.tool_choice = "none"

  def is_immediate(self) -> bool:
    """
    Returns whether the tool call is immediate.
    """
    return self.tool_choice == "required" or isinstance(self.tool_choice, SpecificToolChoice)

  def active_tools(self):
    if self.tool_choice == "none":
      return []
    elif self.tool_choice == "auto" or self.tool_choice == "required":
      return self.tools
    elif isinstance(self.tool_choice, SpecificToolChoice):
      return [tool for tool in self.tools if tool.name == self.tool_choice.function.name]
    else:
      raise ValueError(f"Invalid tool_choice: {self.tool_choice}")

  def start_token(self) -> int:
    """
    Returns the start token for tool calling.
    """
    raise NotImplementedError()

  def tool_grammar(self) -> str:
    """
    Returns an LLGuidance grammar for tool calling. This should include the start token and any end tokens.

    TODO: If tool calling is optional (ie not forced) then this should include a TEXT production which allows for
          arbitrary text to be parsed.
    """
    raise NotImplementedError()

  def parse_tool_calls(self, content: str) -> tuple[str, list[AssistantToolCall.AssistantTooCallInner]]:
    """
    Parses the tool calls from the content.

    Returns a tuple of the remaining content and the list of tool calls.
    """
    raise NotImplementedError()


class WrappedJsonToolParser(ToolParser):
  def __init__(self, tokenizer: Tokenizer, tools: List[ToolDefinition], tool_choice: Optional[ToolChoice]):
    super().__init__(tokenizer, tools, tool_choice)

  def start_token(self):
    return self.tokenizer.encode("<tool_call>")[0]

  def tool_grammar(self):
    # TODO: How do we handle tokens here?
    return f"""
%llguidance {{}}

start: <tool_call> json_body </tool_call>
json_body: %json{generate_tool_call_json_schema(self.active_tools())}
""".strip()

  def parse_tool_calls(self, content: str) -> tuple[str, list[AssistantToolCall.AssistantTooCallInner]]:
    offset = 0
    tool_calls = []

    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", content)):
      if i == 0:
        offset = m.start()
      try:
        tool_calls.append(AssistantToolCall.AssistantTooCallInner.model_validate_json(m.group(1)))
      except json.JSONDecodeError as e:
        if DEBUG >= 2: print(f"Failed to parse standard tool calls: {e}")

    return content[offset:], tool_calls


class LlamaPythonTag(ToolParser):
  def __init__(self, tokenizer: Tokenizer, tools: List[ToolDefinition], tool_choice: Optional[ToolChoice]):
    super().__init__(tokenizer, tools, tool_choice)

  def start_token(self):
    return 128010

  def tool_grammar(self) -> str:
    # This is lifted from https://github.com/guidance-ai/llguidance/blob/cc83715f/docs/syntax.md#special-tokens
    return f"""
%llguidance {{}}

start: TEXT | fun_call
TEXT: /[^{{](.|\n)*/
fun_call: <|python_tag|> json_body <|eom_id|>
json_body: %json{json.dumps(generate_tool_call_json_schema(self.active_tools(), "parameters"))}
    """.strip()

  def parse_tool_calls(self, content: str) -> tuple[str, list[AssistantToolCall.AssistantTooCallInner]]:
    offset = 0
    tool_calls = []

    for i, m in enumerate(re.finditer(r"<\|python_tag\|>(.+)<\|eom_id\|>", content)):
      if i == 0:
        offset = m.end()

      try:
        remapped = json.loads(m.group(1))

        # Rename "parameters" to "arguments" as that is the expected format
        tool_calls.append(AssistantToolCall.AssistantTooCallInner.model_validate({
          "name": remapped["name"],
          "arguments": json.dumps(remapped["parameters"])
        }))
      except json.JSONDecodeError as e:
        if DEBUG >= 2: print(f"Failed to parse python_tag tool calls: {e}")

    return content[offset:], tool_calls


def generate_tool_call_json_schema(tools: List[ToolDefinition], parameter_key: str = "arguments") -> Dict[str, Any]:
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
        # TODO: The LLama example on LLGuidance uses "name": { "const": "get_weather" } which might be easier?
        "name": {
          "type": "string",
          "enum": [tool.name]
        },
        parameter_key: tool.parameters
      },
      "required": ["name", parameter_key],
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


class WattToolParser(ToolParser):
  def __init__(self, tokenizer: Tokenizer, tools: List[ToolDefinition], tool_choice: Optional[ToolChoice]):
    super().__init__(tokenizer, tools, tool_choice)

  def start_token(self):
    return self.tokenizer.encode("[")[0]

  def tool_grammar(self) -> str:
    # Extract JSON value definitions from the JSON grammar
    json_value_defs = "\n".join([
      line for line in JSON_LARK_GRAMMAR.split("\n")
      if any(term in line for term in ["value:", "object:", "array:", "STRING:", "NUMBER:", "WS:"])
    ])

    # Create a grammar that matches the format [func_name(params_name=params_value, ...)]
    return f"""
%llguidance {{}}

start: "[" function_call "]"
function_call: function_name "(" parameter_list ")"
function_name: {self._generate_function_names()}
parameter_list: parameter | parameter "," parameter_list | ""
parameter: parameter_name "=" parameter_value
parameter_name: /[a-zA-Z_][a-zA-Z0-9_]*/
parameter_value: value

value: object
     | array
     | STRING
     | NUMBER
     | ("true" | "false" | "null") WS

object: "{{" WS (STRING ":" WS value ("," WS STRING ":" WS value)*)? "}}" WS
array: "[" WS (value ("," WS value)*)? "]" WS

// escapes
STRING: "\"" ((/[^"\\x7F\\x00-\\x1F]/ | "\\" (/["\\bfnrt]/ | "u" /[0-9a-fA-F]/{{4,4}})))* "\"" WS

NUMBER: "-"? (/[0-9]/ | /[1-9]/ /[0-9]/{{0,15}}) ("." /[0-9]/+)? (/[eE]/ /[-+]/? /[0-9]/ /[1-9]/{{0,15}})? WS

// Optional space: by convention, applied in this grammar after literal chars when allowed
WS: "" | " " | "\n" /[ \t]/{{0,20}}
    """.strip()

  def _generate_function_names(self) -> str:
    """Generate a grammar rule for function names based on available tools."""
    function_names = [tool.name for tool in self.active_tools()]
    if not function_names:
      return '""'  # Empty string if no functions available
    return ' | '.join([f'"{name}"' for name in function_names])

  def parse_tool_calls(self, content: str) -> tuple[str, list[AssistantToolCall.AssistantTooCallInner]]:
    offset = 0
    tool_calls = []

    # Match patterns like [func_name(param1=value1, param2=value2)]
    for i, m in enumerate(re.finditer(r'\[([\w_]+)\((.*?)\)\]', content)):
      if i == 0:
        offset = m.end()

      try:
        func_name = m.group(1)
        params_str = m.group(2)

        # Parse parameters from the format param1=value1, param2=value2
        params = {}
        if params_str.strip():
          # More robust parameter parsing with regex that handles nested structures
          param_pattern = r'(\w+)=([^,]+?)(?=,\s*\w+=|$)'
          param_pairs = re.findall(param_pattern, params_str)

          for param_name, param_value in param_pairs:
            # Try to parse the parameter value - it could be a string, number, boolean, etc.
            try:
              # First try to parse as JSON (for numbers, booleans, null, arrays, objects)
              params[param_name] = json.loads(param_value.strip())
            except json.JSONDecodeError:
              # If that fails, treat it as a string (removing quotes if present)
              param_value = param_value.strip()
              if (param_value.startswith('"') and param_value.endswith('"')) or \
                (param_value.startswith("'") and param_value.endswith("'")):
                param_value = param_value[1:-1]
              params[param_name] = param_value

        # Create the tool call object
        tool_calls.append(AssistantToolCall.AssistantTooCallInner.model_validate({
          "name": func_name,
          "arguments": json.dumps(params)
        }))
      except Exception as e:
        if DEBUG >= 2: print(f"Failed to parse Watt tool calls: {e}")

    return content[offset:], tool_calls
