from typing import Dict, Any, Union, List, Optional
import json
import re
from exo import DEBUG
from exo.tools import ToolDefinition, SpecificToolChoice, ToolChoice, AssistantTooCall
from typing import Protocol


class Tokenizer(Protocol):
  def encode(self, text: str) -> List[int]:
    ...

  def decode(self, tokens: List[int]) -> str:
    ...


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
    """
    raise NotImplementedError()

  def parse_tool_calls_complete(self, content: str) -> Optional[list[dict]]:
    raise NotImplementedError()

  def is_new_tool_call_ready_to_emit(self, buffered_content: str) -> bool:
    """
    This should emit if the prefix matches
    """
    raise NotImplementedError()

  def parse_new_tool_call_emission(self, content: str) -> AssistantTooCall.AssistantTooCallInner:
    raise NotImplementedError()

  def parse_new_tool_call(self, tokens: list[int]) -> Optional[AssistantTooCall.AssistantTooCallInner]:
    content = self.tokenizer.decode(tokens)

    if self.is_new_tool_call_ready_to_emit(content):
      return self.parse_new_tool_call_emission(content)
    else:
      return None


class WrappedJsonToolParser(ToolParser):
  def __init__(self, tokenizer: Tokenizer, tools: List[ToolDefinition], tool_choice: Optional[ToolChoice]):
    super().__init__(tokenizer, tools, tool_choice)

  def start_token(self):
    return self.tokenizer.encode("<tool_call>")[0]

  def tool_grammar(self):
    # TODO: How do we handle tokens here?
    return f"""
    %llguidance {{}}

    start: "<tool_call>" json_body "</tool_call>"
    json_body: %json{{{generate_tool_call_json_schema(self.active_tools())}}}
    """

  def parse_tool_calls_complete(self, tokens: list[int]) -> tuple[list[int], list[dict]]:
    offset = 0
    tool_calls = []

    for i, m in enumerate(re.finditer(r"<tool_call>\n(.+)?\n</tool_call>", tokens)):
      if i == 0:
        offset = m.start()
      try:
        func = json.loads(m.group(1))
        tool_calls.append({"type": "function", "function": func})
        if isinstance(func["arguments"], str):
          func["arguments"] = json.loads(func["arguments"])
      except json.JSONDecodeError as e:
        if DEBUG >= 2: print(f"Failed to parse standard tool calls: {e}")

    return tokens[:offset], tool_calls

  def is_new_tool_call_ready_to_emit(self, content: str) -> bool:
    # TODO: Move to just emitting a tool call entirely in one chunk, incrementally streaming the JSON is a PITA for now and requires incremental parsing. Keep the logic in the API for it though.
    """
    Check if the content contains a complete tool call that is ready to be emitted.
    
    A tool call is ready to emit when it contains enough information to be processed,
    even if the JSON is not fully complete. We need to check for the presence of
    essential fields using string matching since standard JSON parsing may fail
    on incomplete JSON.
    
    At this stage, we only need to verify the name property is present and valid,
    without considering the arguments.
    """
    # Check if content starts with the start token
    if not content.startswith("<tool_call>"):
      return False
      
    # Strip the start token to get just the JSON-like content
    json_content = content[len("<tool_call>"):].strip()
    
    # Check if we have enough content to process
    if not json_content:
      return False
  
    # Check if the content contains the essential "name" field
    return bool(re.search(r'"name"\s*:\s*"([^"]+)"', json_content))

  def parse_new_tool_call_emission(self, content: str) -> AssistantTooCall.AssistantTooCallInner:
    return json.loads(content[len(self._start_token):])


class LlamaPythonTag(ToolParser):
  def __init__(self, tools: List[ToolDefinition], tool_choice: Optional[ToolChoice], start_token: str, end_token: str):
    super().__init__(tools, tool_choice)
    self.start_token = start_token
    self.end_token = end_token

  def tool_grammar(self) -> str:
    # This is lifted from https://github.com/guidance-ai/llguidance/blob/cc83715f/docs/syntax.md#special-tokens
    return f"""
    %llguidance {{}}

    # start: TEXT | fun_call
    # TEXT: /[^{{](.|\n)*/
    start: fun_call
    fun_call: <|python_tag|> json_body <|eom_id|>
    json_body: %json{{{generate_tool_call_json_schema(self.active_tools(), "parameters")}}}
    """

  def parse_tool_calls_complete(self, content: str) -> tuple[list[int], list[dict]]:
    offset = 0
    tool_calls = []

    for i, m in enumerate(re.finditer(r"<\|python_tag\|>(.+)", content)):
      if i == 0:
        offset = m.start()

      try:
        func_data = json.loads(m.group(1))
        # Convert from "parameters" format to "arguments" format if needed
        if "parameters" in func_data and "arguments" not in func_data:
          func = {
            "name": func_data["name"],
            "arguments": func_data["parameters"]
          }
        else:
          func = func_data

        tool_calls.append({"type": "function", "function": func})
      except json.JSONDecodeError as e:
        if DEBUG >= 2: print(f"Failed to parse python_tag tool calls: {e}")


def generate_tool_grammar(tools: List[ToolDefinition], tool_choice: Union[ToolChoice, None]) -> Union[str, None]:
  """
  Generate a grammar for tool calling.
  """

  if tool_choice == "none":
    return None
  elif tool_choice == "required" or tool_choice is None:
    return json_schema_to_grammar(generate_tool_call_json_schema(tools))
  elif isinstance(tool_choice, SpecificToolChoice):
    tool = next((tool for tool in tools if tool.name == tool_choice.function.nam), None)
    if tool is None:
      raise ValueError(f"Tool {tool_choice.function} not found")
    return json_schema_to_grammar(generate_tool_call_json_schema([tool]))
  else:
    raise ValueError(f"Invalid tool choice: {tool_choice}")


def json_schema_to_grammar(json_schema: Dict[str, Any]) -> str:
  """
  Convert a JSON schema to a grammar.
  """
  return json.dumps({"grammars": [{"json_schema": json_schema}]})


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
