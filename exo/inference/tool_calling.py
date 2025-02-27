from pydantic import BaseModel
from typing import Dict, Any, Union, Literal, List, Optional
import json


class ToolDefinition(BaseModel):
  """
  The definition of a tool in the OpenAI API response.
  """
  name: str
  description: str
  parameters: Dict[str, Any]


class AssistantTooCall(BaseModel):
  class AssistantTooCallInner(BaseModel):
    name: str
    arguments: str

  """
  An element of message.tool_calls
  """
  id: str
  type: Literal['function']
  function: AssistantTooCallInner


class SpecificToolChoice(BaseModel):
  class SpecificToolChoiceInner(BaseModel):
    name: str

  type: Literal["function"]
  function: SpecificToolChoiceInner


ToolChoice = Union[
  # none => ignore tools,
  # auto => model decides whether to use tools,
  # required => model must use tools,
  # specific => model must use specific tools
  Literal["none", "auto", "required"],
  SpecificToolChoice
]


class ToolParser:
  def __init__(self, tools: List[ToolDefinition], tool_choice: Optional[ToolChoice]):
    self.tools = tools
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


class WrappedJsonToolParser(ToolParser):
  def __init__(self, tools: List[ToolDefinition], tool_choice: Optional[ToolChoice], start_token: str, end_token: str):
    super().__init__(tools, tool_choice)
    self.start_token = start_token
    self.end_token = end_token


  def start_token(self):
    return self.start_token

  def tool_grammar(self):
    # TODO: How do we handle tokens here?
    return f"""
    %llguidance {{}}

    start: {self.start_token} tool_call {self.end_token}
    tool_call: %json{{{generate_tool_call_json_schema(self.active_tools())}}}
    """

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
