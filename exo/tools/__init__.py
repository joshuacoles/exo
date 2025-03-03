from typing import Dict, Any, Literal, Union

from pydantic import BaseModel, TypeAdapter


class ToolDefinition(BaseModel):
  """
  The definition of a tool in the OpenAI API response.
  """
  name: str
  description: str
  parameters: Dict[str, Any]


class AssistantToolCall(BaseModel):
  class AssistantTooCallInner(BaseModel):
    name: str

    # For complete tool calls this should be the JSON encoded arguments for the function, conforming to the parameters
    # JSON schema defined in the tool definition. In the case of a streamed response this may be an incremental chunk,
    # which when assembled into a whole, conforms to the JSON schema.
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

ToolChoiceModel = TypeAdapter(ToolChoice)
