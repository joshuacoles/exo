from typing import Dict, Any, Literal, Union

from pydantic import BaseModel, TypeAdapter


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

ToolChoiceModel = TypeAdapter(ToolChoice)
