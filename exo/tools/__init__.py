from typing import Literal, Union
from pydantic import BaseModel, TypeAdapter


class SpecificToolChoice(BaseModel):
  class SpecificToolChoiceInner(BaseModel):
    name: str

  type: Literal["function"] = "function"
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
