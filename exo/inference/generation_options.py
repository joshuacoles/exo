from typing import Optional, List, Dict, Any, Union

from exo.inference.tool_calling import ToolParser, WrappedJsonToolParser
from exo.tools import ToolDefinition, ToolChoiceModel


class GenerationOptions:
  max_completion_tokens: Optional[int] = None

  # Textual stop sequences that will halt generation when encountered
  stop: Optional[List[str]] = None
  temperature: Optional[float] = None

  # Stuff to do with
  grammar_definition: Optional[str] = None
  tools: Optional[List[Dict[str, Any]]] = None
  tool_choice: Optional[Dict[str, Any]] = None

  def __init__(
    self,
    max_completion_tokens: Optional[int] = None,
    stop: Optional[List[str]] = None,
    temperature: Optional[float] = None,
    grammar_definition: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
  ):
    self.max_completion_tokens = max_completion_tokens
    self.stop = stop
    self.temperature = temperature
    self.grammar_definition = grammar_definition
    self.tools = tools
    self.tool_choice = tool_choice

  def tool_parser(self) -> Optional[ToolParser]:
    if not self.tools:
      return None

    # Convert the tools list to ToolDefinition objects
    tool_definitions = [ToolDefinition.model_validate(tool) for tool in self.tools]
    tool_choice = ToolChoiceModel.validate_python(self.tool_choice) if self.tool_choice is not None else None

    # Use WrappedJsonToolFormat as the format class
    # We need to provide start and end tokens for the format
    return WrappedJsonToolParser(
      tools=tool_definitions,
      tool_choice=tool_choice,
      start_token="<tool_call>",
      end_token="</tool_call>"
    )
