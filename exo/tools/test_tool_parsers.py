import pytest
import json
from transformers import AutoProcessor

from exo.api.chat_completion_request import ToolDefinition
from exo.download.new_shard_download import exo_home
from exo.tools.watt_tool_parser import WattToolParser
from exo.tools.llama_python_tag_tool_parser import LlamaPythonTag
from exo.tools.llama_3_3_custom_tools import Llama33ToolParser
from exo.inference.buffered_output import BufferedOutput


@pytest.fixture
def tokenizer():
  return AutoProcessor.from_pretrained(
    exo_home() / "downloads" / "mlx-community/Llama-3.2-1B-Instruct-4bit".replace("/", "--")
  )


@pytest.fixture
def watt_parser():
  return WattToolParser()


@pytest.fixture
def llama_parser():
  return LlamaPythonTag()


@pytest.fixture
def llama33_parser():
  return Llama33ToolParser()


def validate_with_guidance(parser, tools, text, tokenizer, required=True):
  """Validate text using actual LLInterpreter setup"""
  grammar = parser.to_grammar(tools, required, parallel_tool_calling=True)

  try:
    # Recreate the BufferedOutput setup used in production
    buffered = BufferedOutput(
      max_tokens=100,
      eos_token_id=-1,
      stop_sequences=[],
      tokenizer=tokenizer,
      grammar_definition=grammar
    )

    tokens = tokenizer.encode(text, add_special_tokens=False)

    # Feed text one character at a time (simple tokenization)
    for token in tokens:
      mask = buffered.get_token_mask()

      if mask is not None and mask[token] == 0:
        print(f"Token {tokenizer.decode([token])} is not in the mask")
        print("Tokens in mask: ", end="")
        for t, m in enumerate(mask):
          if m != 0:
            decoded = tokenizer.decode([t])
            print(f"[{decoded}]", end=", ")

        print()
        return "failed"

      buffered.append(token)

    if buffered.is_finished:
      return "finished"
    else:
      return "accepting"
  except Exception as e:
    if "Parser Error" in str(e):
      return "failed"
    raise


def test_watt_grammar_validation(watt_parser, tokenizer):
  tools = [
    ToolDefinition(
      type="function",
      function=ToolDefinition.FunctionDefinition(
        name="valid_tool",
        parameters={"type": "object", "properties": {}}
      )
    ),
    ToolDefinition(
      type="function",
      function=ToolDefinition.FunctionDefinition(
        name="tool",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}}
      )
    )
  ]

  assert validate_with_guidance(watt_parser, tools, "[valid_tool()]", tokenizer) == "finished"
  assert validate_with_guidance(watt_parser, tools, "[tool(name=\"value\")]", tokenizer) == "finished"
  assert validate_with_guidance(watt_parser, tools, "[tool(", tokenizer) == "accepting"
  assert validate_with_guidance(watt_parser, tools, "[tool(name='value')]", tokenizer) == "failed"


def test_llama_grammar_validation(llama_parser, tokenizer):
  tools = [
    ToolDefinition(
      type="function",
      function=ToolDefinition.FunctionDefinition(
        name="dummy",
        parameters={"type": "object", "properties": {}}
      )
    ),
    ToolDefinition(
      type="function",
      function=ToolDefinition.FunctionDefinition(
        name="with_arg",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}}
      )
    ),
    ToolDefinition(
      type="function",
      function=ToolDefinition.FunctionDefinition(
        name="with_arg_limited",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": False}
      )
    )
  ]

  assert validate_with_guidance(llama_parser, tools,
                                f"<|python_tag|>{json.dumps({"name": "dummy", "parameters": {}})}<|eom_id|>",
                                tokenizer) == "finished"

  assert validate_with_guidance(llama_parser, tools, "<|python_tag|>{}<|eom_id|>", tokenizer) == "failed"
  assert validate_with_guidance(llama_parser, tools, f"<|python_tag|>{json.dumps({"name": "dummy"})}<|eom_id|>",
                                tokenizer) == "failed"

  # Specified arg
  assert validate_with_guidance(llama_parser, tools,
                                f"<|python_tag|>{json.dumps({"name": "with_arg", "parameters": {"name": "value"}})}<|eom_id|>",
                                tokenizer) == "finished"

  # Additional arg in non-limited tool
  assert validate_with_guidance(llama_parser, tools,
                                f"<|python_tag|>{json.dumps({"name": "dummy", "parameters": {"allowed_but_not_in_schema": 1}})}<|eom_id|>",
                                tokenizer) == "finished"

  # Additional arg in limited tool
  assert validate_with_guidance(llama_parser, tools,
                                f"<|python_tag|>{json.dumps({"name": "with_arg_limited", "parameters": {"disallowed": 1}})}<|eom_id|>",
                                tokenizer) == "failed"

  # Wrong type
  assert validate_with_guidance(llama_parser, tools,
                                f"<|python_tag|>{json.dumps({"name": "with_arg", "parameters": {"name": 2}})}<|eom_id|>",
                                tokenizer) == "failed"

  # Not listed tool
  assert validate_with_guidance(llama_parser, tools,
                                f"<|python_tag|>{json.dumps({"name": "wrong_name", "parameters": {"name": "value"}})}<|eom_id|>",
                                tokenizer) == "failed"

  # Required tool call
  assert validate_with_guidance(llama_parser, tools, f"Not a tool call", tokenizer) == "failed"
  assert validate_with_guidance(llama_parser, tools,
                                f"Not a tool call<|python_tag|>{json.dumps({"name": "dummy", "parameters": {}})}<|eom_id|>",
                                tokenizer) == "failed"

  # Text generation
  assert validate_with_guidance(llama_parser, tools, f"Not a tool call", tokenizer, required=False) == "accepting"
  assert validate_with_guidance(llama_parser, tools,
                                f"Cannot generate a later tool call<|python_tag|>{json.dumps({"name": "dummy", "parameters": {}})}<|eom_id|>",
                                tokenizer, required=False) == "failed"


def test_watt_parallel_tool_calling_grammar(watt_parser, tokenizer):
  # Test Watt parallel calls
  watt_tools = [
    ToolDefinition(
      type="function",
      function=ToolDefinition.FunctionDefinition(name=f"tool{i}", parameters={})
    ) for i in range(3)
  ]

  assert validate_with_guidance(
    watt_parser, watt_tools,
    "[tool0(), tool1(), tool2()]",
    tokenizer
  ) == "finished"

  assert validate_with_guidance(
    watt_parser, watt_tools,
    "[tool1(), tool2(),]",
    tokenizer
  ) == "failed"

  assert validate_with_guidance(
    watt_parser, watt_tools,
    "[tool1(), tool2]",
    tokenizer
  ) == "failed"


def test_watt_parallel_tool_calling_parse_complete(watt_parser, tokenizer):
  parsed_tool_calls = watt_parser.parse_complete("[tool1(), tool2(), tool3()]", True)

  assert len(parsed_tool_calls) == 3
  assert parsed_tool_calls[0].name == "tool1"
  assert parsed_tool_calls[1].name == "tool2"
  assert parsed_tool_calls[2].name == "tool3"


# def test_llama_parallel_tool_calling_grammar(llama_parser, tokenizer):
#   llama_tools = [
#     ToolDefinition(
#       type="function",
#       function=ToolDefinition.FunctionDefinition(name=f"tool{i}", parameters={})
#     ) for i in range(2)
#   ]
#
#   assert validate_with_guidance(
#     llama_parser, llama_tools,
#     "<|python_tag|>{\"name\":\"tool1\"}<|eom_id|><|python_tag|>{\"name\":\"tool2\"}<|eom_id|>",
#     tokenizer
#   ) != "failed"


# def test_llama_parallel_tool_calling_parse_complete(llama_parser, tokenizer):
#   llama_tools = [
#     ToolDefinition(
#       type="function",
#       function=ToolDefinition.FunctionDefinition(name=f"tool{i}", parameters={})
#     ) for i in range(2)
#   ]
#
#   tool_calls = llama_parser.parse_complete(
#     "<|python_tag|>{\"name\":\"tool1\"}<|eom_id|><|python_tag|>{\"name\":\"tool2\"}<|eom_id|>",
#     True
#   )
#
#   assert len(tool_calls) == 2
#   assert tool_calls[0].name == "tool1"
#   assert tool_calls[1].name == "tool2"


# Enhanced error handling tests
def test_error_handling(watt_parser, llama_parser, llama33_parser):
  # Test malformed Watt patterns
  assert len(watt_parser.parse_complete("[unclosed_tool(", True)) == 0
  # TODO: This passes
  # assert len(watt_parser.parse_complete("[wrong_quotes(param='value)]", True)) == 0

  # Test malformed Llama patterns
  assert len(llama_parser.parse_complete("<|python_tag|>unclosed", True)) == 0
  assert len(llama_parser.parse_complete("<|python_tag|>{invalid_json}<|eom_id|>", True)) == 0


# Enhanced error handling tests
def test_llama33_error_handling(watt_parser, llama_parser, llama33_parser):
  # Test malformed Llama 3.3 patterns
  assert len(llama33_parser.parse_complete("{unclosed_json", False)) == 0
  assert len(llama33_parser.parse_complete("{invalid_json}", False)) == 0
  assert len(llama33_parser.parse_complete("{\"name\": \"tool\"}", False)) == 0  # Missing parameters field


def test_llama33_grammar_validation(llama33_parser, tokenizer):
  tools = [
    ToolDefinition(
      type="function",
      function=ToolDefinition.FunctionDefinition(
        name="dummy",
        parameters={"type": "object", "properties": {}}
      )
    ),
    ToolDefinition(
      type="function",
      function=ToolDefinition.FunctionDefinition(
        name="with_arg",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}}
      )
    ),
    ToolDefinition(
      type="function",
      function=ToolDefinition.FunctionDefinition(
        name="with_arg_limited",
        parameters={"type": "object", "properties": {"name": {"type": "string"}}, "additionalProperties": False}
      )
    )
  ]

  # Valid tool calls
  assert validate_with_guidance(llama33_parser, tools,
                              json.dumps({"name": "dummy", "parameters": {}}),
                              tokenizer) == "finished"

  assert validate_with_guidance(llama33_parser, tools,
                              json.dumps({"name": "with_arg", "parameters": {"name": "value"}}),
                              tokenizer) == "finished"

  # Invalid tool calls
  assert validate_with_guidance(llama33_parser, tools, "{}", tokenizer) == "failed"
  assert validate_with_guidance(llama33_parser, tools, json.dumps({"name": "dummy"}), tokenizer) == "failed"

  # Additional arg in non-limited tool
  assert validate_with_guidance(llama33_parser, tools,
                              json.dumps({"name": "dummy", "parameters": {"allowed_but_not_in_schema": 1}}),
                              tokenizer) == "finished"

  # Additional arg in limited tool
  assert validate_with_guidance(llama33_parser, tools,
                              json.dumps({"name": "with_arg_limited", "parameters": {"disallowed": 1}}),
                              tokenizer) == "failed"

  # Wrong type
  assert validate_with_guidance(llama33_parser, tools,
                              json.dumps({"name": "with_arg", "parameters": {"name": 2}}),
                              tokenizer) == "failed"

  # Not listed tool
  assert validate_with_guidance(llama33_parser, tools,
                              json.dumps({"name": "wrong_name", "parameters": {"name": "value"}}),
                              tokenizer) == "failed"

  # Required tool call
  assert validate_with_guidance(llama33_parser, tools, "Not a tool call", tokenizer) == "failed"

  # Text generation
  assert validate_with_guidance(llama33_parser, tools, "Not a tool call", tokenizer, required=False) == "accepting"


def test_llama33_parse_complete(llama33_parser):
  # Basic tool call
  simple_tool_call = json.dumps({"name": "simple_tool", "parameters": {}})
  parsed = llama33_parser.parse_complete(simple_tool_call, False)
  assert len(parsed) == 1
  assert parsed[0].name == "simple_tool"
  assert parsed[0].arguments == "{}"

  # Tool call with parameters
  tool_with_params = json.dumps({"name": "param_tool", "parameters": {"key": "value", "number": 42}})
  parsed = llama33_parser.parse_complete(tool_with_params, False)
  assert len(parsed) == 1
  assert parsed[0].name == "param_tool"
  assert json.loads(parsed[0].arguments) == {"key": "value", "number": 42}

  # Tool call embedded in text
  text_with_tool = "Here's a tool call: " + json.dumps({"name": "embedded_tool", "parameters": {"embedded": True}}) + " in text."
  parsed = llama33_parser.parse_complete(text_with_tool, False)
  assert len(parsed) == 1
  assert parsed[0].name == "embedded_tool"
  assert json.loads(parsed[0].arguments) == {"embedded": True}

  # Multiple tool calls in text
  multi_tools = (
    "First tool: " + json.dumps({"name": "first_tool", "parameters": {"order": 1}}) +
    " Second tool: " + json.dumps({"name": "second_tool", "parameters": {"order": 2}})
  )
  parsed = llama33_parser.parse_complete(multi_tools, False)
  assert len(parsed) == 2
  assert parsed[0].name == "first_tool"
  assert parsed[1].name == "second_tool"
  assert json.loads(parsed[0].arguments) == {"order": 1}
  assert json.loads(parsed[1].arguments) == {"order": 2}

  # Invalid JSON
  invalid_json = "{not valid json}"
  parsed = llama33_parser.parse_complete(invalid_json, False)
  assert len(parsed) == 0

  # Missing required fields
  missing_fields = json.dumps({"name": "no_params"})
  parsed = llama33_parser.parse_complete(missing_fields, False)
  assert len(parsed) == 0
