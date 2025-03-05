import uuid
import time
import asyncio
import json
from typing import List, Literal, Union, Dict, Any, Callable, Optional, AsyncIterator
from aiohttp import web
import traceback
from pydantic import BaseModel
from exo import DEBUG, VERSION
from exo.api.inference_result_manager import InferenceResultManager
from exo.inference.grammars import lark_grammar
from exo.inference.tokenizers import resolve_tokenizer
from exo.orchestration import Node
from exo.inference.generation_options import GenerationOptions
from exo.models import build_base_shard, model_cards, get_repo, get_model_card
from exo.api.response_formats import ResponseFormat, ResponseFormatAdapter, ResponseFormatUnion
from exo.tools import ToolChoiceModel, WrappedToolDefinition, AssistantToolCall
from exo.tools.tool_parsers import ToolParser, get_parser_class, Tokenizer


class Message(BaseModel):
  role: str
  content: Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]
  tools: Optional[List[Dict]] = None

  def to_dict(self):
    data = {"role": self.role, "content": self.content}
    if self.tools:
      data["tools"] = self.tools
    return data


class ChatCompletionRequest(BaseModel):
  model: str
  messages: List[Message]
  temperature: float = 0.7
  tools: Optional[List[Dict]] = None
  max_completion_tokens: Optional[int] = None
  stop: Optional[Union[str, List[str]]] = None
  response_format: Optional[ResponseFormatUnion] = None
  tool_choice: Optional[Union[str, Dict]] = None
  tool_call_format: Optional[str] = None

  @classmethod
  def parse_chat_request(cls, data: dict, default_model: str):
    # Ensure model is provided, falling back to the default
    model = ensure_model(data.get("model"), default_model)
    data["model"] = model

    # Get model card and fill in default details from there
    model_card = get_model_card(model) or {}
    if not "tool_call_format" in data:
      data["tool_call_format"] = model_card.get("default_tool_call_format", None)

    if not "max_completion_tokens" in data:
      data["max_completion_tokens"] = model_card.get("max_tokens", None)

    # Create the request object
    return cls.model_validate(data)


  def tool_parser(self, tokenizer: Tokenizer) -> Optional[ToolParser]:
    if not self.tools:
      return None

    # Convert the tools list to ToolDefinition objects
    tool_definitions = [WrappedToolDefinition.model_validate(tool).function for tool in self.tools]
    tool_choice = ToolChoiceModel.validate_python(self.tool_choice) if self.tool_choice is not None else None

    # Fix: Use a literal value instead of a string variable
    tool_call_format_literal: Literal['tool_call', 'llama_json', 'watt'] = "llama_json"
    if self.tool_call_format in ['tool_call', 'llama_json', 'watt']:
      tool_call_format_literal = self.tool_call_format

    tool_parser = get_parser_class(tool_call_format_literal)

    return tool_parser(
      tokenizer=tokenizer,
      tools=tool_definitions,
      tool_choice=tool_choice,
    )

  def to_generation_options(self, tokenizer: Tokenizer) -> GenerationOptions:
    # Determine grammar - either from response_format or function calling format
    grammar = None
    if self.response_format:
      grammar = self.response_format.to_grammar()

    tool_parser = self.tool_parser(tokenizer)

    if tool_parser and grammar:
      raise ValueError("Grammar definition and tool parser cannot both be provided")
    elif tool_parser:
      grammar = lark_grammar(tool_parser.tool_grammar())

    if isinstance(self.stop, list):
      stop = self.stop
    elif self.stop:
      stop = [self.stop]
    else:
      stop = None

    return GenerationOptions(
      max_completion_tokens=self.max_completion_tokens,
      stop=stop,
      grammar_definition=grammar,
      temperature=self.temperature,
    )


class ToolCallFunction(BaseModel):
  name: str = ""
  arguments: str = ""

  @classmethod
  def from_tool_call(cls, tool_call):
    return cls(
      name=tool_call.name,
      arguments=""
    )


class ToolCall(BaseModel):
  index: Optional[int] = None
  id: Optional[str] = None
  type: Literal["function"] = "function"
  function: ToolCallFunction

  @classmethod
  def from_tool_call(cls, tool_call: AssistantToolCall.AssistantTooCallInner, index: int, id: Optional[str] = None):
    return cls(
      index=index,
      id=id,
      type="function",
      function=ToolCallFunction.from_tool_call(tool_call)
    )


class Delta(BaseModel):
  role: Optional[str] = None
  content: Optional[str] = None
  tool_calls: Optional[List[ToolCall]] = None


class AssistantMessage(BaseModel):
  role: Literal["assistant"] = "assistant"
  content: Optional[str] = None
  tool_calls: Optional[List[Dict[str, Any]]] = None


class BaseChoice(BaseModel):
  index: int
  logprobs: Optional[Any] = None
  finish_reason: Optional[str] = None


class StreamingChoice(BaseChoice):
  delta: Delta

  @classmethod
  def create_tool_call_choice(cls, tool_call_index: int, tool_call: AssistantToolCall.AssistantTooCallInner, tool_call_id: Optional[str] = None):
    return cls(
      index=tool_call_index,
      logprobs=None,
      finish_reason=None,
      delta=Delta(
        tool_calls=[
          ToolCall.from_tool_call(tool_call, tool_call_index, tool_call_id)
        ]
      )
    )

  @classmethod
  def create_tool_call_arguments_choice(cls, tool_call_index: int, arguments: str):
    return cls(
      index=tool_call_index,
      logprobs=None,
      finish_reason=None,
      delta=Delta(
        tool_calls=[
          ToolCall(
            index=tool_call_index,
            function=ToolCallFunction(
              arguments=arguments
            )
          )
        ]
      )
    )

  @classmethod
  def create_content_choice(cls, content: str):
    return cls(
      index=0,
      logprobs=None,
      finish_reason=None,
      delta=Delta(
        role="assistant",
        content=content
      )
    )

  @classmethod
  def create_finish_choice(cls, finish_reason: Optional[str] = None):
    return cls(
      index=0,
      logprobs=None,
      finish_reason=finish_reason,
      delta=Delta()
    )


class NonStreamingChoice(BaseChoice):
  message: AssistantMessage

  @classmethod
  def create_choice(cls, content: Optional[str], tool_calls=None, finish_reason: Optional[str] = None,
                    api_tool_parser=None):
    tool_calls_list = None
    if api_tool_parser and tool_calls:
      tool_calls_list = [
        {
          "id": str(uuid.uuid4()),
          "type": "function",
          "function": tool_call.model_dump()
        } for tool_call in tool_calls
      ]

    return cls(
      index=0,
      logprobs=None,
      finish_reason=finish_reason,
      message=AssistantMessage(
        role="assistant",
        content=content if not tool_calls else None,
        tool_calls=tool_calls_list
      )
    )


class CompletionObject(BaseModel):
  id: str
  object: Literal["chat.completion", "text_completion"]
  created: int
  model: str
  system_fingerprint: str
  choices: List[Union[StreamingChoice, NonStreamingChoice]]

  def with_choice(self, choice: Union[StreamingChoice, NonStreamingChoice]) -> "CompletionObject":
    return self.model_copy(update={"choices": [choice]})

  def with_choices(self, choices: List[Union[StreamingChoice, NonStreamingChoice]]) -> "CompletionObject":
    return self.model_copy(update={"choices": choices})

  @classmethod
  def chat_completion(cls, request_id: str, model: str,
                      choices: Optional[List[Union[StreamingChoice, NonStreamingChoice]]] = None) -> "CompletionObject":
    if choices is None:
      choices = []

    return cls(
      id=f"chatcmpl-{request_id}",
      object="chat.completion",
      created=int(time.time()),
      model=model,
      system_fingerprint=f"exo_{VERSION}",
      choices=choices,
    )

  @classmethod
  def text_completion(cls, request_id: str, model: str,
                      choices: Optional[List[Union[StreamingChoice, NonStreamingChoice]]] = None) -> "CompletionObject":
    if choices is None:
      choices = []

    return cls(
      id=f"chatcmpl-{request_id}",
      object="text_completion",
      created=int(time.time()),
      model=model,
      system_fingerprint=f"exo_{VERSION}",
      choices=choices,
    )


def remap_messages(messages: List[Message]) -> List[Message]:
  remapped_messages = []
  last_image = None
  for message in messages:
    if not isinstance(message.content, list):
      remapped_messages.append(message)
      continue

    remapped_content = []
    for content in message.content:
      if isinstance(content, dict):
        if content.get("type") in ["image_url", "image"]:
          image_url = content.get("image_url", {}).get("url") or content.get("image")
          if image_url:
            last_image = {"type": "image", "image": image_url}
            remapped_content.append({"type": "text", "text": "[An image was uploaded but is not displayed here]"})
        else:
          remapped_content.append(content)
      else:
        remapped_content.append(content)
    remapped_messages.append(Message(role=message.role, content=remapped_content))

  if last_image:
    # Replace the last image placeholder with the actual image content
    for message in reversed(remapped_messages):
      for i, content in enumerate(message.content):
        if isinstance(content, dict):
          if content.get("type") == "text" and content.get(
            "text") == "[An image was uploaded but is not displayed here]":
            message.content[i] = last_image
            return remapped_messages

  return remapped_messages


def build_prompt(tokenizer: Tokenizer, _messages: List[Message], tools: Optional[List[Dict]] = None,
                 tool_choice: Optional[Union[str, dict]] = None):
  messages = remap_messages(_messages)
  chat_template_args = {
    "conversation": [m.to_dict() for m in messages],
    "tokenize": False,
    "add_generation_prompt": True
  }

  if tools and tool_choice != "none":
    chat_template_args["tools"] = tools

  try:
    prompt = tokenizer.apply_chat_template(**chat_template_args)
    if DEBUG >= 3: print(f"!!! Prompt: {prompt}")
    return prompt
  except UnicodeEncodeError:
    # Handle Unicode encoding by ensuring everything is UTF-8
    chat_template_args["conversation"] = [
      {k: v.encode('utf-8').decode('utf-8') if isinstance(v, str) else v
       for k, v in m.to_dict().items()}
      for m in messages
    ]
    prompt = tokenizer.apply_chat_template(**chat_template_args)
    if DEBUG >= 3: print(f"!!! Prompt (UTF-8 encoded): {prompt}")
    return prompt
  except Exception as e:
    if DEBUG >= 1:
      print(f"Error applying chat template: {e}")
      print(f"Falling back to simple message concatenation")

    # Simple fallback if no chat template defined
    prompt = ""
    for m in messages:
      if m.role == "user":
        prompt += f"User: {m.content}\n"
      elif m.role == "assistant":
        prompt += f"Assistant: {m.content}\n"
      elif m.role == "system":
        prompt += f"{m.content}\n"
    prompt += "Assistant: "
    return prompt


def parse_chat_request(data: dict, default_model: str):
  # Parse response_format if provided
  response_format = None
  if "response_format" in data:
    response_format = ResponseFormat.parse_from_request(data["response_format"])

  model = ensure_model(data.get("model"), default_model)

  # Parse messages
  messages = [Message.model_validate(msg) for msg in data.get("messages", [])]

  # Get model card and handle None case for default_tool_call_format
  model_card = get_model_card(model) or {}
  default_tool_call_format = model_card.get("default_tool_call_format", None)

  # Create the request object
  return ChatCompletionRequest(
    model=model,
    messages=messages,
    temperature=data.get("temperature", 0.7),
    tools=data.get("tools"),
    max_completion_tokens=data.get("max_completion_tokens", data.get("max_tokens", None)),
    stop=data.get("stop"),
    response_format=response_format,
    tool_choice=data.get("tool_choice", None),
    tool_call_format=data.get("tool_call_format", default_tool_call_format)
  )


class ChatApi:
  inference_engine_classname: str
  result_manager: InferenceResultManager
  node: Node
  default_mode: str
  system_prompt: Optional[str]
  response_timeout: int

  def __init__(self, inference_engine_classname: str, node: Node,
               result_manager: InferenceResultManager,
               default_model: Optional[str] = None,
               system_prompt: Optional[str] = None,
               response_timeout: int = 90,
               on_chat_completion_request: Optional[Callable[[str, ChatCompletionRequest, str], None]] = None):
    self.inference_engine_classname = inference_engine_classname
    self.node = node
    self.default_model = default_model or "llama-3.2-1b"
    self.system_prompt = system_prompt
    self.response_timeout = response_timeout
    self.on_chat_completion_request = on_chat_completion_request
    self.result_manager = result_manager

  def sse_data_chunk(self, json_data: Union[dict, BaseModel]):
    if isinstance(json_data, BaseModel):
      json_data = json_data.model_dump()

    return f"data: {json.dumps(json_data)}\n\n".encode()

  async def handle_post_chat_completions(self, request):
    data = await request.json()
    if DEBUG >= 2: print(f"Handling chat completions request from {request.remote}: {data}")

    stream = data.get("stream", False)
    chat_request = parse_chat_request(data, self.default_model)

    shard = build_base_shard(chat_request.model, self.inference_engine_classname)
    if not shard:
      supported_models = [model for model, info in model_cards.items() if
                          self.inference_engine_classname in info.get("repo", {})]
      return web.json_response(
        {
          "detail": f"Unsupported model: {chat_request.model} with inference engine {self.inference_engine_classname}. Supported models for this engine: {supported_models}"},
        status=400,
      )

    # Fix: Handle the case where shard.model_id might be None
    model_id = getattr(shard, "model_id", None)
    if model_id is None:
      return web.json_response({"detail": "Invalid model configuration"}, status=500)

    repo_id = get_repo(model_id, self.inference_engine_classname)
    if repo_id is None:
      return web.json_response({"detail": "Could not resolve repository for model"}, status=500)

    tokenizer = await resolve_tokenizer(repo_id)
    if DEBUG >= 4: print(f"[ChatGPTAPI] Resolved tokenizer: {tokenizer}")

    # Register the tokenizer with our result manager
    request_id = str(uuid.uuid4())
    self.result_manager.register_tokenizer(request_id, tokenizer)
    self.result_manager.register_model_for_request(request_id, chat_request.model)

    # Add system prompt if set
    if self.system_prompt and not any(msg.role == "system" for msg in chat_request.messages):
      chat_request.messages.insert(0, Message(role="system", content=self.system_prompt))

    prompt = build_prompt(tokenizer, chat_request.messages, chat_request.tools, chat_request.tool_choice)

    if self.on_chat_completion_request:
      try:
        self.on_chat_completion_request(request_id, chat_request, prompt)
      except Exception as e:
        if DEBUG >= 2: traceback.print_exc()

    if DEBUG >= 2: print(f"[ChatGPTAPI] Processing prompt: {request_id=} {shard=} {prompt=}")

    try:
      generation_options = chat_request.to_generation_options(tokenizer)

      # Create tool parser if needed
      api_tool_parser = chat_request.tool_parser(tokenizer)

      # Process the prompt
      await asyncio.wait_for(asyncio.shield(asyncio.create_task(self.node.process_prompt(
        shard,
        prompt,
        request_id=request_id,
        generation_options=generation_options
      ))), timeout=self.response_timeout)

      if DEBUG >= 2: print(f"[ChatGPTAPI] Waiting for response to finish. timeout={self.response_timeout}s")

      if stream:
        response = web.StreamResponse(
          status=200,
          reason="OK",
          headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
          },
        )
        await response.prepare(request)

        try:
          # Use our new async iterator
          async for completion in self.process_inference_stream(request_id, api_tool_parser, self.response_timeout):
            await response.write(self.sse_data_chunk(completion))

          # Send the DONE event when the stream is finished
          await response.write(b"data: [DONE]\n\n")
          await response.write_eof()
          return response

        except asyncio.TimeoutError:
          if DEBUG >= 2: print(f"[ChatGPTAPI] Timeout waiting for token: {request_id=}")
          return web.json_response({"detail": "Response generation timed out"}, status=408)

        except Exception as e:
          if DEBUG >= 2:
            print(f"[ChatGPTAPI] Error processing prompt: {e}")
            traceback.print_exc()

          return web.json_response(
            {"detail": f"Error processing prompt: {str(e)}"},
            status=500
          )
      else:
        # Non-streaming response - use our new method
        completion = await self.get_complete_response(request_id, api_tool_parser, self.response_timeout)
        return web.json_response(completion.model_dump())

    except asyncio.TimeoutError:
      return web.json_response({"detail": "Response generation timed out"}, status=408)
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": f"Error processing prompt (see logs with DEBUG>=2): {str(e)}"}, status=500)

  async def handle_post_chat_token_encode(self, request):
    data = await request.json()
    model = ensure_model(data.get("model"), self.default_model)
    shard = build_base_shard(model, self.inference_engine_classname)
    messages = [Message.model_validate(msg) for msg in data.get("messages", [])]
    tokenizer = await resolve_tokenizer(get_repo(shard.model_id, self.inference_engine_classname))
    prompt = build_prompt(tokenizer, messages, data.get("tools", None))
    tokens = tokenizer.encode(prompt)
    return web.json_response({
      "tokens": len(tokens),
      "truncated": False,  # Not implementing truncation here
    })

  async def process_inference_stream(self,
                                     request_id: str,
                                     api_tool_parser: Optional[ToolParser] = None,
                                     timeout: int = 90) -> AsyncIterator[CompletionObject]:
    """
    Process the inference stream and yield completion objects.

    Args:
        request_id: The unique ID for this request
        api_tool_parser: Optional tool parser for handling tool calls
        timeout: Timeout in seconds

    Yields:
        CompletionObject instances for each chunk of the response
    """
    stream_loc_type: Literal["content", "tool_calls"] = "content"
    tool_call_index: int = -1
    base_completion = CompletionObject.chat_completion(
      request_id,
      self.result_manager.get_model_for_request(request_id),
    )

    i = 0
    tool_parsing_chunk = None

    # Stream results using the inference manager
    async for chunk in self.result_manager.get_inference_result(request_id, timeout=timeout):
      # TODO: This is a hacky buffering process to avoid doing incremental tool parsing, we should replace it with proper
      #   incremental parsing logic in the future.
      if api_tool_parser:
        if i == 0 and chunk.text.startswith(api_tool_parser.start_prefix()):
          tool_parsing_chunk = chunk
          i += 1
          continue

        if tool_parsing_chunk is not None:
          tool_parsing_chunk.text = tool_parsing_chunk.text + chunk.text
          # These will always be False / None for the tool_parsing_chunk at this point
          tool_parsing_chunk.finish_reason = chunk.finish_reason
          tool_parsing_chunk.is_finished = chunk.is_finished

          # If we haven't finished parsing the tool call, we need to wait for the next chunk
          if not tool_parsing_chunk.is_finished:
            continue
          else:
            chunk = tool_parsing_chunk

        decoded = chunk.text

        try:
          decoded, tool_calls = api_tool_parser.parse_tool_calls(decoded)

          if tool_calls is not None:
            for tool_call in tool_calls:
              stream_loc_type = "tool_calls"
              tool_call_index += 1
              tool_call_id = str(uuid.uuid4())

              # When starting a new tool call we emit an initial chunk with the tool call id and name
              completion = base_completion.with_choices([
                StreamingChoice.create_tool_call_choice(
                  tool_call_index,
                  tool_call,
                  tool_call_id
                )
              ])

              yield completion
        except Exception as e:
          if DEBUG >= 2:
            print(f"Error parsing tool calls: {e}")

      # Check if we have content to send
      if decoded != '':
        if stream_loc_type == "content":
          completion = base_completion.with_choice(
            StreamingChoice.create_content_choice(decoded)
          )
        else:
          completion = base_completion.with_choice(
            StreamingChoice.create_tool_call_arguments_choice(
              tool_call_index,
              decoded
            )
          )

        yield completion

      # Handle completion
      if chunk.is_finished:
        completion = base_completion.with_choice(
          StreamingChoice.create_finish_choice(chunk.finish_reason)
        )

        yield completion
        break

  async def get_complete_response(self, request_id: str,
                                  api_tool_parser: Optional[ToolParser] = None,
                                  timeout: int = 90) -> CompletionObject:
    """
    Get the complete response for a non-streaming request.

    Args:
        request_id: The unique ID for this request
        api_tool_parser: Optional tool parser for handling tool calls
        timeout: Timeout in seconds

    Returns:
        A CompletionObject with the complete response
    """
    chunk = await self.result_manager.get_complete_inference_result(request_id, timeout=timeout)
    complete_text = chunk.text
    finish_reason = chunk.finish_reason

    tool_calls = []
    if api_tool_parser:
      try:
        remaining_decoded_content, tool_calls = api_tool_parser.parse_tool_calls(complete_text)
      except Exception as e:
        if DEBUG >= 2:
          print(f"Error parsing tool calls: {e}")
        remaining_decoded_content = complete_text
    else:
      remaining_decoded_content = complete_text

    return CompletionObject.chat_completion(
      request_id,
      self.result_manager.get_model_for_request(request_id),
      [
        NonStreamingChoice.create_choice(remaining_decoded_content, tool_calls, finish_reason, api_tool_parser)
      ]
    )


def ensure_model(model: Optional[str], default_model: str) -> str:
  if not model:
    return default_model

  if model and model.startswith("gpt-"):  # Handle gpt- model requests
    return default_model

  if model not in model_cards:
    if DEBUG >= 1:
      print(f"Invalid model: {model}. Supported: {list(model_cards.keys())}. Defaulting to {default_model}")

    return default_model

  return model
