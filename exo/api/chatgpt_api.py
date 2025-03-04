import uuid
import time
import asyncio
import json
from pathlib import Path
from typing import List, Literal, Union, Dict, Any, Callable, Optional
from aiohttp import web
import aiohttp_cors
import traceback
import signal
from exo import DEBUG, VERSION
from exo.api.model_api import ModelApi
from exo.api.image_api import ImageApi
from exo.api.response_formats import ResponseFormat
from exo.helpers import PrefixDict, shutdown, get_exo_images_dir
from exo.inference.tokenizers import resolve_tokenizer
from exo.orchestration import Node
from exo.inference.generation_options import GenerationOptions
from exo.models import build_base_shard, model_cards, get_repo, get_model_card
import platform
from exo.tools import AssistantToolCall
from exo.api.inference_result_manager import InferenceResultManager

if platform.system().lower() == "darwin" and platform.machine().lower() == "arm64":
  import mlx.core as mx
else:
  import numpy as mx


class Message:
  def __init__(self, role: str, content: Union[str, List[Dict[str, Union[str, Dict[str, str]]]]],
               tools: Optional[List[Dict]] = None):
    self.role = role
    self.content = content
    self.tools = tools

  def to_dict(self):
    data = {"role": self.role, "content": self.content}
    if self.tools:
      data["tools"] = self.tools
    return data


class ChatCompletionRequest:
  def __init__(self, model: str, messages: List[Message], temperature: float, tools: Optional[List[Dict]] = None,
               max_completion_tokens: Optional[int] = None, stop: Optional[Union[str, List[str]]] = None,
               response_format: Optional[ResponseFormat] = None, tool_choice: Optional[Union[str, Dict]] = None,
               tool_call_format: Optional[str] = None):
    self.model = model
    self.messages = messages
    self.temperature = temperature
    self.tools = tools
    self.max_completion_tokens = max_completion_tokens
    self.stop = stop if isinstance(stop, list) else [stop] if isinstance(stop, str) else None
    self.response_format = response_format
    self.tool_choice = tool_choice
    self.tool_call_format = tool_call_format

  def to_dict(self):
    return {"model": self.model, "messages": [message.to_dict() for message in self.messages],
            "temperature": self.temperature, "tools": self.tools, "max_completion_tokens": self.max_completion_tokens,
            "stop": self.stop, "response_format": self.response_format, "tool_choice": self.tool_choice}

  def to_generation_options(self) -> GenerationOptions:
    # Determine grammar - either from response_format or function calling format
    grammar = None
    if self.response_format:
      grammar = self.response_format.to_grammar()

    return GenerationOptions(
      max_completion_tokens=self.max_completion_tokens,
      stop=self.stop,
      grammar_definition=grammar,
      temperature=self.temperature,
      tools=self.tools,
      tool_choice=self.tool_choice,
      tool_call_format=self.tool_call_format
    )


def completion_wrapper(
  request_id: str,
  object_type: Literal["chat.completion", "text_completion"],
  model: str,
  choices: List[dict[str, Any]]) -> dict:
  return {
    "id": f"chatcmpl-{request_id}",
    "object": object_type,
    "created": int(time.time()),
    "model": model,
    "system_fingerprint": f"exo_{VERSION}",
    "choices": choices,
  }


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


def build_prompt(tokenizer, _messages: List[Message], tools: Optional[List[Dict]] = None,
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


def parse_message(data: dict):
  if "role" not in data or "content" not in data:
    raise ValueError(f"Invalid message: {data}. Must have 'role' and 'content'")
  return Message(data["role"], data["content"], data.get("tools"))


def parse_chat_request(data: dict, default_model: str):
  # Parse response_format if provided
  response_format = None
  if "response_format" in data:
    response_format = ResponseFormat.parse_from_request(data["response_format"])

  # Get tool_choice parameter
  tool_choice = data.get("tool_choice", None)

  model = data.get("model", default_model)

  # To be compatible with ChatGPT tools, point all gpt- model requests to default model
  if model and model.startswith("gpt-"):
    model = default_model

  if not model or model not in model_cards:
    if DEBUG >= 1: print(
      f"[ChatGPTAPI] Invalid model: {model}. Supported: {list(model_cards.keys())}. Defaulting to {default_model}")
    model = default_model

  # `max_tokens` is deprecated, but some clients may still use it, fall back to that value if max_completion_tokens is not provided.
  max_completion_tokens = data.get("max_completion_tokens", data.get("max_tokens", None))

  return ChatCompletionRequest(
    model,
    [parse_message(msg) for msg in data["messages"]],
    data.get("temperature", 0.0),
    data.get("tools", None),
    max_completion_tokens,
    data.get("stop", None),
    response_format,
    tool_choice,
    tool_call_format=data.get("tool_call_format", get_model_card(model).get("default_tool_call_format", None))
  )


class PromptSession:
  def __init__(self, request_id: str, timestamp: int, prompt: str):
    self.request_id = request_id
    self.timestamp = timestamp
    self.prompt = prompt


class ChatGPTAPI:
  def __init__(
    self,
    node: Node,
    inference_engine_classname: str,
    response_timeout: int = 90,
    on_chat_completion_request: Callable[[str, ChatCompletionRequest, str], None] = None,
    default_model: Optional[str] = None,
    system_prompt: Optional[str] = None
  ):
    self.node = node
    self.inference_engine_classname = inference_engine_classname
    self.response_timeout = response_timeout
    self.on_chat_completion_request = on_chat_completion_request
    self.app = web.Application(client_max_size=100 * 1024 * 1024)  # 100MB to support image upload
    self.prompts: PrefixDict[str, PromptSession] = PrefixDict()
    self.prev_token_lens: Dict[str, int] = {}
    self.stream_tasks: Dict[str, asyncio.Task] = {}
    self.default_model = default_model or "llama-3.2-1b"

    # Initialize the inference result manager
    self.result_manager = InferenceResultManager(node)

    self.system_prompt = system_prompt
    self.models_api = ModelApi(self.inference_engine_classname, self.node)
    self.image_api = ImageApi(self.inference_engine_classname, self.node)
    self.image_api.response_timeout = self.response_timeout

    cors = aiohttp_cors.setup(self.app)
    cors_options = aiohttp_cors.ResourceOptions(
      allow_credentials=True,
      expose_headers="*",
      allow_headers="*",
      allow_methods="*",
    )
    cors.add(self.app.router.add_get("/models", self.models_api.handle_get_models), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/models", self.models_api.handle_get_models), {"*": cors_options})
    cors.add(self.app.router.add_post("/chat/token/encode", self.handle_post_chat_token_encode), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/chat/token/encode", self.handle_post_chat_token_encode), {"*": cors_options})
    cors.add(self.app.router.add_post("/chat/completions", self.handle_post_chat_completions), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/chat/completions", self.handle_post_chat_completions), {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/image/generations", self.image_api.handle_post_image_generations), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/download/progress", self.handle_get_download_progress), {"*": cors_options})
    cors.add(self.app.router.add_get("/modelpool", self.models_api.handle_model_support), {"*": cors_options})
    cors.add(self.app.router.add_get("/healthcheck", self.handle_healthcheck), {"*": cors_options})
    cors.add(self.app.router.add_post("/quit", self.handle_quit), {"*": cors_options})
    cors.add(self.app.router.add_delete("/models/{model_name}", self.models_api.handle_delete_model),
             {"*": cors_options})
    cors.add(self.app.router.add_get("/initial_models", self.models_api.handle_get_initial_models), {"*": cors_options})
    cors.add(self.app.router.add_post("/create_animation", self.image_api.handle_create_animation), {"*": cors_options})
    cors.add(self.app.router.add_post("/download", self.models_api.handle_post_download), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/topology", self.handle_get_topology), {"*": cors_options})
    cors.add(self.app.router.add_get("/topology", self.handle_get_topology), {"*": cors_options})

    # Add static routes
    if "__compiled__" not in globals():
      self.static_dir = Path(__file__).parent.parent / "tinychat"
      self.app.router.add_get("/", self.handle_root)
      self.app.router.add_static("/", self.static_dir, name="static")

    # Always add images route, regardless of compilation status
    self.images_dir = get_exo_images_dir()
    self.images_dir.mkdir(parents=True, exist_ok=True)
    self.app.router.add_static('/images/', self.images_dir, name='static_images')

    self.app.middlewares.append(self._timeout_middleware)
    self.app.middlewares.append(self._log_request)

  async def _timeout_middleware(self, request, handler):
    try:
      return await asyncio.wait_for(handler(request), timeout=self.response_timeout)
    except asyncio.TimeoutError:
      return web.json_response({"detail": "Request timed out"}, status=408)

  async def _log_request(self, request, handler):
    if DEBUG >= 2: print(f"Received request: {request.method} {request.path}")
    return await handler(request)

  async def handle_quit(self, request):
    if DEBUG >= 1: print("Received quit signal")
    response = web.json_response({"detail": "Quit signal received"}, status=200)
    await response.prepare(request)
    await response.write_eof()
    # Schedule the shutdown to happen after the response is sent
    asyncio.create_task(shutdown(signal.SIGINT, asyncio.get_event_loop(), self.node.server))
    return response

  async def handle_root(self, request):
    return web.FileResponse(self.static_dir / "index.html")

  async def handle_healthcheck(self, request):
    return web.json_response({"status": "ok"})

  async def handle_post_chat_token_encode(self, request):
    data = await request.json()
    model = data.get("model", self.default_model)
    if model and model.startswith("gpt-"):  # Handle gpt- model requests
      model = self.default_model
    if not model or model not in model_cards:
      if DEBUG >= 1: print(
        f"Invalid model: {model}. Supported: {list(model_cards.keys())}. Defaulting to {self.default_model}")
      model = self.default_model
    shard = build_base_shard(model, self.inference_engine_classname)
    messages = [parse_message(msg) for msg in data.get("messages", [])]
    tokenizer = await resolve_tokenizer(get_repo(shard.model_id, self.inference_engine_classname))
    prompt = build_prompt(tokenizer, messages, data.get("tools", None))
    tokens = tokenizer.encode(prompt)
    return web.json_response({
      "length": len(prompt),
      "num_tokens": len(tokens),
      "encoded_tokens": tokens,
      "encoded_prompt": prompt,
    })

  async def handle_get_download_progress(self, request):
    return await self.models_api.handle_get_download_progress(request)

  def sse_data_chunk(self, json_data: dict):
    return f"data: {json.dumps(json_data)}\n\n".encode()

  async def handle_post_chat_completions(self, request):
    data = await request.json()
    if DEBUG >= 2: print(f"[ChatGPTAPI] Handling chat completions request from {request.remote}: {data}")
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

    tokenizer = await resolve_tokenizer(get_repo(shard.model_id, self.inference_engine_classname))
    if DEBUG >= 4: print(f"[ChatGPTAPI] Resolved tokenizer: {tokenizer}")

    # Register the tokenizer with our result manager
    request_id = str(uuid.uuid4())
    self.result_manager.register_tokenizer(request_id, tokenizer)

    # Add system prompt if set
    if self.system_prompt and not any(msg.role == "system" for msg in chat_request.messages):
      chat_request.messages.insert(0, Message("system", self.system_prompt))

    prompt = build_prompt(tokenizer, chat_request.messages, chat_request.tools, chat_request.tool_choice)

    if self.on_chat_completion_request:
      try:
        self.on_chat_completion_request(request_id, chat_request, prompt)
      except Exception as e:
        if DEBUG >= 2: traceback.print_exc()

    if DEBUG >= 2: print(f"[ChatGPTAPI] Processing prompt: {request_id=} {shard=} {prompt=}")

    try:
      generation_options = chat_request.to_generation_options()
      api_tool_parser = generation_options.tool_parser(tokenizer)

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
          stream_loc_type: Literal["content", "tool_calls"] = "content"
          tool_call_index: int = -1

          # Stream results using the inference manager
          async for chunk in self.result_manager.get_inference_result(request_id, timeout=self.response_timeout):
            decoded = chunk.text

            if api_tool_parser:
              decoded, tool_calls = api_tool_parser.parse_tool_calls(decoded)

              if tool_calls is not None:
                for tool_call in tool_calls:
                  stream_loc_type = "tool_calls"
                  tool_call_index += 1
                  tool_call_id = str(uuid.uuid4())

                  # When starting a new tool call we emit an initial chunk with the tool call id and name that will later be updated with the arguments
                  # streamed in subsequent chunks.
                  completion = completion_wrapper(
                    request_id,
                    "chat.completion",
                    chat_request.model,
                    [{
                      "index": tool_call_index,
                      "logprobs": None,
                      "finish_reason": None,
                      "delta": {
                        "tool_calls": [
                          {
                            "index": tool_call_index,
                            "id": tool_call_id,
                            "type": "function",
                            "function": tool_call.model_dump()
                          }
                        ]
                      },
                    }]
                  )

                  await response.write(self.sse_data_chunk(completion))

            # Check if we have content to send
            if decoded != '':
              if stream_loc_type == "content":
                completion = completion_wrapper(
                  request_id,
                  "chat.completion",
                  chat_request.model,
                  [{
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": None,
                    "delta": {
                      "role": "assistant",
                      "content": decoded
                    }
                  }]
                )
              else:
                completion = completion_wrapper(
                  request_id,
                  "chat.completion",
                  chat_request.model,
                  [{
                    "index": tool_call_index,
                    "logprobs": None,
                    "finish_reason": None,
                    "delta": {
                      "tool_calls": [
                        {
                          "index": tool_call_index,
                          "function": {"arguments": decoded}
                        }
                      ]
                    },
                  }]
                )

              await response.write(self.sse_data_chunk(completion))

            # Handle completion
            if chunk.is_finished:
              completion = completion_wrapper(
                request_id,
                "chat.completion",
                chat_request.model,
                [{
                  "index": 0,
                  "logprobs": None,
                  "finish_reason": chunk.finish_reason,
                  "delta": {}
                }]
              )

              await response.write(self.sse_data_chunk(completion))
              break

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
        # Non-streaming response
        chunk = await self.result_manager.get_complete_inference_result(request_id)
        complete_text = chunk.text
        finish_reason = chunk.finish_reason

        if api_tool_parser:
          remaining_decoded_content, tool_calls = api_tool_parser.parse_tool_calls(complete_text)
        else:
          remaining_decoded_content = complete_text
          tool_calls = []

        return web.json_response(completion_wrapper(
          request_id,
          "chat.completion",
          chat_request.model,
          [{
            "index": 0,
            "logprobs": None,
            "finish_reason": finish_reason,
            "message": {
              "role": "assistant",
              "content": remaining_decoded_content if not tool_calls else None,
              "tool_calls": [AssistantToolCall(
                id=str(uuid.uuid4()),
                type="function",
                function=tool_call,
              ).model_dump() for tool_call in tool_calls] if api_tool_parser else None,
            }
          }]
        ))
    except asyncio.TimeoutError:
      return web.json_response({"detail": "Response generation timed out"}, status=408)
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": f"Error processing prompt (see logs with DEBUG>=2): {str(e)}"}, status=500)

  async def handle_get_topology(self, request):
    try:
      topology = self.node.current_topology
      if topology:
        return web.json_response(topology.to_json())
      else:
        return web.json_response({})
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": f"Error getting topology: {str(e)}"}, status=500)

  async def run(self, host: str = "0.0.0.0", port: int = 52415):
    runner = web.AppRunner(self.app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
