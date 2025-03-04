import asyncio
from pathlib import Path
from typing import List, Literal, Union, Dict, Any, Callable, Optional
from aiohttp import web
import aiohttp_cors
import traceback
import signal
from exo import DEBUG, VERSION
from exo.api.model_api import ModelApi
from exo.api.image_api import ImageApi
from exo.helpers import shutdown, get_exo_images_dir
from exo.orchestration import Node
from exo.api.inference_result_manager import InferenceResultManager
from exo.api.chat_api import ChatApi, ChatCompletionRequest


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
    self.app = web.Application(client_max_size=100 * 1024 * 1024)  # 100MB to support image upload

    # Initialize the inference result manager
    self.result_manager = InferenceResultManager(node)

    self.models_api = ModelApi(self.inference_engine_classname, self.node)
    self.image_api = ImageApi(self.inference_engine_classname, self.node)
    self.image_api.response_timeout = self.response_timeout

    # Initialize the chat API
    self.chat_api = ChatApi(
      self.inference_engine_classname,
      self.node,
      self.result_manager,
      default_model,
      system_prompt,
      self.response_timeout,
      on_chat_completion_request
    )

    cors = aiohttp_cors.setup(self.app)
    cors_options = aiohttp_cors.ResourceOptions(
      allow_credentials=True,
      expose_headers="*",
      allow_headers="*",
      allow_methods="*",
    )
    cors.add(self.app.router.add_get("/models", self.models_api.handle_get_models), {"*": cors_options})
    cors.add(self.app.router.add_get("/v1/models", self.models_api.handle_get_models), {"*": cors_options})
    cors.add(self.app.router.add_post("/chat/token/encode", self.chat_api.handle_post_chat_token_encode),
             {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/chat/token/encode", self.chat_api.handle_post_chat_token_encode),
             {"*": cors_options})
    cors.add(self.app.router.add_post("/chat/completions", self.chat_api.handle_post_chat_completions),
             {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/chat/completions", self.chat_api.handle_post_chat_completions),
             {"*": cors_options})
    cors.add(self.app.router.add_post("/v1/image/generations", self.image_api.handle_post_image_generations),
             {"*": cors_options})
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

  async def _timeout_middleware(self, app, handler):
    async def middleware(request):
      try:
        return await asyncio.wait_for(handler(request), timeout=self.response_timeout)
      except asyncio.TimeoutError:
        return web.json_response({"detail": "Request timed out"}, status=408)

    return middleware

  async def _log_request(self, app, handler):
    async def middleware(request):
      if DEBUG >= 2: print(f"Received request: {request.method} {request.path}")
      return await handler(request)

    return middleware

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

  async def handle_get_download_progress(self, request):
    return await self.models_api.handle_get_download_progress(request)

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
