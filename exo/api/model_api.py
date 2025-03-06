import asyncio
import json
import traceback

from exo import DEBUG
from exo.download.download_progress import RepoProgressEvent
from exo.download.new_shard_download import delete_model
from exo.models import model_cards, get_supported_models, get_pretty_name, build_full_shard
from aiohttp import web

from exo.orchestration import Node


class ModelApi:
  inference_engine_classname: str
  node: Node

  def __init__(self, inference_engine_classname: str, node: Node):
    self.inference_engine_classname = inference_engine_classname
    self.node = node

  async def handle_get_models(self, request):
    models_list = [
      {"id": model_name, "object": "model", "owned_by": "exo", "ready": True}
      for model_name, _ in model_cards.items()
    ]

    return web.json_response({"object": "list", "data": models_list})

  async def handle_model_support(self, request):
    try:
      response = web.StreamResponse(status=200, reason='OK',
                                    headers={'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache',
                                             'Connection': 'keep-alive'})
      await response.prepare(request)
      async for path, s in self.node.shard_downloader.get_shard_download_status(self.inference_engine_classname):
        model_data = {s.shard.model_id: {"downloaded": s.downloaded_bytes == s.total_bytes,
                                         "download_percentage": 100 if s.downloaded_bytes == s.total_bytes else 100 * float(
                                           s.downloaded_bytes) / float(s.total_bytes), "total_size": s.total_bytes,
                                         "total_downloaded": s.downloaded_bytes}}
        await response.write(f"data: {json.dumps(model_data)}\n\n".encode())
      await response.write(b"data: [DONE]\n\n")
      return response

    except Exception as e:
      print(f"Error in handle_model_support: {str(e)}")
      traceback.print_exc()
      return web.json_response({"detail": f"Server error: {str(e)}"}, status=500)

  async def handle_get_download_progress(self, request):
    progress_data = {}
    for node_id, progress_event in self.node.node_download_progress.items():
      if isinstance(progress_event, RepoProgressEvent):
        if progress_event.status != "in_progress": continue
        progress_data[node_id] = progress_event.to_dict()
      else:
        print(f"Unknown progress event type: {type(progress_event)}. {progress_event}")
    return web.json_response(progress_data)

  async def handle_delete_model(self, request):
    model_id = request.match_info.get('model_name')

    try:
      if await delete_model(model_id, self.inference_engine_classname):
        return web.json_response({"status": "success", "message": f"Model {model_id} deleted successfully"})
      else:
        return web.json_response({"detail": f"Model {model_id} files not found"}, status=404)
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"detail": f"Error deleting model: {str(e)}"}, status=500)

  async def handle_get_initial_models(self, request):
    model_data = {}
    for model_id in get_supported_models([[self.inference_engine_classname]]):
      model_data[model_id] = {
        "name": get_pretty_name(model_id),
        "downloaded": None,  # Initially unknown
        "download_percentage": None,  # Change from 0 to null
        "total_size": None,
        "total_downloaded": None,
        "loading": True  # Add loading state
      }
    return web.json_response(model_data)

  async def handle_post_download(self, request):
    try:
      data = await request.json()
      model_name = data.get("model")
      if not model_name: return web.json_response({"error": "model parameter is required"}, status=400)
      if model_name not in model_cards: return web.json_response(
        {"error": f"Invalid model: {model_name}. Supported models: {list(model_cards.keys())}"}, status=400)
      shard = build_full_shard(model_name, self.inference_engine_classname)
      if not shard: return web.json_response({"error": f"Could not build shard for model {model_name}"}, status=400)
      asyncio.create_task(self.node.inference_engine.shard_downloader.ensure_shard(
        shard, self.inference_engine_classname
      ))

      return web.json_response({"status": "success", "message": f"Download started for model: {model_name}"})
    except Exception as e:
      if DEBUG >= 2: traceback.print_exc()
      return web.json_response({"error": str(e)}, status=500)
