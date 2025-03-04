import asyncio
import json
import traceback
import uuid
import tempfile
from pathlib import Path
import base64
from io import BytesIO
import numpy as np
from PIL import Image
from aiohttp import web
import platform

from exo import DEBUG
from exo.helpers import get_exo_images_dir
from exo.apputil import create_animation_mp4
from exo.models import build_base_shard
from exo.orchestration import Node

if platform.system().lower() == "darwin" and platform.machine().lower() == "arm64":
    import mlx.core as mx
else:
    import numpy as mx


class ImageApi:
    def __init__(self, inference_engine_classname: str, node: Node):
        self.inference_engine_classname = inference_engine_classname
        self.node = node
        self.images_dir = get_exo_images_dir()
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.response_timeout = 90  # Default timeout value
    
    async def handle_post_image_generations(self, request):
        data = await request.json()

        if DEBUG >= 2: print(f"Handling image generations request from {request.remote}: {data}")
        stream = data.get("stream", False)
        model = data.get("model", "")
        prompt = data.get("prompt", "")
        image_url = data.get("image_url", "")
        if DEBUG >= 2: print(f"model: {model}, prompt: {prompt}, stream: {stream}")
        shard = build_base_shard(model, self.inference_engine_classname)
        if DEBUG >= 2: print(f"shard: {shard}")
        if not shard:
            return web.json_response(
                {"error": f"Unsupported model: {model} with inference engine {self.inference_engine_classname}"}, status=400)

        request_id = str(uuid.uuid4())
        callback_id = f"chatgpt-api-wait-response-{request_id}"
        callback = self.node.on_token.register(callback_id)
        try:
            if image_url != "" and image_url is not None:
                img = self.base64_decode(image_url)
            else:
                img = None
            await asyncio.wait_for(asyncio.shield(asyncio.create_task(
                self.node.process_prompt(shard, prompt, request_id=request_id, inference_state={"image": img}))),
                timeout=self.response_timeout)

            response = web.StreamResponse(status=200, reason='OK', headers={
                'Content-Type': 'application/octet-stream',
                "Cache-Control": "no-cache",
            })
            await response.prepare(request)

            stream_task = None

            async def stream_image(_request_id: str, result, is_finished: bool):
                if isinstance(result, list):
                    await response.write(
                        json.dumps({'progress': self.get_progress_bar((result[0]), (result[1]))}).encode('utf-8') + b'\n')

                elif isinstance(result, np.ndarray):
                    try:
                        im = Image.fromarray(np.array(result))
                        # Save the image to a file
                        image_filename = f"{_request_id}.png"
                        image_path = self.images_dir / image_filename
                        im.save(image_path)

                        # Get URL for the saved image
                        try:
                            base_url = f"{request.scheme}://{request.host}"
                            full_image_url = f"{base_url}/images/{image_filename}"

                            await response.write(
                                json.dumps({'images': [{'url': str(full_image_url), 'content_type': 'image/png'}]}).encode(
                                    'utf-8') + b'\n')
                        except KeyError as e:
                            if DEBUG >= 2: print(f"Error getting image URL: {e}")
                            # Fallback to direct file path if URL generation fails
                            await response.write(
                                json.dumps({'images': [{'url': str(image_path), 'content_type': 'image/png'}]}).encode('utf-8') + b'\n')

                        if is_finished:
                            await response.write_eof()

                    except Exception as e:
                        if DEBUG >= 2: print(f"Error processing image: {e}")
                        if DEBUG >= 2: traceback.print_exc()
                        await response.write(json.dumps({'error': str(e)}).encode('utf-8') + b'\n')

            def on_result(_request_id: str, result, is_finished: bool):
                nonlocal stream_task
                stream_task = asyncio.create_task(stream_image(_request_id, result, is_finished))
                return _request_id == request_id and is_finished

            await callback.wait(on_result, timeout=self.response_timeout * 10)

            if stream_task is not None:
                await stream_task

            return response

        except Exception as e:
            if DEBUG >= 2: traceback.print_exc()
            return web.json_response({"detail": f"Error processing prompt (see logs with DEBUG>=2): {str(e)}"}, status=500)

    async def handle_create_animation(self, request):
        try:
            data = await request.json()
            replacement_image_path = data.get("replacement_image_path")
            device_name = data.get("device_name", "Local Device")
            prompt_text = data.get("prompt", "")

            if DEBUG >= 2: print(
                f"Creating animation with params: replacement_image={replacement_image_path}, device={device_name}, prompt={prompt_text}")

            if not replacement_image_path:
                return web.json_response({"error": "replacement_image_path is required"}, status=400)

            # Create temp directory if it doesn't exist
            tmp_dir = Path(tempfile.gettempdir()) / "exo_animations"
            tmp_dir.mkdir(parents=True, exist_ok=True)

            # Generate unique output filename in temp directory
            output_filename = f"animation_{uuid.uuid4()}.mp4"
            output_path = str(tmp_dir / output_filename)

            if DEBUG >= 2: print(
                f"Animation temp directory: {tmp_dir}, output file: {output_path}, directory exists: {tmp_dir.exists()}, directory permissions: {oct(tmp_dir.stat().st_mode)[-3:]}")

            # Create the animation
            create_animation_mp4(replacement_image_path, output_path, device_name, prompt_text)

            return web.json_response({"status": "success", "output_path": output_path})

        except Exception as e:
            if DEBUG >= 2: traceback.print_exc()
            return web.json_response({"error": str(e)}, status=500)

    def get_progress_bar(self, current_step, total_steps, bar_length=50):
        # Calculate the percentage of completion
        percent = float(current_step) / total_steps
        # Calculate the number of hashes to display
        arrow = '-' * int(round(percent * bar_length) - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        # Create the progress bar string
        progress_bar = f'Progress: [{arrow}{spaces}] {int(percent * 100)}% ({current_step}/{total_steps})'
        return progress_bar

    def base64_decode(self, base64_string):
        # decode and reshape image
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(image_data))
        W, H = (dim - dim % 64 for dim in (img.width, img.height))
        if W != img.width or H != img.height:
            if DEBUG >= 2: print(f"Warning: image shape is not divisible by 64, downsampling to {W}x{H}")
            img = img.resize((W, H), Image.Resampling.NEAREST)  # Fix: use actual enum instead of the deprecated constant
        img = mx.array(np.array(img))
        # Fix the astype issue by using a different approach
        if isinstance(mx, np):
            # For numpy arrays
            img = (img[:, :, :3].astype(np.float32) / 255) * 2 - 1
        else:
            # For mlx arrays
            img = (img[:, :, :3] / 255.0) * 2 - 1
        img = img[None]
        return img 