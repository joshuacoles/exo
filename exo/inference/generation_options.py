from typing import Optional, List


class GenerationOptions:
  max_completion_tokens: Optional[int] = None
  stop: Optional[List[str]] = None

  # Whether to include the stop sequence in the output, by default we do not include it. This does not apply to the stop token.
  include_stop: bool = False

  def __init__(self, max_completion_tokens: Optional[int] = None, stop: Optional[List[str]] = None):
    self.max_completion_tokens = max_completion_tokens
    self.stop = stop
