from typing import List, Tuple, Optional

from llguidance import LLInterpreter
from llguidance.hf import from_tokenizer as llg_from_tokenizer
import numpy as np

from exo import DEBUG
from exo.inference.tool_calling import ToolParser


class BufferedOutput:
  stop_sequences: List[str]
  max_tokens: int
  eos_token_id: int
  stop_seq_buffer_char_size: int

  _token_count: int = 0
  buffer: List[Tuple[int, str]]

  is_finished: bool = False
  finish_reason: Optional[str] = None

  # Grammar for full output structural generation
  guidance_interpreter: Optional[LLInterpreter] = None

  # Guided generation for tool calls.
  tool_parser: Optional[ToolParser] = None
  tool_mode: bool = False
  tool_guidance_interpreter: Optional[LLInterpreter] = None

  def __init__(
    self,
    max_tokens: int,
    eos_token_id: int,
    stop_sequences: List[str],
    tokenizer,
    grammar_definition: Optional[str] = None,
    tool_parser: Optional[ToolParser] = None,
  ):
    self.buffer = []
    self.stop_seq_buffer_char_size = max(len(stop_sequence) for stop_sequence in stop_sequences) if len(
      stop_sequences) > 0 else 0
    self.max_tokens = max_tokens
    self.eos_token_id = eos_token_id
    self.stop_sequences = stop_sequences
    self.tokenizer = tokenizer

    if grammar_definition and tool_parser:
      raise ValueError("Cannot specify both grammar_definition and tool_parser")

    # If we are generating structured responses initialize the guidance
    if grammar_definition:
      print(f"Initializing guidance with grammar definition {grammar_definition}")
      self.initialize_guidance(grammar_definition)
    elif tool_parser:
      print(f"BufferedOutput with tool parser {tool_parser}")
      self.tool_parser = tool_parser

      if tool_parser.is_immediate():
        # This will have the effect of immediately entering tool mode.
        # TODO: Is there a better way to handle this?
        # TODO: Can this be handled by FF_tokens and a whole output grammar? Do we want to merge the two?
        # TODO: Should this be handled as a total output grammar?
        self.append(tool_parser.start_token())

  def initialize_guidance(self, grammar_definition: str):
    self.guidance_interpreter = LLInterpreter(
      llg_from_tokenizer(self.tokenizer, n_vocab=self.tokenizer.vocab_size),
      grammar_definition,
      # TODO: Try enable these, I think this will involve linking out state machine into theirs more
      enable_ff_tokens=False,
      enable_backtrack=False,
      log_level=2
    )

    self.guidance_interpreter.start_without_prompt()

  def enter_tool_mode(self):
    self.tool_mode = True
    self.tool_guidance_interpreter = LLInterpreter(
      llg_from_tokenizer(self.tokenizer, n_vocab=self.tokenizer.vocab_size),
      self.tool_parser.tool_grammar(),
      enable_ff_tokens=False,
      enable_backtrack=False,
      log_level=2
    )

    self.tool_guidance_interpreter.start_without_prompt()
    self.tool_guidance_interpreter.commit_token(self.tool_parser.start_token())

  def append(self, token: int):
    # Guidance interpreter is used for whole output structural generation
    if self.guidance_interpreter:
      valid = self.guidance_interpreter.commit_token(token)
      if not valid:
        raise ValueError(f"Schema violation at token {token} ('{self.tokenizer.decode([token])}')")
    elif self.tool_mode:
      valid = self.tool_guidance_interpreter.commit_token(token)
      if not valid:
        raise ValueError(f"Schema violation at token {token} ('{self.tokenizer.decode([token])}')")

    self.buffer.append((token, self.tokenizer.decode([token])))
    self._token_count += 1

    # If we are in tool mode and the tool guidance interpreter has a pending stop, we can finish
    if self.tool_mode and self.tool_guidance_interpreter.has_pending_stop():
      self.is_finished = True
      self.finish_reason = "stop"
      return

    if token == self.eos_token_id:
      self.is_finished = True
      self.finish_reason = "stop"
    elif self._token_count >= self.max_tokens:
      self.is_finished = True
      self.finish_reason = "length"
    elif self.tool_parser and token == self.tool_parser.start_token():
      self.enter_tool_mode()
    elif self.guidance_interpreter and self.guidance_interpreter.has_pending_stop():
      # TODO: We should handle the different stop reasons
      self.is_finished = True
      self.finish_reason = "stop"
    elif len(self.stop_sequences) > 0:
      self.attempt_to_match_stop_sequences()

  def assembled_text(self) -> str:
    return "".join([text for _, text in self.buffer])

  def attempt_to_match_stop_sequences(self):
    assembled_text = self.assembled_text()
    if DEBUG >= 2: print(f"Attempting to match stop sequences against: {assembled_text=}")

    for stop_sequence in self.stop_sequences:
      if len(assembled_text) < len(stop_sequence):
        continue

      if DEBUG >= 2: print(f"Checking if {assembled_text=} matches {stop_sequence=}")

      if stop_sequence in assembled_text:
        if DEBUG >= 2: print(f"Match found: {assembled_text=} matches {stop_sequence=}")

        # Find character index where stop sequence starts
        char_idx = assembled_text.index(stop_sequence)

        # Find which token contains this character index and where in that token the sequence starts
        current_char_pos = 0
        tokens_to_keep = 0
        for _, text in self.buffer:
          next_char_pos = current_char_pos + len(text)
          if current_char_pos <= char_idx < next_char_pos:
            # Found the token containing the stop sequence
            # Get the text before the stop sequence
            token_offset = char_idx - current_char_pos
            truncated_text = text[:token_offset]

            # This is a little bit of a hack as the SendResults GRPC call expects tokens so to return the truncated text
            # we need to retokenize it. This is not ideal as it means we are not returning the exact tokens that were
            # generated by the model. However, it is the simplest way to handle this case.

            # Retokenize the truncated text
            new_tokens = self.tokenizer.encode(truncated_text, add_special_tokens=False)

            # Replace the final token with the retokenized truncated text
            self.buffer = self.buffer[:tokens_to_keep]
            for token in new_tokens:
              self.buffer.append((token, self.tokenizer.decode([token])))
            break

          current_char_pos = next_char_pos
          tokens_to_keep += 1
        else:
          # If we didn't find the token, just keep everything up to char_idx
          self.buffer = self.buffer[:tokens_to_keep]

        self.is_finished = True
        self.finish_reason = "stop"
        break

  def token_count(self) -> int:
    return self._token_count

  def next_tokens(self) -> List[int]:
    # Simplification: The only emission that happens in tool call mode is at this point.
    # This does not allow for tool streaming but greatly simplifies the code involved
    if self.is_finished:
      # Return all remaining tokens if finished
      tokens = [token for token, _ in self.buffer]
      self.buffer = []
      return tokens

    # In non-tool mode we need to check if the stop sequence buffer is satisfied
    if not self.tool_mode:
      stop_buffer_satisfied = len(self.assembled_text()) >= self.stop_seq_buffer_char_size

      # If so return the oldest token in the buffer
      if stop_buffer_satisfied:
        token, _ = self.buffer.pop(0)
        return [token]

    # Not enough tokens yet
    return []

  def get_token_mask(self) -> Optional[np.ndarray]:
    if self.tool_mode:
      mask, _ = self.tool_guidance_interpreter.compute_mask()
    elif self.guidance_interpreter:
      mask, _ = self.guidance_interpreter.compute_mask()
    else:
      return None

    if mask is None:
      return None

    return np.array(list(mask), dtype="int32")
