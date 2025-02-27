from typing import Literal, Optional, Any
import json

from pydantic import BaseModel

from exo.inference.grammars import JSON_LARK_GRAMMAR


class ResponseFormat(BaseModel):
  type: Literal["text", "json_object", "json_schema"]

  def to_grammar(self) -> Optional[str]:
    raise NotImplementedError()

  def is_guided(self):
    """
    If the response format requires guided generation. By default, this is true. If this returns true you must return
    a grammar from to_grammar.
    """

    return True

  @staticmethod
  def parse_from_request(obj: dict):
    if obj["type"] == "text":
      return TextResponseFormat.model_validate(obj)
    elif obj["type"] == "json_object":
      return JsonObjectResponseFormat.model_validate(obj)
    elif obj["type"] == "json_schema":
      return JsonSchemaResponseFormat.model_validate(obj)
    elif obj["type"] == "lark_grammar":
      return LarkGrammarResponseFormat.model_validate(obj)
    elif obj["type"] == "regex":
      return RegexResponseFormat.model_validate(obj)
    else:
      raise ValueError(f"Unknown response format type: {obj['type']}")


class TextResponseFormat(ResponseFormat):
  type: Literal["text"]

  def is_guided(self):
    return False

  def to_grammar(self) -> Optional[str]:
    return None


class JsonObjectResponseFormat(BaseModel):
  type: Literal["json_object"]

  def to_grammar(self) -> Optional[str]:
    return json.dumps({
      "grammars": [{"lark_grammar": JSON_LARK_GRAMMAR}]
    })


class JsonSchemaResponseFormat(BaseModel):
  type: Literal["json_schema"]
  json_schema: Any

  def to_grammar(self) -> Optional[str]:
    return json.dumps({
      "grammars": [{"json_schema": self.json_schema}]
    })


# Aligns with https://github.com/guidance-ai/llgtrt
class LarkGrammarResponseFormat(BaseModel):
  type: Literal["lark_grammar"]
  lark_grammar: str

  def to_grammar(self) -> Optional[str]:
    return json.dumps({
      "grammars": [{"lark_grammar": self.lark_grammar}]
    })


class RegexResponseFormat(BaseModel):
  type: Literal["regex"]
  regex: str

  def to_grammar(self) -> Optional[str]:
    return json.dumps({
      "grammars": [{"lark_grammar": f"start: /{self.regex}/"}]
    })
