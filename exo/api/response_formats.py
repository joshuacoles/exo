from typing import Literal, Optional, Any
import json

from pydantic import BaseModel, TypeAdapter
from abc import ABC
from exo.inference.grammars import JSON_LARK_GRAMMAR


class ResponseFormat(ABC, BaseModel):
  type: str

  def to_grammar(self) -> Optional[str]:
    raise NotImplementedError()


class TextResponseFormat(ResponseFormat):
  type: Literal["text"]

  def to_grammar(self) -> Optional[str]:
    return None


class JsonObjectResponseFormat(ResponseFormat):
  type: Literal["json_object"]

  def to_grammar(self) -> Optional[str]:
    return json.dumps({
      "grammars": [{"lark_grammar": JSON_LARK_GRAMMAR}]
    })


class JsonSchemaResponseFormat(ResponseFormat):
  type: Literal["json_schema"]
  json_schema: Any

  def to_grammar(self) -> Optional[str]:
    return json.dumps({
      "grammars": [{"json_schema": self.json_schema}]
    })


# Aligns with https://github.com/guidance-ai/llgtrt
class LarkGrammarResponseFormat(ResponseFormat):
  type: Literal["lark_grammar"]
  lark_grammar: str

  def to_grammar(self) -> Optional[str]:
    return json.dumps({
      "grammars": [{"lark_grammar": self.lark_grammar}]
    })


class RegexResponseFormat(ResponseFormat):
  type: Literal["regex"]
  regex: str

  def to_grammar(self) -> Optional[str]:
    return json.dumps({
      "grammars": [{"lark_grammar": f"start: /{self.regex}/"}]
    })

ResponseFormatUnion = (TextResponseFormat |
  JsonObjectResponseFormat | JsonSchemaResponseFormat |
  LarkGrammarResponseFormat | RegexResponseFormat)

ResponseFormatAdapter = TypeAdapter(ResponseFormatUnion)
