from abc import ABC, abstractmethod
from typing import Any


class ToolParser(ABC):
    @abstractmethod
    def parse(self, tool_name: str, tool_call: dict) -> Any:
        ...


def tool_parser_by_name(name: str) -> ToolParser:
    ...