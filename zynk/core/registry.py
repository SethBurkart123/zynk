"""Registry backed by the Zynk API graph IR."""

from __future__ import annotations

import types
from enum import Enum
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin

from pydantic import BaseModel

from .ir import ApiGraph, Endpoint
from .types import model_def

if TYPE_CHECKING:
    from zynk.registry import CommandInfo
    from zynk.static import StaticInfo
    from zynk.upload import UploadInfo
    from zynk.websocket import MessageHandlerInfo


class ApiRegistry:
    _instance: ApiRegistry | None = None

    def __new__(cls) -> ApiRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.graph = ApiGraph()
            cls._instance._commands: dict[str, CommandInfo] = {}
            cls._instance._message_handlers: dict[str, MessageHandlerInfo] = {}
            cls._instance._uploads: dict[str, UploadInfo] = {}
            cls._instance._statics: dict[str, StaticInfo] = {}
        return cls._instance

    @classmethod
    def get_instance(cls) -> ApiRegistry:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        if cls._instance is None:
            return
        cls._instance.graph = ApiGraph()
        cls._instance._commands.clear()
        cls._instance._message_handlers.clear()
        cls._instance._uploads.clear()
        cls._instance._statics.clear()

    def register_endpoint(self, endpoint: Endpoint) -> None:
        if endpoint.name in self.graph.endpoints:
            existing = self.graph.endpoints[endpoint.name]
            raise ValueError(
                f"Endpoint name conflict: '{endpoint.name}' is defined in both "
                f"'{existing.module}' and '{endpoint.module}'. Names must be unique."
            )
        self.graph.endpoints[endpoint.name] = endpoint

    def register_model(self, model: type[BaseModel]) -> None:
        if model.__name__ not in self.graph.models:
            self.graph.models[model.__name__] = model_def(model)

    def collect_models_from_type(self, type_hint: Any) -> None:
        if type_hint is None:
            return
        origin = get_origin(type_hint)
        if origin is Union or origin is types.UnionType:
            for arg in get_args(type_hint):
                if arg is not type(None):
                    self.collect_models_from_type(arg)
            return
        if origin is not None:
            for arg in get_args(type_hint):
                self.collect_models_from_type(arg)
            return
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            self.register_model(type_hint)
            for field_info in type_hint.model_fields.values():
                self.collect_models_from_type(field_info.annotation)
        elif isinstance(type_hint, type) and issubclass(type_hint, Enum):
            if type_hint.__name__ not in self.graph.enums:
                self.graph.enums[type_hint.__name__] = type_hint

    def get_graph(self) -> ApiGraph:
        return self.graph
