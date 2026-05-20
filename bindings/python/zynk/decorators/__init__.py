"""Public decorator entry points."""

from zynk.registry import command, message
from zynk.static import static
from zynk.upload import upload

__all__ = ["command", "message", "static", "upload"]
