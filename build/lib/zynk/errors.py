"""
Error Handling Module

Defines exception types and error response formatting for Zynk.
Ensures consistent error propagation from Python to TypeScript.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class BridgeError(Exception):
    """
    Base exception for Zynk errors.

    Attributes:
        code: A machine-readable error code (e.g., "VALIDATION_ERROR").
        message: A human-readable error message.
        details: Optional additional error details.
    """

    def __init__(
        self,
        code: str,
        message: str,
        details: Any | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON response."""
        result = {
            "code": self.code,
            "message": self.message,
        }
        if self.details is not None:
            result["details"] = self.details
        return result


class ValidationError(BridgeError):
    """Raised when request validation fails."""

    def __init__(self, message: str, details: Any | None = None):
        super().__init__("VALIDATION_ERROR", message, details)


class CommandNotFoundError(BridgeError):
    """Raised when a requested command doesn't exist."""

    def __init__(self, command_name: str):
        super().__init__(
            "COMMAND_NOT_FOUND",
            f"Command '{command_name}' not found",
            {"command": command_name},
        )


class CommandExecutionError(BridgeError):
    """Raised when a command execution fails."""

    def __init__(self, message: str, details: Any | None = None):
        super().__init__("EXECUTION_ERROR", message, details)


class ChannelError(BridgeError):
    """Raised when a channel operation fails."""

    def __init__(self, message: str, channel_id: str | None = None):
        super().__init__(
            "CHANNEL_ERROR",
            message,
            {"channel_id": channel_id} if channel_id else None,
        )


class InternalError(BridgeError):
    """Raised for unexpected internal errors."""

    def __init__(self, message: str = "An internal error occurred"):
        super().__init__("INTERNAL_ERROR", message)


# Response models for API

class ErrorResponse(BaseModel):
    """Standard error response format."""
    code: str
    message: str
    details: Any | None = None


class SuccessResponse(BaseModel):
    """Standard success response format."""
    result: Any


class ChannelInitResponse(BaseModel):
    """Response for channel initialization."""
    channel_id: str

    model_config = {"populate_by_name": True}

    def model_dump(self, **kwargs):
        # Override to use camelCase for JSON
        data = super().model_dump(**kwargs)
        return {"channelId": data["channel_id"]}
