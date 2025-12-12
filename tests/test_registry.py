"""
Tests for the Zynk registry module.
"""


import pytest
from pydantic import BaseModel

from zynk.registry import CommandRegistry, command, get_registry


class TestUser(BaseModel):
    """Test user model."""
    __test__ = False
    id: int
    name: str


class TestPost(BaseModel):
    """Test post model with nested user."""
    __test__ = False
    id: int
    title: str
    author: TestUser


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the registry before each test."""
    CommandRegistry.reset()
    yield
    CommandRegistry.reset()


def test_command_registration():
    """Test basic command registration."""
    @command
    async def test_func(x: int) -> str:
        return str(x)

    registry = get_registry()
    cmd = registry.get_command("test_func")

    assert cmd is not None
    assert cmd.name == "test_func"
    assert "x" in cmd.params
    assert cmd.params["x"] is int
    assert cmd.return_type is str
    assert cmd.is_async is True


def test_command_with_custom_name():
    """Test command registration with custom name."""
    @command(name="custom_name")
    async def internal_func() -> str:
        return "hello"

    registry = get_registry()

    assert registry.get_command("internal_func") is None
    assert registry.get_command("custom_name") is not None


def test_pydantic_model_registration():
    """Test that Pydantic models are automatically registered."""
    @command
    async def get_user(user_id: int) -> TestUser:
        return TestUser(id=user_id, name="Test")

    registry = get_registry()
    models = registry.get_all_models()

    assert "TestUser" in models
    assert models["TestUser"] == TestUser


def test_nested_model_registration():
    """Test that nested Pydantic models are registered."""
    @command
    async def get_post(post_id: int) -> TestPost:
        return TestPost(
            id=post_id,
            title="Test",
            author=TestUser(id=1, name="Author")
        )

    registry = get_registry()
    models = registry.get_all_models()

    assert "TestPost" in models
    assert "TestUser" in models


def test_list_return_type():
    """Test command with list return type."""
    @command
    async def list_users() -> list[TestUser]:
        return []

    registry = get_registry()
    cmd = registry.get_command("list_users")

    assert cmd is not None
    # Model should be registered from List[TestUser]
    models = registry.get_all_models()
    assert "TestUser" in models


def test_optional_return_type():
    """Test command with optional return type."""
    @command
    async def find_user(user_id: int) -> TestUser | None:
        return None

    registry = get_registry()
    cmd = registry.get_command("find_user")

    assert cmd is not None
    models = registry.get_all_models()
    assert "TestUser" in models


def test_duplicate_command_raises_error():
    """Test that duplicate command names raise an error."""
    @command
    async def duplicate_name() -> str:
        return "first"

    with pytest.raises(ValueError) as exc_info:
        @command
        async def duplicate_name() -> str:  # noqa: F811
            return "second"

    assert "conflict" in str(exc_info.value).lower()


def test_sync_function():
    """Test that sync functions are also supported."""
    @command
    def sync_func(x: int) -> int:
        return x * 2

    registry = get_registry()
    cmd = registry.get_command("sync_func")

    assert cmd is not None
    assert cmd.is_async is False


def test_command_docstring():
    """Test that docstrings are preserved."""
    @command
    async def documented_func(x: int) -> str:
        """This is a documented function.

        It does things.
        """
        return str(x)

    registry = get_registry()
    cmd = registry.get_command("documented_func")

    assert cmd.docstring is not None
    assert "documented function" in cmd.docstring


def test_channel_command():
    """Test command with channel parameter."""
    from zynk.channel import Channel

    @command
    async def stream_data(query: str, channel: Channel[dict]) -> None:
        pass

    registry = get_registry()
    cmd = registry.get_command("stream_data")

    assert cmd is not None
    assert cmd.has_channel is True
    # Channel should not be in params
    assert "channel" not in cmd.params
    assert "query" in cmd.params


def test_get_all_commands():
    """Test getting all registered commands."""
    @command
    async def cmd1() -> str:
        return "1"

    @command
    async def cmd2() -> str:
        return "2"

    registry = get_registry()
    commands = registry.get_all_commands()

    assert len(commands) == 2
    assert "cmd1" in commands
    assert "cmd2" in commands
