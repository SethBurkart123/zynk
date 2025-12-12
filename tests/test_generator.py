"""
Tests for the Zynk TypeScript generator module.
"""

import os
import tempfile

import pytest
from pydantic import BaseModel

from zynk.generator import TypeScriptGenerator, generate_typescript
from zynk.registry import CommandRegistry, command


class SimpleModel(BaseModel):
    """A simple test model."""
    id: int
    name: str
    is_active: bool = True


class NestedModel(BaseModel):
    """A model with nested types."""
    items: list[SimpleModel]
    metadata: dict[str, str]
    optional_field: str | None = None


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the registry before each test."""
    CommandRegistry.reset()
    yield
    CommandRegistry.reset()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_basic_generation(temp_dir):
    """Test basic TypeScript generation."""
    @command
    async def get_item(item_id: int) -> SimpleModel:
        return SimpleModel(id=item_id, name="Test")

    output_path = os.path.join(temp_dir, "api.ts")
    generate_typescript(output_path)

    assert os.path.exists(output_path)

    with open(output_path) as f:
        content = f.read()

    # Check interface generation
    assert "export interface SimpleModel" in content
    assert "id: number" in content
    assert "name: string" in content
    assert "isActive" in content  # camelCase conversion for model fields

    # Check function generation
    assert "export async function getItem" in content
    assert "args: { itemId: number }" in content  # camelCase in TypeScript signature
    assert "Promise<SimpleModel>" in content
    # Verify the conversion happens at generation time
    assert "{ item_id: args.itemId }" in content


def test_internal_module_generation(temp_dir):
    """Test that the internal module is generated without conversion functions."""
    @command
    async def dummy() -> str:
        return "test"

    output_path = os.path.join(temp_dir, "api.ts")
    generate_typescript(output_path)

    internal_path = os.path.join(temp_dir, "_internal.ts")
    assert os.path.exists(internal_path)

    with open(internal_path) as f:
        content = f.read()

    assert "initBridge" in content
    assert "request" in content
    assert "BridgeError" in content
    assert "BridgeRequestError" in content

    assert "convertKeysToSnakeCase" not in content
    assert "convertKeysToCamelCase" not in content


def test_snake_to_camel_conversion(temp_dir):
    """Test that snake_case is converted to camelCase at generation time."""
    @command
    async def get_user_by_email(user_email: str) -> SimpleModel:
        return SimpleModel(id=1, name="Test")

    output_path = os.path.join(temp_dir, "api.ts")
    generate_typescript(output_path)

    with open(output_path) as f:
        content = f.read()

    assert "getUserByEmail" in content
    assert "args: { userEmail: string }" in content
    # Verify conversion at generation time
    assert "{ user_email: args.userEmail }" in content


def test_list_type_generation(temp_dir):
    """Test List type handling."""
    @command
    async def list_items() -> list[SimpleModel]:
        return []

    output_path = os.path.join(temp_dir, "api.ts")
    generate_typescript(output_path)

    with open(output_path) as f:
        content = f.read()

    assert "Promise<SimpleModel[]>" in content


def test_optional_type_generation(temp_dir):
    """Test Optional type handling."""
    @command
    async def find_item(item_id: int) -> SimpleModel | None:
        return None

    output_path = os.path.join(temp_dir, "api.ts")
    generate_typescript(output_path)

    with open(output_path) as f:
        content = f.read()

    assert "SimpleModel | undefined" in content


def test_dict_type_generation(temp_dir):
    """Test Dict type handling."""
    @command
    async def get_metadata() -> dict[str, int]:
        return {}

    output_path = os.path.join(temp_dir, "api.ts")
    generate_typescript(output_path)

    with open(output_path) as f:
        content = f.read()

    assert "Record<string, number>" in content


def test_nested_model_generation(temp_dir):
    """Test nested model handling."""
    @command
    async def get_nested() -> NestedModel:
        return NestedModel(items=[], metadata={})

    output_path = os.path.join(temp_dir, "api.ts")
    generate_typescript(output_path)

    with open(output_path) as f:
        content = f.read()

    assert "export interface NestedModel" in content
    assert "export interface SimpleModel" in content
    assert "items" in content
    assert "SimpleModel[]" in content


def test_no_args_function(temp_dir):
    """Test function with no arguments."""
    @command
    async def get_version() -> str:
        return "1.0.0"

    output_path = os.path.join(temp_dir, "api.ts")
    generate_typescript(output_path)

    with open(output_path) as f:
        content = f.read()

    assert "getVersion()" in content
    assert "Promise<string>" in content


def test_void_return_type(temp_dir):
    """Test function with no return value."""
    @command
    async def do_something(value: str) -> None:
        pass

    output_path = os.path.join(temp_dir, "api.ts")
    generate_typescript(output_path)

    with open(output_path) as f:
        content = f.read()

    assert "Promise<void>" in content


def test_docstring_becomes_jsdoc(temp_dir):
    """Test that docstrings become JSDoc comments."""
    @command
    async def documented(x: int) -> str:
        """This is a documented function.

        It has multiple lines.
        """
        return str(x)

    output_path = os.path.join(temp_dir, "api.ts")
    generate_typescript(output_path)

    with open(output_path) as f:
        content = f.read()

    assert "/**" in content
    assert "This is a documented function" in content
    assert "*/" in content


def test_channel_function_generation(temp_dir):
    """Test channel/streaming function generation."""
    from zynk.channel import Channel

    @command
    async def stream_data(query: str, channel: Channel[SimpleModel]) -> None:
        pass

    output_path = os.path.join(temp_dir, "api.ts")
    generate_typescript(output_path)

    with open(output_path) as f:
        content = f.read()

    # Should use createChannel instead of request
    assert "BridgeChannel<SimpleModel>" in content
    assert "createChannel" in content


def test_creates_output_directory(temp_dir):
    """Test that nested output directories are created."""
    output_path = os.path.join(temp_dir, "nested", "deep", "api.ts")

    @command
    async def dummy() -> str:
        return "test"

    generate_typescript(output_path)

    assert os.path.exists(output_path)


def test_multiple_commands(temp_dir):
    """Test generation with multiple commands."""
    @command
    async def cmd1(x: int) -> str:
        return str(x)

    @command
    async def cmd2(y: str) -> int:
        return len(y)

    @command
    async def cmd3() -> bool:
        return True

    output_path = os.path.join(temp_dir, "api.ts")
    generate_typescript(output_path)

    with open(output_path) as f:
        content = f.read()

    assert "cmd1" in content
    assert "cmd2" in content
    assert "cmd3" in content


def test_type_mapping():
    """Test Python to TypeScript type mapping."""
    generator = TypeScriptGenerator()
    models: set = set()

    assert generator._type_to_ts(str, models) == "string"
    assert generator._type_to_ts(int, models) == "number"
    assert generator._type_to_ts(float, models) == "number"
    assert generator._type_to_ts(bool, models) == "boolean"
    assert generator._type_to_ts(None, models) == "void"
    assert generator._type_to_ts(type(None), models) == "undefined"
