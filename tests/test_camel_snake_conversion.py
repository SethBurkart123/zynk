"""
Tests for camelCase/snake_case conversion between TypeScript frontend and Python backend.

These tests verify that:
1. Pydantic models with snake_case fields work correctly when receiving camelCase data from frontend
2. Nested models handle conversion properly
3. The full request/response cycle maintains correct field naming
"""

import os
import tempfile

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from zynk.bridge import Bridge
from zynk.generator import generate_typescript
from zynk.registry import CommandRegistry, command


# --- Test Models with snake_case fields ---


class ProviderSettings(BaseModel):
    """Settings for an API provider."""
    provider: str
    api_key: str
    base_url: str | None = None
    max_retries: int = 3


class NestedConfig(BaseModel):
    """Config with nested model containing snake_case fields."""
    config_name: str
    provider_settings: ProviderSettings
    is_enabled: bool = True


class UserProfile(BaseModel):
    """User profile with various snake_case fields."""
    user_id: int
    full_name: str
    email_address: str
    is_active: bool = True
    profile_picture_url: str | None = None


class BulkOperation(BaseModel):
    """Model for bulk operations with list of nested models."""
    operation_type: str
    target_users: list[UserProfile]
    notify_on_complete: bool = False


class DeeplyNestedModel(BaseModel):
    """Model with multiple levels of nesting."""
    outer_field: str
    nested_config: NestedConfig
    additional_data: dict[str, str] | None = None


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


class TestCamelCaseModelParameters:
    """Test that commands accepting Pydantic models work with camelCase input."""

    def test_simple_model_with_snake_case_fields(self, temp_dir):
        """Test command that accepts a model with snake_case fields."""
        @command
        async def save_provider(settings: ProviderSettings) -> ProviderSettings:
            """Save provider settings."""
            return settings

        # Generate TypeScript
        output_path = os.path.join(temp_dir, "api.ts")
        generate_typescript(output_path)

        with open(output_path) as f:
            content = f.read()

        # Verify interface uses camelCase
        assert "export interface ProviderSettings" in content
        assert "apiKey: string" in content  # snake_case -> camelCase
        assert "baseUrl?: string" in content
        assert "maxRetries?: number" in content

        # Verify the function uses a converter to map camelCase back to snake_case
        assert "saveProvider" in content
        assert "convertProviderSettings(args.settings)" in content

    def test_command_execution_with_camel_case_input(self, temp_dir):
        """Test that commands execute correctly when receiving snake_case data (after TS conversion)."""
        received_settings = None

        @command
        async def save_provider_settings(settings: ProviderSettings) -> dict:
            nonlocal received_settings
            received_settings = settings
            return {"status": "ok", "provider": settings.provider}

        bridge = Bridge(host="127.0.0.1", port=8000)
        client = TestClient(bridge.app)

        # Send snake_case data (as the TypeScript client would after conversion)
        response = client.post(
            "/command/save_provider_settings",
            json={
                "settings": {
                    "provider": "openai",
                    "api_key": "sk-test-key",
                    "base_url": "https://api.openai.com",
                    "max_retries": 5,
                }
            },
        )

        assert response.status_code == 200, f"Failed: {response.json()}"
        result = response.json()["result"]
        assert result["status"] == "ok"
        assert result["provider"] == "openai"

        # Verify the model was properly instantiated
        assert received_settings is not None
        assert received_settings.provider == "openai"
        assert received_settings.api_key == "sk-test-key"
        assert received_settings.base_url == "https://api.openai.com"
        assert received_settings.max_retries == 5

    def test_nested_model_with_camel_case_input(self, temp_dir):
        """Test nested models with snake_case fields receive snake_case data correctly."""
        received_config = None

        @command
        async def save_config(config: NestedConfig) -> str:
            nonlocal received_config
            received_config = config
            return f"Saved: {config.config_name}"

        bridge = Bridge(host="127.0.0.1", port=8000)
        client = TestClient(bridge.app)

        # Send snake_case data with nested model
        response = client.post(
            "/command/save_config",
            json={
                "config": {
                    "config_name": "production",
                    "provider_settings": {
                        "provider": "anthropic",
                        "api_key": "sk-ant-key",
                        "base_url": None,
                        "max_retries": 3,
                    },
                    "is_enabled": True,
                }
            },
        )

        assert response.status_code == 200, f"Failed: {response.json()}"
        assert "Saved: production" in response.json()["result"]

        # Verify nested model was properly instantiated
        assert received_config is not None
        assert received_config.config_name == "production"
        assert received_config.provider_settings.provider == "anthropic"
        assert received_config.provider_settings.api_key == "sk-ant-key"
        assert received_config.is_enabled is True

    def test_model_with_list_of_nested_models(self, temp_dir):
        """Test model containing a list of models with snake_case fields."""
        received_operation = None

        @command
        async def bulk_update(operation: BulkOperation) -> int:
            nonlocal received_operation
            received_operation = operation
            return len(operation.target_users)

        bridge = Bridge(host="127.0.0.1", port=8000)
        client = TestClient(bridge.app)

        # Send snake_case data with list of nested models
        response = client.post(
            "/command/bulk_update",
            json={
                "operation": {
                    "operation_type": "deactivate",
                    "target_users": [
                        {
                            "user_id": 1,
                            "full_name": "Alice Smith",
                            "email_address": "alice@example.com",
                            "is_active": True,
                            "profile_picture_url": "https://example.com/alice.jpg",
                        },
                        {
                            "user_id": 2,
                            "full_name": "Bob Jones",
                            "email_address": "bob@example.com",
                            "is_active": True,
                            "profile_picture_url": None,
                        },
                    ],
                    "notify_on_complete": True,
                }
            },
        )

        assert response.status_code == 200, f"Failed: {response.json()}"
        assert response.json()["result"] == 2

        # Verify list of models was properly instantiated
        assert received_operation is not None
        assert received_operation.operation_type == "deactivate"
        assert len(received_operation.target_users) == 2
        assert received_operation.target_users[0].full_name == "Alice Smith"
        assert received_operation.target_users[0].email_address == "alice@example.com"
        assert received_operation.target_users[1].user_id == 2
        assert received_operation.notify_on_complete is True

    def test_deeply_nested_model(self, temp_dir):
        """Test deeply nested models with snake_case fields."""
        received_data = None

        @command
        async def save_deep(data: DeeplyNestedModel) -> str:
            nonlocal received_data
            received_data = data
            return data.outer_field

        bridge = Bridge(host="127.0.0.1", port=8000)
        client = TestClient(bridge.app)

        response = client.post(
            "/command/save_deep",
            json={
                "data": {
                    "outer_field": "outer-value",
                    "nested_config": {
                        "config_name": "deep-config",
                        "provider_settings": {
                            "provider": "google",
                            "api_key": "google-key",
                            "max_retries": 2,
                        },
                        "is_enabled": False,
                    },
                    "additional_data": {"keyOne": "value1", "keyTwo": "value2"},
                }
            },
        )

        assert response.status_code == 200, f"Failed: {response.json()}"
        assert response.json()["result"] == "outer-value"

        # Verify deep nesting was properly handled
        assert received_data is not None
        assert received_data.outer_field == "outer-value"
        assert received_data.nested_config.config_name == "deep-config"
        assert received_data.nested_config.provider_settings.provider == "google"
        assert received_data.nested_config.is_enabled is False


class TestMixedParameterTypes:
    """Test commands with mixed parameter types (primitives + models)."""

    def test_model_with_primitive_params(self, temp_dir):
        """Test command with both Pydantic model and primitive parameters."""
        received_values = {}

        @command
        async def update_user_settings(
            user_id: int,
            settings: ProviderSettings,
            notify_user: bool = False,
        ) -> dict:
            nonlocal received_values
            received_values = {
                "user_id": user_id,
                "settings": settings,
                "notify_user": notify_user,
            }
            return {"updated": True}

        bridge = Bridge(host="127.0.0.1", port=8000)
        client = TestClient(bridge.app)

        # Send with snake_case primitive params and camelCase model fields
        response = client.post(
            "/command/update_user_settings",
            json={
                "user_id": 42,
                "settings": {
                    "provider": "azure",
                    "api_key": "azure-key",
                    "base_url": "https://azure.com",
                },
                "notify_user": True,
            },
        )

        assert response.status_code == 200, f"Failed: {response.json()}"

        # Verify all values received correctly
        assert received_values["user_id"] == 42
        assert received_values["settings"].provider == "azure"
        assert received_values["settings"].api_key == "azure-key"
        assert received_values["notify_user"] is True


class TestReturnValueSerialization:
    """Test that return values serialize snake_case fields correctly."""

    def test_return_model_serializes_to_snake_case(self, temp_dir):
        """Verify returned models use snake_case in JSON response."""
        @command
        async def get_user(user_id: int) -> UserProfile:
            return UserProfile(
                user_id=user_id,
                full_name="Test User",
                email_address="test@example.com",
                profile_picture_url="https://example.com/pic.jpg",
            )

        bridge = Bridge(host="127.0.0.1", port=8000)
        client = TestClient(bridge.app)

        response = client.post("/command/get_user", json={"user_id": 123})

        assert response.status_code == 200
        result = response.json()["result"]

        # Pydantic model_dump() returns snake_case by default
        assert result["user_id"] == 123
        assert result["full_name"] == "Test User"
        assert result["email_address"] == "test@example.com"
        assert result["profile_picture_url"] == "https://example.com/pic.jpg"

    def test_return_list_of_models(self, temp_dir):
        """Verify returned lists of models serialize correctly."""
        @command
        async def list_users() -> list[UserProfile]:
            return [
                UserProfile(user_id=1, full_name="User One", email_address="one@example.com"),
                UserProfile(user_id=2, full_name="User Two", email_address="two@example.com"),
            ]

        bridge = Bridge(host="127.0.0.1", port=8000)
        client = TestClient(bridge.app)

        response = client.post("/command/list_users", json={})

        assert response.status_code == 200
        result = response.json()["result"]

        assert len(result) == 2
        assert result[0]["full_name"] == "User One"
        assert result[1]["email_address"] == "two@example.com"


class TestTypeScriptGenerationFieldMapping:
    """Test TypeScript generation produces correct field mappings."""

    def test_model_interface_uses_camel_case(self, temp_dir):
        """Verify generated interfaces use camelCase field names."""
        @command
        async def dummy(settings: ProviderSettings) -> None:
            pass

        output_path = os.path.join(temp_dir, "api.ts")
        generate_typescript(output_path)

        with open(output_path) as f:
            content = f.read()

        # Interface should have camelCase fields
        assert "export interface ProviderSettings {" in content
        assert "provider: string" in content
        assert "apiKey: string" in content
        assert "baseUrl" in content
        assert "maxRetries" in content

        # Extract just the interface section to verify no snake_case there
        # (converter functions intentionally use snake_case for the API)
        interface_start = content.find("export interface ProviderSettings {")
        interface_end = content.find("}", interface_start) + 1
        interface_section = content[interface_start:interface_end]
        
        # Should NOT have snake_case in interface (but converters will have them)
        assert "api_key:" not in interface_section
        assert "base_url:" not in interface_section
        assert "max_retries:" not in interface_section
        
        # Verify converter function exists and maps to snake_case
        assert "convertProviderSettings" in content
        assert "api_key: obj.apiKey" in content

    def test_deeply_nested_model_interfaces(self, temp_dir):
        """Verify nested models all generate with camelCase."""
        @command
        async def dummy(data: DeeplyNestedModel) -> None:
            pass

        output_path = os.path.join(temp_dir, "api.ts")
        generate_typescript(output_path)

        with open(output_path) as f:
            content = f.read()

        # All interfaces should use camelCase
        assert "outerField: string" in content
        assert "nestedConfig: NestedConfig" in content
        assert "configName: string" in content
        assert "providerSettings: ProviderSettings" in content
        assert "isEnabled" in content


class TestEdgeCases:
    """Test edge cases in camelCase/snake_case handling."""

    def test_single_word_fields(self, temp_dir):
        """Single word fields should remain unchanged."""

        class SimpleModel(BaseModel):
            name: str
            value: int

        @command
        async def test_simple(data: SimpleModel) -> str:
            return data.name

        bridge = Bridge(host="127.0.0.1", port=8000)
        client = TestClient(bridge.app)

        response = client.post(
            "/command/test_simple",
            json={"data": {"name": "test", "value": 42}},
        )

        assert response.status_code == 200
        assert response.json()["result"] == "test"

    def test_optional_model_parameter(self, temp_dir):
        """Test optional model parameter with camelCase fields."""
        received_settings = "not_called"

        @command
        async def maybe_settings(settings: ProviderSettings | None = None) -> str:
            nonlocal received_settings
            received_settings = settings
            return "ok" if settings else "no settings"

        bridge = Bridge(host="127.0.0.1", port=8000)
        client = TestClient(bridge.app)

        # Test with settings
        response = client.post(
            "/command/maybe_settings",
            json={
                "settings": {
                    "provider": "test",
                    "api_key": "key",
                }
            },
        )
        assert response.status_code == 200
        assert response.json()["result"] == "ok"
        assert received_settings.api_key == "key"

        # Test without settings
        response = client.post("/command/maybe_settings", json={})
        assert response.status_code == 200
        assert response.json()["result"] == "no settings"

    def test_model_with_field_alias(self, temp_dir):
        """Test model with Pydantic Field alias."""

        class AliasedModel(BaseModel):
            internal_name: str = Field(..., alias="externalName")

            model_config = {"populate_by_name": True}

        @command
        async def test_alias(data: AliasedModel) -> str:
            return data.internal_name

        bridge = Bridge(host="127.0.0.1", port=8000)
        client = TestClient(bridge.app)

        # Send with alias name
        response = client.post(
            "/command/test_alias",
            json={"data": {"externalName": "aliased-value"}},
        )

        assert response.status_code == 200
        assert response.json()["result"] == "aliased-value"


class TestErrorHandling:
    """Test error handling for invalid snake_case data."""

    def test_missing_required_camel_case_field(self, temp_dir):
        """Test error when required snake_case field is missing."""
        @command
        async def requires_all(settings: ProviderSettings) -> str:
            return "ok"

        bridge = Bridge(host="127.0.0.1", port=8000)
        client = TestClient(bridge.app)

        # Missing api_key (required field)
        response = client.post(
            "/command/requires_all",
            json={
                "settings": {
                    "provider": "test",
                    # api_key missing
                }
            },
        )

        # Should fail validation
        assert response.status_code == 400

    def test_wrong_type_for_camel_case_field(self, temp_dir):
        """Test error when snake_case field has wrong type."""
        @command
        async def type_check(settings: ProviderSettings) -> str:
            return "ok"

        bridge = Bridge(host="127.0.0.1", port=8000)
        client = TestClient(bridge.app)

        # max_retries should be int, not string
        response = client.post(
            "/command/type_check",
            json={
                "settings": {
                    "provider": "test",
                    "api_key": "key",
                    "base_url": "https://api.openai.com",
                    "max_retries": "not-a-number",
                }
            },
        )

        # Should fail validation
        assert response.status_code == 400
