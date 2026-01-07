"""
Tests for the upload module.
"""

import io
import tempfile
from pathlib import Path

import pytest
from fastapi import UploadFile as FastAPIUploadFile
from pydantic import BaseModel

from zynk import UploadFile, upload, get_registry, generate_typescript
from zynk.upload import parse_size, matches_mime_type, UploadInfo


class TestParseSize:
    """Tests for the parse_size utility function."""

    def test_parse_bytes(self):
        assert parse_size(1024) == 1024
        assert parse_size(0) == 0

    def test_parse_kb(self):
        assert parse_size("1KB") == 1024
        assert parse_size("10KB") == 10 * 1024
        assert parse_size("1kb") == 1024

    def test_parse_mb(self):
        assert parse_size("1MB") == 1024 * 1024
        assert parse_size("5MB") == 5 * 1024 * 1024
        assert parse_size("10MB") == 10 * 1024 * 1024

    def test_parse_gb(self):
        assert parse_size("1GB") == 1024 * 1024 * 1024

    def test_parse_with_spaces(self):
        assert parse_size("  10 MB  ") == 10 * 1024 * 1024

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            parse_size("invalid")


class TestMatchesMimeType:
    """Tests for the matches_mime_type utility function."""

    def test_exact_match(self):
        assert matches_mime_type("image/png", ["image/png"])
        assert matches_mime_type("application/pdf", ["application/pdf"])

    def test_wildcard_match(self):
        assert matches_mime_type("image/png", ["image/*"])
        assert matches_mime_type("image/jpeg", ["image/*"])
        assert matches_mime_type("video/mp4", ["video/*"])

    def test_no_match(self):
        assert not matches_mime_type("text/plain", ["image/*"])
        assert not matches_mime_type("application/json", ["image/png", "image/jpeg"])

    def test_empty_patterns(self):
        assert matches_mime_type("anything/here", [])

    def test_multiple_patterns(self):
        assert matches_mime_type("image/png", ["application/pdf", "image/*"])
        assert matches_mime_type("application/pdf", ["application/pdf", "image/*"])


class TestUploadDecorator:
    """Tests for the @upload decorator."""

    def setup_method(self):
        """Reset registry before each test."""
        from zynk.registry import CommandRegistry

        CommandRegistry.reset()

    def test_basic_registration(self):
        """Test that a basic upload handler is registered."""

        class Result(BaseModel):
            filename: str

        @upload
        async def test_upload(file: UploadFile) -> Result:
            return Result(filename=file.filename)

        registry = get_registry()
        uploads = registry.get_all_uploads()

        assert "test_upload" in uploads
        handler = uploads["test_upload"]
        assert handler.name == "test_upload"
        assert handler.file_param == "file"
        assert handler.is_multi_file is False

    def test_multi_file_registration(self):
        """Test that multi-file upload is detected."""

        class Result(BaseModel):
            count: int

        @upload
        async def upload_many(files: list[UploadFile]) -> Result:
            return Result(count=len(files))

        registry = get_registry()
        handler = registry.get_upload("upload_many")

        assert handler is not None
        assert handler.file_param == "files"
        assert handler.is_multi_file is True

    def test_with_validation_options(self):
        """Test that validation options are stored."""

        class Result(BaseModel):
            ok: bool

        @upload(max_size="5MB", allowed_types=["image/*", "application/pdf"])
        async def upload_validated(file: UploadFile) -> Result:
            return Result(ok=True)

        registry = get_registry()
        handler = registry.get_upload("upload_validated")

        assert handler is not None
        assert handler.max_size == 5 * 1024 * 1024
        assert handler.allowed_types == ["image/*", "application/pdf"]

    def test_with_additional_params(self):
        """Test that additional parameters are captured."""

        class Result(BaseModel):
            filename: str

        @upload
        async def upload_with_params(
            file: UploadFile,
            folder_id: str,
            quality: int = 80,
        ) -> Result:
            return Result(filename=file.filename)

        registry = get_registry()
        handler = registry.get_upload("upload_with_params")

        assert handler is not None
        assert "folder_id" in handler.params
        assert "quality" in handler.params
        assert "quality" in handler.optional_params
        assert "folder_id" not in handler.optional_params

    def test_custom_name(self):
        """Test custom handler name."""

        class Result(BaseModel):
            ok: bool

        @upload(name="custom_upload_name")
        async def internal_name(file: UploadFile) -> Result:
            return Result(ok=True)

        registry = get_registry()
        assert registry.get_upload("custom_upload_name") is not None
        assert registry.get_upload("internal_name") is None

    def test_missing_file_param_raises(self):
        """Test that missing UploadFile parameter raises TypeError."""
        with pytest.raises(TypeError, match="must have a parameter"):

            @upload
            async def no_file(name: str) -> str:
                return name


class TestUploadInfoValidation:
    """Tests for UploadInfo.validate_file."""

    def test_validate_size(self):
        """Test file size validation."""
        info = UploadInfo(
            name="test",
            func=lambda: None,
            params={},
            file_param="file",
            is_multi_file=False,
            return_type=None,
            is_async=True,
            docstring=None,
            module="test",
            optional_params=set(),
            max_size=1024,  # 1KB
            allowed_types=[],
        )

        # Create a mock UploadFile
        class MockUploadFile:
            filename = "test.txt"
            content_type = "text/plain"
            size = 2048  # 2KB - too big

        with pytest.raises(ValueError, match="exceeds maximum size"):
            info.validate_file(MockUploadFile())

    def test_validate_type(self):
        """Test file type validation."""
        info = UploadInfo(
            name="test",
            func=lambda: None,
            params={},
            file_param="file",
            is_multi_file=False,
            return_type=None,
            is_async=True,
            docstring=None,
            module="test",
            optional_params=set(),
            max_size=None,
            allowed_types=["image/*"],
        )

        class MockUploadFile:
            filename = "test.txt"
            content_type = "text/plain"  # Not an image
            size = 100

        with pytest.raises(ValueError, match="not in allowed types"):
            info.validate_file(MockUploadFile())


class TestTypeScriptGeneration:
    """Tests for TypeScript generation of upload handlers."""

    def setup_method(self):
        """Reset registry before each test."""
        from zynk.registry import CommandRegistry

        CommandRegistry.reset()

    def test_generates_upload_function(self):
        """Test that upload functions are generated correctly."""

        class Result(BaseModel):
            filename: str

        @upload
        async def upload_test(file: UploadFile) -> Result:
            return Result(filename="test")

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "api.ts"
            generate_typescript(str(output))

            content = output.read_text()

            assert "export function uploadTest" in content
            assert "UploadHandle<Result>" in content
            assert "file: File" in content
            assert "createUpload" in content

    def test_generates_multi_file_function(self):
        """Test that multi-file upload uses File[]."""

        class Result(BaseModel):
            count: int

        @upload
        async def upload_many(files: list[UploadFile]) -> list[Result]:
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "api.ts"
            generate_typescript(str(output))

            content = output.read_text()

            assert "files: File[]" in content
            assert "UploadHandle<Result[]>" in content

    def test_generates_with_additional_params(self):
        """Test that additional params are included."""

        class Result(BaseModel):
            ok: bool

        @upload
        async def upload_params(
            file: UploadFile,
            folder_id: str,
            quality: int = 80,
        ) -> Result:
            return Result(ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "api.ts"
            generate_typescript(str(output))

            content = output.read_text()

            assert "folderId: string" in content
            assert "quality?: number" in content

    def test_internal_module_has_upload_types(self):
        """Test that _internal.ts includes upload types."""

        class Result(BaseModel):
            ok: bool

        @upload
        async def test_upload(file: UploadFile) -> Result:
            return Result(ok=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "api.ts"
            generate_typescript(str(output))

            internal = Path(tmpdir) / "_internal.ts"
            content = internal.read_text()

            assert "interface UploadProgressEvent" in content
            assert "interface UploadHandle<T>" in content
            assert "function createUpload<T>" in content
            assert "xhr.upload.onprogress" in content
