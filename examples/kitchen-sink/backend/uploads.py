"""
Uploads Module

Demonstrates file upload support with progress tracking and validation.
"""

import base64
import hashlib
import uuid
from datetime import datetime

from pydantic import BaseModel

from zynk import UploadFile, upload


class FileInfo(BaseModel):
    """Information about an uploaded file."""

    id: str
    filename: str
    size: int
    content_type: str
    checksum: str
    uploaded_at: str


class ImageUploadResult(BaseModel):
    """Result of an image upload."""

    id: str
    filename: str
    width: int | None = None
    height: int | None = None
    thumbnail_base64: str | None = None


class DocumentUploadResult(BaseModel):
    """Result of a document upload."""

    id: str
    filename: str
    size: int
    page_count: int | None = None


# In-memory storage for demo purposes
_uploaded_files: dict[str, bytes] = {}


@upload
async def upload_file(file: UploadFile) -> FileInfo:
    """
    Upload a single file of any type.

    Returns file information including a checksum.
    """
    content = await file.read()
    file_id = str(uuid.uuid4())

    # Store in memory (in real app, save to disk/cloud)
    _uploaded_files[file_id] = content

    # Calculate checksum
    checksum = hashlib.sha256(content).hexdigest()[:16]

    return FileInfo(
        id=file_id,
        filename=file.filename,
        size=len(content),
        content_type=file.content_type,
        checksum=checksum,
        uploaded_at=datetime.now().isoformat(),
    )


@upload
async def upload_files(files: list[UploadFile]) -> list[FileInfo]:
    """
    Upload multiple files at once.

    Returns information about all uploaded files.
    """
    results = []

    for file in files:
        content = await file.read()
        file_id = str(uuid.uuid4())

        _uploaded_files[file_id] = content
        checksum = hashlib.sha256(content).hexdigest()[:16]

        results.append(
            FileInfo(
                id=file_id,
                filename=file.filename,
                size=len(content),
                content_type=file.content_type,
                checksum=checksum,
                uploaded_at=datetime.now().isoformat(),
            )
        )

    return results


@upload(max_size="5MB", allowed_types=["image/*"])
async def upload_image(
    file: UploadFile, generate_thumbnail: bool = False
) -> ImageUploadResult:
    """
    Upload an image file with optional thumbnail generation.

    Only accepts image files up to 5MB.
    """
    content = await file.read()
    file_id = str(uuid.uuid4())

    _uploaded_files[file_id] = content

    result = ImageUploadResult(
        id=file_id,
        filename=file.filename,
    )

    # Try to get image dimensions (would need PIL in real implementation)
    # For demo, we'll just return None for dimensions

    if generate_thumbnail:
        # In a real app, you'd resize the image
        # For demo, just return first 100 bytes as base64
        result.thumbnail_base64 = base64.b64encode(content[:100]).decode()

    return result


@upload(
    max_size="10MB",
    allowed_types=[
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ],
)
async def upload_document(
    file: UploadFile, extract_metadata: bool = True
) -> DocumentUploadResult:
    """
    Upload a document (PDF, DOC, DOCX).

    Only accepts document files up to 10MB.
    """
    content = await file.read()
    file_id = str(uuid.uuid4())

    _uploaded_files[file_id] = content

    result = DocumentUploadResult(
        id=file_id,
        filename=file.filename,
        size=len(content),
    )

    if extract_metadata:
        # In a real app, you'd parse the document for page count
        # For demo, estimate based on file size
        result.page_count = max(1, len(content) // 50000)

    return result


@upload(max_size="50MB", allowed_types=["image/*", "video/*", "audio/*"])
async def upload_media(
    files: list[UploadFile], album_name: str | None = None
) -> list[FileInfo]:
    """
    Upload multiple media files (images, videos, audio).

    Accepts files up to 50MB each, optionally grouped into an album.
    """
    results = []

    for file in files:
        content = await file.read()
        file_id = str(uuid.uuid4())

        _uploaded_files[file_id] = content
        checksum = hashlib.sha256(content).hexdigest()[:16]

        results.append(
            FileInfo(
                id=file_id,
                filename=f"{album_name}/{file.filename}"
                if album_name
                else file.filename,
                size=len(content),
                content_type=file.content_type,
                checksum=checksum,
                uploaded_at=datetime.now().isoformat(),
            )
        )

    return results
