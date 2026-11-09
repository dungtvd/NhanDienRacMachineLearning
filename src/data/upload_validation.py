from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from pathlib import PurePosixPath
from typing import Sequence

from PIL import Image, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def validate_image_bytes(content: bytes, max_bytes: int) -> str:
    if not content:
        raise ValueError("Image file is empty.")
    if len(content) > max_bytes:
        raise ValueError(f"Image size exceeds the limit of {max_bytes} bytes.")
    try:
        with Image.open(io.BytesIO(content)) as image:
            if image.format not in {"JPEG", "PNG", "BMP"}:
                raise ValueError(f"Unsupported image format: {image.format}")
            width, height = image.size
            if width < 1 or height < 1 or width * height > 25_000_000:
                raise ValueError("Image dimensions are invalid or exceed 25 megapixels.")
            image.verify()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise ValueError(f"Invalid image file: {error}") from error
    return hashlib.sha256(content).hexdigest()


def read_labeled_zip(
    archive_bytes: bytes,
    class_names: Sequence[str],
    max_image_bytes: int,
    max_files: int = 500,
    max_total_uncompressed_bytes: int = 100 * 1024 * 1024,
) -> list[tuple[str, str, bytes]]:
    """Read labeled image members without extracting archive paths to disk."""
    if not archive_bytes:
        raise ValueError("The ZIP archive is empty.")
    known_classes = set(class_names)
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except (OSError, zipfile.BadZipFile) as error:
        raise ValueError(f"Invalid ZIP archive: {error}") from error

    with archive:
        members = archive.infolist()
        files = [member for member in members if not member.is_dir()]
        if not files:
            raise ValueError("The ZIP archive contains no files.")
        if len(files) > max_files:
            raise ValueError(f"ZIP archive exceeds the limit of {max_files} files.")
        if sum(member.file_size for member in files) > max_total_uncompressed_bytes:
            raise ValueError("ZIP archive exceeds the total uncompressed size limit.")

        candidates: list[tuple[zipfile.ZipInfo, str, PurePosixPath]] = []
        for member in files:
            raw_name = member.filename.replace("\\", "/")
            path = PurePosixPath(raw_name)
            mode = member.external_attr >> 16
            if (
                path.is_absolute()
                or not path.parts
                or any(part in {"", ".", ".."} for part in path.parts)
                or ":" in path.parts[0]
                or stat.S_ISLNK(mode)
                or member.flag_bits & 0x1
            ):
                raise ValueError(f"Unsafe ZIP member path or file type: {member.filename}")
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError(f"ZIP contains a non-supported image file: {member.filename}")
            candidates.append((member, raw_name, path))

        matching_offsets = [
            offset
            for offset in (0, 1)
            if all(
                len(path.parts) > offset + 1
                and path.parts[offset] in known_classes
                for _, _, path in candidates
            )
        ]
        if len(matching_offsets) != 1:
            raise ValueError(
                "ZIP images must use class/image.ext folders (nested folders are "
                "allowed), optionally inside one common enclosing folder."
            )
        prefix_length = matching_offsets[0]

        seen_paths: set[str] = set()
        seen_hashes: dict[str, str] = {}
        result = []
        for member, original_name, path in candidates:
            class_name = path.parts[prefix_length]
            if class_name not in known_classes:
                raise ValueError(f"Unknown class folder in ZIP: {class_name}")
            if path.parts[prefix_length + 1] == "":
                raise ValueError(f"Invalid image path in ZIP: {original_name}")
            normalized = path.as_posix().casefold()
            if normalized in seen_paths:
                raise ValueError(f"Duplicate file path in ZIP: {original_name}")
            seen_paths.add(normalized)
            if member.file_size > max_image_bytes:
                raise ValueError(f"Image exceeds the size limit: {original_name}")
            try:
                content = archive.read(member)
            except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                raise ValueError(f"Could not read ZIP member {original_name}: {error}") from error
            digest = validate_image_bytes(content, max_image_bytes)
            previous_class = seen_hashes.get(digest)
            if previous_class is not None and previous_class != class_name:
                raise ValueError(
                    "Identical image bytes appear under conflicting class folders."
                )
            if previous_class is not None:
                continue
            seen_hashes[digest] = class_name
            result.append((original_name, class_name, content))
        return result
