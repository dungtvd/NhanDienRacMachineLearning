from __future__ import annotations

import io
import zipfile

from PIL import Image
import pytest

from src.data.upload_validation import read_labeled_zip, validate_image_bytes


def png_bytes(color: tuple[int, int, int]) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(output, format="PNG")
    return output.getvalue()


def make_zip(entries: list[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return output.getvalue()


def test_reads_class_folders_without_extracting(tmp_path) -> None:
    content = png_bytes((32, 64, 96))
    archive_bytes = make_zip(
        [
            ("dataset/cardboard/one.png", content),
            ("dataset/glass/two.png", png_bytes((96, 64, 32))),
        ]
    )

    records = read_labeled_zip(
        archive_bytes,
        ("cardboard", "glass"),
        max_image_bytes=1024,
    )
    assert [(label, name.rsplit("/", 1)[-1]) for name, label, _ in records] == [
        ("cardboard", "one.png"),
        ("glass", "two.png"),
    ]
    assert not list(tmp_path.iterdir())


def test_reads_nested_class_folders() -> None:
    records = read_labeled_zip(
        make_zip(
            [
                ("dataset/cardboard/session-a/one.png", png_bytes((32, 64, 96))),
                ("dataset/glass/session-b/two.png", png_bytes((96, 64, 32))),
            ]
        ),
        ("cardboard", "glass"),
        max_image_bytes=1024,
    )
    assert [label for _, label, _ in records] == ["cardboard", "glass"]


def test_rejects_identical_images_with_conflicting_zip_labels() -> None:
    identical = png_bytes((32, 64, 96))
    with pytest.raises(ValueError, match="conflicting class folders"):
        read_labeled_zip(
            make_zip(
                [
                    ("cardboard/one.png", identical),
                    ("glass/two.png", identical),
                ]
            ),
            ("cardboard", "glass"),
            max_image_bytes=1024,
        )


@pytest.mark.parametrize(
    "member",
    [
        ("../cardboard/escape.png", png_bytes((1, 2, 3))),
        ("cardboard/../../escape.png", png_bytes((1, 2, 3))),
        ("cardboard/readme.txt", b"not an image"),
        ("unknown/image.png", png_bytes((1, 2, 3))),
    ],
)
def test_rejects_unsafe_or_unlabeled_zip_members(member) -> None:
    with pytest.raises(ValueError):
        read_labeled_zip(
            make_zip([member]),
            ("cardboard", "glass"),
            max_image_bytes=1024,
        )


def test_enforces_zip_limits_and_validates_images() -> None:
    content = png_bytes((1, 2, 3))
    with pytest.raises(ValueError, match="limit"):
        read_labeled_zip(
            make_zip(
                [
                    ("cardboard/one.png", content),
                    ("cardboard/two.png", png_bytes((4, 5, 6))),
                ]
            ),
            ("cardboard",),
            max_image_bytes=1024,
            max_files=1,
        )
    with pytest.raises(ValueError, match="size exceeds"):
        validate_image_bytes(content, max_bytes=1)
