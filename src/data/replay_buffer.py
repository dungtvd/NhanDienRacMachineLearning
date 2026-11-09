from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from PIL import Image, UnidentifiedImageError


class ReplayBuffer:
    """A bounded, class-balanced store of confirmed, non-test training images."""

    def __init__(
        self,
        root: str | Path,
        class_names: Sequence[str],
        images_per_class: int,
        seed: int = 42,
    ) -> None:
        if images_per_class < 1:
            raise ValueError("images_per_class must be positive.")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.class_names = tuple(class_names)
        self.images_per_class = images_per_class
        self.seed = seed
        self.manifest_path = self.root / "manifest.json"
        self._entries = self._load_manifest()

    def _load_manifest(self) -> list[dict]:
        if not self.manifest_path.exists():
            return []
        with self.manifest_path.open(encoding="utf-8") as file:
            entries = json.load(file)
        if not isinstance(entries, list):
            raise ValueError(f"Invalid replay manifest: {self.manifest_path}")
        for entry in entries:
            if entry.get("class_name") not in self.class_names:
                raise ValueError("Replay manifest contains an unknown class.")
            if not (self.root / entry["file"]).is_file():
                raise FileNotFoundError(
                    f"Replay image is missing: {self.root / entry['file']}"
                )
        return entries

    def _save_manifest(self) -> None:
        temporary = self.manifest_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as file:
            json.dump(self._entries, file, indent=2)
        temporary.replace(self.manifest_path)

    def add(
        self, records: Sequence[tuple[Path, int]], source: str = "confirmed_update"
    ) -> int:
        seen = {entry["sha256"] for entry in self._entries}
        added = 0
        for path, label in records:
            if not 0 <= label < len(self.class_names):
                raise ValueError(f"Unknown replay label index: {label}")
            source_path = Path(path)
            try:
                content = source_path.read_bytes()
                with Image.open(source_path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError, ValueError) as error:
                raise ValueError(f"Invalid replay image {source_path}: {error}") from error

            digest = hashlib.sha256(content).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            class_name = self.class_names[label]
            destination = Path(class_name) / f"{uuid4().hex}{source_path.suffix.lower()}"
            target = self.root / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, target)
            self._entries.append(
                {
                    "file": destination.as_posix(),
                    "class_name": class_name,
                    "source": source,
                    "sha256": digest,
                }
            )
            added += 1

        rng = random.Random(self.seed)
        retained = []
        for class_name in self.class_names:
            class_entries = [
                entry for entry in self._entries if entry["class_name"] == class_name
            ]
            rng.shuffle(class_entries)
            retained.extend(class_entries[: self.images_per_class])
        retained_files = {entry["file"] for entry in retained}
        for entry in self._entries:
            if entry["file"] not in retained_files:
                (self.root / entry["file"]).unlink(missing_ok=True)
        self._entries = retained
        self._save_manifest()
        return added

    def sample(self, count: int, seed: int | None = None) -> list[tuple[Path, int]]:
        if count < 0:
            raise ValueError("Replay sample count cannot be negative.")
        rng = random.Random(self.seed if seed is None else seed)
        selected = []
        per_class = {name: [] for name in self.class_names}
        for entry in self._entries:
            per_class[entry["class_name"]].append(entry)
        while len(selected) < min(count, len(self._entries)):
            progressed = False
            for class_name in self.class_names:
                if per_class[class_name] and len(selected) < count:
                    selected.append(per_class[class_name].pop(rng.randrange(len(per_class[class_name]))))
                    progressed = True
            if not progressed:
                break
        name_to_label = {name: index for index, name in enumerate(self.class_names)}
        return [
            (self.root / entry["file"], name_to_label[entry["class_name"]])
            for entry in selected
        ]

    def counts(self) -> dict[str, int]:
        return {
            class_name: sum(entry["class_name"] == class_name for entry in self._entries)
            for class_name in self.class_names
        }

