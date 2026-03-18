"""
Датасет PyTorch для классификации изображений.

Поддерживает:
  - Классификацию целого изображения (метка из JSON аннотации, без bbox)
  - Классификацию вырезанных областей (метка + bbox из JSON аннотации)
  - Резервный вариант по имени папки (имя родительской папки = ключ класса)
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms

from config import DEFAULT_IMAGE_SIZE, MODEL_DEFS
from app.utils import annotation as ann_io
from app.utils.file_utils import collect_images, IMAGE_EXTS


# ── Преобразования ────────────────────────────────────────────────────────────

def get_train_transforms(image_size: tuple = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: tuple = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


# ── Датасет ───────────────────────────────────────────────────────────────────

class SpaceDataset(Dataset):
    """
    Загружает изображения с метками классов для конкретного model_id.

    Порядок определения метки:
      1. Первая аннотация в JSON, принадлежащая model_id
      2. Имя папки совпадает с ключом класса для model_id
      3. Изображение пропускается (метка не найдена)

    Также может быть создан из готовых сэмплов (_samples kwarg)
    для использования с SQLite датасетами.
    """

    def __init__(
        self,
        image_paths: list[str] | None = None,
        model_id:    str = "",
        transform:   transforms.Compose | None = None,
        _samples:    list | None = None,   # готовые (path, label_idx, bbox|None)
    ):
        self.model_id  = model_id
        self.transform = transform
        self.classes   = MODEL_DEFS.get(model_id, {}).get("classes", {})
        self.class_to_idx = {k: i for i, k in enumerate(sorted(self.classes))}

        if _samples is not None:
            self.samples = _samples
        else:
            self.samples: list[tuple[str, int, ann_io.BBox | None]] = []
            for path in (image_paths or []):
                item = self._resolve(path)
                if item is not None:
                    self.samples.append(item)

    # Позволяет использовать как Subset, так и прямой SpaceDataset в тренере
    @property
    def dataset(self):
        return self

    def _resolve(self, path: str):
        ia = ann_io.load(path)
        # Проверка аннотаций
        for a in ia.annotations:
            if a.label in self.class_to_idx:
                return (path, self.class_to_idx[a.label], a.bbox)
        # Резервный вариант по имени папки
        folder_name = Path(path).parent.name.lower()
        for key in self.class_to_idx:
            if key.lower() in folder_name or folder_name in key.lower():
                return (path, self.class_to_idx[key], None)
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, bbox = self.samples[idx]
        img = Image.open(path).convert("RGB")

        if bbox is not None:
            # Вырезать область по bounding box с небольшим отступом
            x, y, w, h = bbox.x, bbox.y, bbox.w, bbox.h
            margin = int(min(w, h) * 0.05)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(img.width,  x + w + margin)
            y2 = min(img.height, y + h + margin)
            img = img.crop((x1, y1, x2, y2))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    @property
    def class_names(self) -> list[str]:
        return [k for k in sorted(self.classes)]


# ── Фабрика ───────────────────────────────────────────────────────────────────

def build_datasets_from_db(
    db_path:    str,
    model_id:   str,
    image_size: tuple = DEFAULT_IMAGE_SIZE,
) -> tuple["SpaceDataset", "SpaceDataset"]:
    """Строит датасеты train/val из SQLite файла датасета (созданного вкладкой БД)."""
    import sqlite3

    classes = MODEL_DEFS[model_id]["classes"]
    class_to_idx = {k: i for i, k in enumerate(sorted(classes))}

    conn = sqlite3.connect(db_path)
    train_samples: list = []
    val_samples:   list = []

    for split, target in [("train", train_samples),
                          ("val",   val_samples),
                          ("test",  val_samples)]:
        rows = conn.execute(
            """
            SELECT i.path, a.label, a.bbox_x, a.bbox_y, a.bbox_w, a.bbox_h
            FROM images i
            JOIN annotations a ON a.image_id = i.id
            WHERE i.split = ?
            """,
            (split,)
        ).fetchall()
        for path, label, bx, by, bw, bh in rows:
            if label not in class_to_idx:
                continue
            bbox = ann_io.BBox(int(bx), int(by), int(bw), int(bh)) if bx is not None else None
            target.append((path, class_to_idx[label], bbox))

    conn.close()

    if not train_samples:
        raise ValueError(
            f"В базе нет обучающих примеров для модели '{model_id}'.\n"
            "Убедитесь, что датасет создан с аннотациями для этой модели."
        )
    if not val_samples:
        # резервный вариант: использовать 20% от обучающей выборки как val
        n = max(1, len(train_samples) // 5)
        val_samples = train_samples[-n:]
        train_samples = train_samples[:-n]

    train_ds = SpaceDataset(
        model_id=model_id,
        transform=get_train_transforms(image_size),
        _samples=train_samples,
    )
    val_ds = SpaceDataset(
        model_id=model_id,
        transform=get_val_transforms(image_size),
        _samples=val_samples,
    )
    return train_ds, val_ds


def build_datasets(
    folder:    str,
    model_id:  str,
    val_split: float = 0.2,
    image_size: tuple = DEFAULT_IMAGE_SIZE,
) -> tuple[SpaceDataset, SpaceDataset]:
    """
    Строит обучающий и валидационный датасеты из папки с изображениями ИЛИ файла .db.
    Возвращает (train_dataset, val_dataset).
    """
    if folder.lower().endswith((".db", ".sqlite")):
        return build_datasets_from_db(folder, model_id, image_size)

    image_paths = collect_images(folder)
    if not image_paths:
        raise ValueError(f"No images found in: {folder}")

    full = SpaceDataset(
        image_paths=image_paths,
        model_id=model_id,
        transform=get_train_transforms(image_size),
    )
    if len(full) == 0:
        raise ValueError(
            f"No annotated images found for model '{model_id}'. "
            "Make sure images have .json annotations or are in labelled subfolders."
        )

    n_val   = max(1, int(len(full) * val_split))
    n_train = len(full) - n_val

    train_ds, val_ds = random_split(
        full, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    # Применить val-преобразования к валидационной выборке
    val_ds.dataset = SpaceDataset(
        image_paths=[full.samples[i][0] for i in val_ds.indices],
        model_id=model_id,
        transform=get_val_transforms(image_size),
    )
    val_ds.indices = list(range(len(val_ds.dataset)))

    return train_ds, val_ds
