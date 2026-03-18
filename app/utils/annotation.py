"""
Ввод/вывод аннотаций.

Формат файла (JSON, хранится рядом с каждым изображением):
{
  "image":  "photo.jpg",
  "width":  1920,
  "height": 1080,
  "annotations": [
    {
      "id":      1,
      "label":   "cosmonaut",        # ключ из config.ALL_LABELS
      "model":   "station_detail",   # model_id, которому принадлежит метка
      "bbox":    {"x":100,"y":200,"w":150,"h":300} | null,
      "source":  "manual" | "table" | "model",
      "confidence": null | float
    }
  ]
}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from config import MODEL_DEFS, ALL_LABELS


# ─── Классы данных ─────────────────────────────────────────────────────────────

@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    def as_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}

    @staticmethod
    def from_dict(d: dict) -> "BBox":
        return BBox(int(d["x"]), int(d["y"]), int(d["w"]), int(d["h"]))

    def clamp(self, img_w: int, img_h: int) -> "BBox":
        x = max(0, min(img_w - 1, self.x))
        y = max(0, min(img_h - 1, self.y))
        w = max(1, min(img_w - x, self.w))
        h = max(1, min(img_h - y, self.h))
        return BBox(x, y, w, h)


@dataclass
class Annotation:
    label:      str
    model:      str  = ""
    bbox:       Optional[BBox] = None
    source:     str  = "manual"
    confidence: Optional[float] = None
    id:         int  = 0

    @property
    def display_name(self) -> str:
        return ALL_LABELS.get(self.label, self.label)

    def as_dict(self) -> dict:
        return {
            "id":         self.id,
            "label":      self.label,
            "model":      self.model,
            "bbox":       self.bbox.as_dict() if self.bbox else None,
            "source":     self.source,
            "confidence": self.confidence,
        }

    @staticmethod
    def from_dict(d: dict) -> "Annotation":
        bbox = BBox.from_dict(d["bbox"]) if d.get("bbox") else None
        return Annotation(
            label      = d.get("label", ""),
            model      = d.get("model", ""),
            bbox       = bbox,
            source     = d.get("source", "manual"),
            confidence = d.get("confidence"),
            id         = d.get("id", 0),
        )


@dataclass
class ImageAnnotation:
    image_path: str
    width:      int = 0
    height:     int = 0
    annotations: list[Annotation] = field(default_factory=list)

    @property
    def json_path(self) -> str:
        return os.path.splitext(self.image_path)[0] + ".json"

    def next_id(self) -> int:
        return (max((a.id for a in self.annotations), default=0) + 1)


# ─── Ввод/вывод ────────────────────────────────────────────────────────────────

def load(image_path: str) -> ImageAnnotation:
    """Загружает JSON аннотации для image_path. Возвращает пустой объект, если файл отсутствует."""
    ia = ImageAnnotation(image_path=image_path)
    json_path = os.path.splitext(image_path)[0] + ".json"
    if not os.path.exists(json_path):
        return ia
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ia.width  = data.get("width", 0)
        ia.height = data.get("height", 0)
        for i, a in enumerate(data.get("annotations", [])):
            ann = Annotation.from_dict(a)
            if ann.id == 0:
                ann.id = i + 1
            ia.annotations.append(ann)
    except Exception:
        pass
    return ia


def save(ia: ImageAnnotation) -> None:
    """Записывает аннотацию в JSON-файл рядом с изображением."""
    data = {
        "image":       os.path.basename(ia.image_path),
        "width":       ia.width,
        "height":      ia.height,
        "annotations": [a.as_dict() for a in ia.annotations],
    }
    with open(ia.json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ─── Вспомогательные функции ───────────────────────────────────────────────────

def label_to_model(label: str) -> str:
    """Возвращает model_id, которому принадлежит данная метка, или пустую строку."""
    for model_id, mdef in MODEL_DEFS.items():
        if label in mdef["classes"]:
            return model_id
    return ""


def model_for_label(label: str) -> str:
    return label_to_model(label)
