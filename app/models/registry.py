"""
Реестр моделей — управляет сохранением и загрузкой весов PyTorch-моделей
для каждого model_id, определённого в config.MODEL_DEFS.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from config import MODEL_DEFS, MODELS_DIR
from app.models.torch_model import build_model, ClassificationModel, get_device


class ModelRegistry:
    """Реестр в виде синглтона; один экземпляр используется во всём приложении."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.dir = models_dir
        self.dir.mkdir(exist_ok=True)
        self._loaded: dict[str, ClassificationModel] = {}

    # ── Пути ───────────────────────────────────────────────────────────────────

    def _weights_path(self, model_id: str) -> Path:
        return self.dir / f"{model_id}.pt"

    def _meta_path(self, model_id: str) -> Path:
        return self.dir / f"{model_id}.json"

    # ── Сохранение / загрузка ──────────────────────────────────────────────────

    def save(self, model_id: str, model: ClassificationModel, meta: dict | None = None) -> None:
        """Сохраняет веса модели и дополнительные метаданные."""
        torch.save(model.state_dict(), self._weights_path(model_id))
        info = {
            "model_id":    model_id,
            "num_classes": model.num_classes,
            "classes":     MODEL_DEFS.get(model_id, {}).get("classes", {}),
        }
        if meta:
            info.update(meta)
        with open(self._meta_path(model_id), "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        self._loaded[model_id] = model

    def load(self, model_id: str, device: torch.device | None = None) -> ClassificationModel | None:
        """Загружает модель с диска. Возвращает None, если файл не найден."""
        w = self._weights_path(model_id)
        if not w.exists():
            return None
        if device is None:
            device = get_device()
        mdef = MODEL_DEFS.get(model_id, {})
        num_classes = len(mdef.get("classes", {})) or 2
        model = build_model(num_classes=num_classes)
        model.load_state_dict(torch.load(w, map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        self._loaded[model_id] = model
        return model

    def get(self, model_id: str) -> ClassificationModel | None:
        """Возвращает кэшированную модель или загружает её с диска."""
        if model_id not in self._loaded:
            return self.load(model_id)
        return self._loaded[model_id]

    def is_trained(self, model_id: str) -> bool:
        return self._weights_path(model_id).exists()

    def list_trained(self) -> list[str]:
        return [mid for mid in MODEL_DEFS if self.is_trained(mid)]

    def get_meta(self, model_id: str) -> dict:
        p = self._meta_path(model_id)
        if not p.exists():
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def invalidate(self, model_id: str):
        self._loaded.pop(model_id, None)
