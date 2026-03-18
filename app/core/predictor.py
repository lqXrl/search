"""
Предиктор — выполняет инференс на одном изображении с помощью загруженной модели.
"""

from __future__ import annotations

import torch
from PIL import Image

from app.core.dataset import get_val_transforms
from app.models.torch_model import ClassificationModel, get_device
from config import MODEL_DEFS, DEFAULT_IMAGE_SIZE


class Predictor:
    def __init__(self, model: ClassificationModel, model_id: str,
                 device: torch.device | None = None):
        self.model    = model
        self.model_id = model_id
        self.device   = device or get_device()
        self.model.to(self.device)
        self.model.eval()

        mdef = MODEL_DEFS.get(model_id, {})
        self.class_keys    = sorted(mdef.get("classes", {}).keys())
        self.class_labels  = {k: mdef["classes"][k] for k in self.class_keys}
        self.class_to_idx  = {k: i for i, k in enumerate(self.class_keys)}
        self.idx_to_class  = {i: k for k, i in self.class_to_idx.items()}

        self._transform = get_val_transforms(DEFAULT_IMAGE_SIZE)

    def predict(self, image_path: str) -> dict:
        """
        Возвращает {
          'class_key':   str,
          'class_label': str,
          'confidence':  float,
          'probs':       {class_key: float, ...}
        }
        """
        img  = Image.open(image_path).convert("RGB")
        return self.predict_pil(img)

    def predict_pil(self, img: Image.Image) -> dict:
        tensor = self._transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()

        idx        = int(torch.tensor(probs).argmax())
        class_key  = self.idx_to_class.get(idx, str(idx))
        confidence = probs[idx]
        probs_dict = {self.idx_to_class.get(i, str(i)): round(p, 4)
                      for i, p in enumerate(probs)}

        return {
            "class_key":   class_key,
            "class_label": self.class_labels.get(class_key, class_key),
            "confidence":  round(confidence, 4),
            "probs":       probs_dict,
        }
