"""
Глобальная конфигурация приложения.

3 плоских модели:
  station      — Оборудование | Космонавт
  outside      — Космический объект | Планета Земля
  earth_surface — Океан | Суша | Городская застройка | Облака
"""

from pathlib import Path

# ── Пути ───────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent
MODELS_DIR = ROOT_DIR / "saved_models"
DATA_DIR   = ROOT_DIR / "data"

MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# ── Реестр моделей ─────────────────────────────────────────────────────────────
MODEL_DEFS: dict[str, dict] = {
    "station": {
        "name":        "Станция",
        "description": "Внутри станции — оборудование или космонавты",
        "classes": {
            "equipment": "Оборудование",
            "cosmonaut": "Космонавт",
        },
        "color": "#e67e22",
        "icon":  "🛸",
    },
    "outside": {
        "name":        "Снаружи",
        "description": "Снаружи станции — космические объекты или Земля",
        "classes": {
            "space_object": "Космический объект",
            "earth":        "Планета Земля",
        },
        "color": "#2979ff",
        "icon":  "🌌",
    },
    "earth_surface": {
        "name":        "Земная поверхность",
        "description": "Типы поверхности Земли, видимые из космоса",
        "classes": {
            "ocean":  "Океан / Водоём",
            "land":   "Суша / Природа",
            "urban":  "Городская застройка",
            "clouds": "Облака / Атмосфера",
        },
        "color": "#27ae60",
        "icon":  "🌍",
    },
}

# Все доступные метки по всем моделям (ключ → отображаемое имя)
ALL_LABELS: dict[str, str] = {}
for _m in MODEL_DEFS.values():
    ALL_LABELS.update(_m["classes"])

# ── Параметры обучения по умолчанию ───────────────────────────────────────────
DEFAULT_IMAGE_SIZE  = (224, 224)
DEFAULT_BATCH_SIZE  = 16
DEFAULT_EPOCHS      = 20
DEFAULT_LR          = 1e-4
DEFAULT_VAL_SPLIT   = 0.2
DEFAULT_BACKBONE    = "resnet18"

# ── Интерфейс ─────────────────────────────────────────────────────────────────
APP_TITLE   = "Space Vision — Классификация фотоснимков"
APP_VERSION = "2.0.0"

LABEL_COLORS = [
    "#e67e22", "#2979ff", "#27ae60", "#d500f9",
    "#00bcd4", "#f50057", "#ffea00", "#00c853",
]
