"""Утилиты для работы с файлами и изображениями."""

from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def collect_images(folder: str) -> list[str]:
    """Рекурсивно собирает все файлы изображений в папке."""
    result = []
    for p in sorted(Path(folder).rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            result.append(str(p))
    return result


def collect_images_flat(folder: str) -> list[str]:
    """Собирает изображения только в указанной папке (без рекурсии)."""
    result = []
    for p in sorted(Path(folder).iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            result.append(str(p))
    return result


def image_size(path: str) -> tuple[int, int]:
    """Возвращает (ширину, высоту) изображения без загрузки всех пикселей."""
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (w, h)
