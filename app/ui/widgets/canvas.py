"""
ImageCanvas — виджет холста для аннотации.

Хранит все аннотации в пиксельных координатах ОРИГИНАЛЬНОГО ИЗОБРАЖЕНИЯ.
Конвертирует в экранные координаты только для отрисовки.
Обрабатывает: рисование, выбор, удаление, контекстное меню.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QPoint, QRect, Signal
from PySide6.QtGui import (
    QColor, QFont, QMouseEvent, QPainter, QPen, QPixmap, QBrush,
    QAction, QPainterPath
)
from PySide6.QtWidgets import QLabel, QMenu, QInputDialog

from app.utils.annotation import Annotation, BBox
from config import LABEL_COLORS


# ── Реестр цветов ──────────────────────────────────────────────────────────────

class _ColorReg:
    def __init__(self):
        self._map: dict[str, QColor] = {}
        self._idx = 0

    def get(self, label: str) -> QColor:
        if label not in self._map:
            hex_c = LABEL_COLORS[self._idx % len(LABEL_COLORS)]
            self._map[label] = QColor(hex_c)
            self._idx += 1
        return self._map[label]


_COLORS = _ColorReg()


# ── Холст ──────────────────────────────────────────────────────────────────────

class ImageCanvas(QLabel):
    """
    Отображает изображение; позволяет рисовать, выбирать и удалять bounding box'ы.
    Аннотации хранятся в пиксельных координатах ОРИГИНАЛЬНОГО изображения.
    """

    annotation_added    = Signal(int)    # индекс новой аннотации
    annotation_selected = Signal(int)    # индекс выбранной аннотации
    annotations_changed = Signal()       # любое изменение

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#1e1e2e; border:1px solid #44475a;")
        self.setText("Загрузите изображение")
        self.setFont(QFont("Arial", 14))

        self._orig: QPixmap | None = None
        self._orig_w = 0
        self._orig_h = 0

        self._annotations: list[Annotation] = []
        self._selected: int = -1

        self._drawing     = False
        self._drag_start: QPoint | None = None
        self._cur_rect:   QRect  | None = None

    # ── Публичный API ──────────────────────────────────────────────────────────

    def set_image(self, pixmap: QPixmap):
        self._orig   = pixmap
        self._orig_w = pixmap.width()
        self._orig_h = pixmap.height()
        self._annotations.clear()
        self._selected = -1
        self._refresh()

    def load_annotations(self, annotations: list[Annotation]):
        self._annotations = list(annotations)
        self._selected = -1
        self.update()

    def get_annotations(self) -> list[Annotation]:
        return list(self._annotations)

    def set_selected(self, idx: int):
        self._selected = idx
        self.update()

    def delete_annotation(self, idx: int):
        if 0 <= idx < len(self._annotations):
            del self._annotations[idx]
            self._selected = min(self._selected, len(self._annotations) - 1)
            self.update()
            self.annotations_changed.emit()

    def update_label(self, idx: int, label: str, model: str = ""):
        if 0 <= idx < len(self._annotations):
            self._annotations[idx].label = label
            if model:
                self._annotations[idx].model = model
            self.update()
            self.annotations_changed.emit()

    def clear_annotations(self):
        self._annotations.clear()
        self._selected = -1
        self.update()
        self.annotations_changed.emit()

    # ── Преобразование координат ───────────────────────────────────────────────

    def _render_info(self) -> tuple[float, float, float]:
        """Возвращает (смещение_x, смещение_y, масштаб) для текущего размера виджета."""
        if not self._orig_w or not self._orig_h:
            return 0.0, 0.0, 1.0
        lw, lh = self.width(), self.height()
        scale  = min(lw / self._orig_w, lh / self._orig_h)
        rw, rh = self._orig_w * scale, self._orig_h * scale
        return (lw - rw) / 2, (lh - rh) / 2, scale

    def _s2i(self, sx: float, sy: float) -> tuple[float, float]:
        ox, oy, s = self._render_info()
        return (sx - ox) / s, (sy - oy) / s

    def _i2s(self, ix: float, iy: float) -> tuple[float, float]:
        ox, oy, s = self._render_info()
        return ix * s + ox, iy * s + oy

    def _ann_rect(self, ann: Annotation) -> QRect:
        if ann.bbox is None:
            return QRect()
        sx, sy = self._i2s(ann.bbox.x, ann.bbox.y)
        ex, ey = self._i2s(ann.bbox.x + ann.bbox.w, ann.bbox.y + ann.bbox.h)
        return QRect(int(sx), int(sy), int(ex - sx), int(ey - sy))

    def _hit_test(self, pos: QPoint) -> int:
        for i in reversed(range(len(self._annotations))):
            if self._ann_rect(self._annotations[i]).contains(pos):
                return i
        return -1

    # ── Отображение ────────────────────────────────────────────────────────────

    def _refresh(self):
        if self._orig is None:
            super().setPixmap(QPixmap())
            self.setText("Загрузите изображение")
            return
        self.setText("")
        scaled = self._orig.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        super().setPixmap(scaled)
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._orig:
            self._refresh()

    # ── Мышь ───────────────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent):
        if self._orig is None:
            return
        pos = event.position().toPoint()

        if event.button() == Qt.LeftButton:
            hit = self._hit_test(pos)
            if hit >= 0:
                self._selected = hit
                self.annotation_selected.emit(hit)
                self.update()
            else:
                self._drawing   = True
                self._drag_start = pos
                self._cur_rect  = None

        elif event.button() == Qt.RightButton:
            hit = self._hit_test(pos)
            if hit >= 0:
                self._show_ctx(event.globalPosition().toPoint(), hit)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drawing:
            self._cur_rect = QRect(self._drag_start,
                                   event.position().toPoint()).normalized()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self._drawing or event.button() != Qt.LeftButton:
            return
        self._drawing = False
        pos   = event.position().toPoint()
        srect = QRect(self._drag_start, pos).normalized()
        self._cur_rect = None

        if srect.width() < 5 or srect.height() < 5:
            self.update()
            return

        ix1, iy1 = self._s2i(srect.x(), srect.y())
        ix2, iy2 = self._s2i(srect.x() + srect.width(),
                              srect.y() + srect.height())

        ix1 = max(0.0, min(float(self._orig_w), ix1))
        iy1 = max(0.0, min(float(self._orig_h), iy1))
        ix2 = max(0.0, min(float(self._orig_w), ix2))
        iy2 = max(0.0, min(float(self._orig_h), iy2))

        if (ix2 - ix1) < 2 or (iy2 - iy1) < 2:
            self.update()
            return

        bbox = BBox(int(ix1), int(iy1), int(ix2 - ix1), int(iy2 - iy1))
        ann  = Annotation(label="object", bbox=bbox, source="manual")
        self._annotations.append(ann)
        self._selected = len(self._annotations) - 1
        self.update()
        self.annotation_added.emit(self._selected)
        self.annotations_changed.emit()

    # ── Контекстное меню ───────────────────────────────────────────────────────

    def _show_ctx(self, gpos: QPoint, idx: int):
        menu = QMenu(self)
        a_rename = QAction("✏  Переименовать метку", self)
        a_delete = QAction("✕  Удалить", self)
        menu.addAction(a_rename)
        menu.addSeparator()
        menu.addAction(a_delete)

        a_rename.triggered.connect(lambda: self._rename(idx))
        a_delete.triggered.connect(lambda: self.delete_annotation(idx))
        menu.exec(gpos)

    def _rename(self, idx: int):
        old = self._annotations[idx].label
        new, ok = QInputDialog.getText(self, "Метка", "Введите метку:", text=old)
        if ok and new.strip():
            self.update_label(idx, new.strip())

    # ── Отрисовка ──────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._orig is None:
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        font = QFont("Arial", 9, QFont.Bold)
        p.setFont(font)
        fm = p.fontMetrics()

        for i, ann in enumerate(self._annotations):
            if ann.bbox is None:
                continue
            rect   = self._ann_rect(ann)
            color  = _COLORS.get(ann.label)
            sel    = (i == self._selected)

            # Рамка
            pen = QPen(color, 3 if sel else 2)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(rect)

            # Угловые маркеры при выборе
            if sel:
                hs = 5
                p.setPen(Qt.NoPen)
                p.setBrush(QBrush(color))
                for cx, cy in [(rect.left(), rect.top()), (rect.right(), rect.top()),
                               (rect.left(), rect.bottom()), (rect.right(), rect.bottom())]:
                    p.drawEllipse(QPoint(cx, cy), hs, hs)

            # Бейдж с меткой
            disp_name = ann.display_name or ann.label
            if ann.confidence is not None:
                disp_name += f"  {ann.confidence:.0%}"
            tw = fm.horizontalAdvance(disp_name) + 10
            th = fm.height() + 4
            bx = rect.x()
            by = rect.y() - th - 2
            if by < 0:
                by = rect.y() + 2
            p.setBrush(QBrush(color))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(bx, by, tw, th, 3, 3)
            p.setPen(QPen(QColor(255, 255, 255)))
            p.drawText(bx + 5, by + th - 4, disp_name)

        # Текущий прямоугольник рисования
        if self._cur_rect:
            p.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(self._cur_rect)

        p.end()
