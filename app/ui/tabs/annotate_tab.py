"""
Вкладка аннотации — ручная разметка изображений.

Макет:
  [Панель инструментов: загрузить папку / файлы / назад / вперёд / сохранить / авто-сохранение]
  ┌─────────────────────────┬─────────────────────┐
  │  ImageCanvas            │  Боковая панель      │
  │                         │  • список меток      │
  │                         │  • список моделей    │
  │                         │  • список аннотаций  │
  │                         │  • применить/удалить │
  │                         │  • список изображений│
  └─────────────────────────┴─────────────────────┘
  [Строка состояния]
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFormLayout, QHBoxLayout, QInputDialog,
    QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QSplitter, QVBoxLayout, QWidget,
)

from app.ui.widgets.canvas import ImageCanvas
from app.utils import annotation as ann_io
from app.utils.annotation import Annotation, BBox
from app.utils.db_utils import read_table, TableData
from app.utils.file_utils import IMAGE_EXTS
from config import ALL_LABELS, MODEL_DEFS


# ── Диалог выбора колонок ──────────────────────────────────────────────────────

class _ColPickerDialog(QDialog):
    """Выбор колонки с именем файла и колонки с меткой."""

    def __init__(self, tbl: "TableData", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Настройка колонок таблицы")
        self.setMinimumWidth(400)

        from PySide6.QtWidgets import QCheckBox, QGroupBox
        layout = QFormLayout(self)

        # Информация: количество строк
        info = QLabel(f"Загружено строк: {len(tbl.rows)}  |  Колонок: {len(tbl.headers)}")
        info.setStyleSheet("color:#6272a4; font-size:11px;")
        layout.addRow(info)

        # Предпросмотр первой строки данных
        if tbl.rows:
            first = "  |  ".join(str(v) for v in tbl.rows[0][:6])
            prev_lbl = QLabel(f"Первая строка данных: {first}")
            prev_lbl.setStyleSheet("color:#f1fa8c; font-size:10px;")
            prev_lbl.setWordWrap(True)
            layout.addRow(prev_lbl)

        self._file_cb  = QComboBox()
        self._label_cb = QComboBox()
        self._file_cb.addItems(tbl.headers)
        self._label_cb.addItems(tbl.headers)

        # Автоопределение колонок
        for i, h in enumerate(tbl.headers):
            hl = h.lower()
            if any(k in hl for k in ("file", "name", "path", "image", "фото", "снимок", "имя")):
                self._file_cb.setCurrentIndex(i)
                break
        for i, h in enumerate(tbl.headers):
            hl = h.lower()
            if any(k in hl for k in ("label", "class", "метка", "тип", "категория", "объект", "класс")):
                self._label_cb.setCurrentIndex(i)
                break
        # Если обе колонки угаданы как одна, установить метку на вторую колонку
        if (self._file_cb.currentIndex() == self._label_cb.currentIndex()
                and len(tbl.headers) > 1):
            self._label_cb.setCurrentIndex(1)

        layout.addRow("Колонка с именем файла:", self._file_cb)
        layout.addRow("Колонка с меткой/классом:", self._label_cb)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addRow(btns)

    @property
    def file_col(self) -> str:
        return self._file_cb.currentText()

    @property
    def label_col(self) -> str:
        return self._label_cb.currentText()


class AnnotateTab(QWidget):
    images_changed = Signal(list)   # передаёт текущий список изображений

    def __init__(self, parent=None):
        super().__init__(parent)
        self._images:        list[str]        = []
        self._cur_idx:       int              = -1
        self._ia:            ann_io.ImageAnnotation | None = None
        self._autosave:      bool             = False
        # сопоставление таблицы: stem → label_key
        self._table_map:     dict[str, str]   = {}
        self._workers:       list             = []   # защита от GC


        self._build_ui()

    # ── Интерфейс ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 4)
        root.setSpacing(4)

        # Панель инструментов
        tb = QHBoxLayout()
        tb.setSpacing(6)
        self._btn_folder = QPushButton("📂 Папка")
        self._btn_files  = QPushButton("🖼 Файлы")
        self._btn_table  = QPushButton("📊 Таблица")
        self._btn_table.setToolTip("Загрузить таблицу для автоподстановки меток")
        self._table_lbl  = QLabel()
        self._table_lbl.setStyleSheet("color:#50fa7b; font-size:10px;")
        self._btn_prev   = QPushButton("◀")
        self._btn_next   = QPushButton("▶")
        self._btn_save   = QPushButton("💾 Сохранить")
        self._btn_save.setStyleSheet(
            "QPushButton{background:#27ae60;color:white;font-weight:bold;padding:4px 10px;}"
            "QPushButton:disabled{background:#555;}"
        )
        self._btn_autosave = QPushButton("Авто-сохр: ВЫКЛ")
        self._btn_autosave.setCheckable(True)

        self._counter = QLabel("0 / 0")
        for w in (self._btn_folder, self._btn_files, self._btn_table,
                  self._btn_prev, self._btn_next, self._btn_save, self._btn_autosave):
            w.setFixedHeight(28)
            tb.addWidget(w)
        tb.addWidget(self._table_lbl)
        tb.addStretch()
        tb.addWidget(self._counter)
        root.addLayout(tb)

        # Основной разделитель
        spl = QSplitter(Qt.Horizontal)

        # Холст
        self._canvas = ImageCanvas()
        spl.addWidget(self._canvas)

        # Боковая панель
        side = QWidget()
        side.setMinimumWidth(210)
        side.setMaximumWidth(280)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(4, 4, 4, 4)
        sl.setSpacing(4)

        sl.addWidget(QLabel("Модель:"))
        self._model_combo = QComboBox()
        self._model_combo.addItem("— любая —", "")
        for mid, mdef in MODEL_DEFS.items():
            self._model_combo.addItem(mdef["name"], mid)
        sl.addWidget(self._model_combo)

        sl.addWidget(QLabel("Метка:"))
        self._label_combo = QComboBox()
        self._label_combo.setEditable(True)
        self._update_label_combo()
        sl.addWidget(self._label_combo)

        btn_add_lbl = QPushButton("+ Новая метка")
        btn_add_lbl.setFixedHeight(24)
        sl.addWidget(btn_add_lbl)

        sl.addWidget(QLabel("Аннотации:"))
        self._ann_list = QListWidget()
        self._ann_list.setSelectionMode(QAbstractItemView.SingleSelection)
        sl.addWidget(self._ann_list, stretch=2)

        ann_btns = QHBoxLayout()
        self._btn_apply   = QPushButton("✓ Применить")
        self._btn_del_ann = QPushButton("✕ Удалить")
        self._btn_del_ann.setStyleSheet("color:#e74c3c;")
        ann_btns.addWidget(self._btn_apply)
        ann_btns.addWidget(self._btn_del_ann)
        sl.addLayout(ann_btns)

        sl.addWidget(QLabel("Изображения:"))
        self._img_list = QListWidget()
        self._img_list.setMaximumHeight(140)
        sl.addWidget(self._img_list)

        spl.addWidget(side)
        spl.setStretchFactor(0, 4)
        spl.setStretchFactor(1, 1)
        root.addWidget(spl)

        # Строка состояния
        self._status = QLabel("Готов")
        self._status.setStyleSheet("color:#6272a4; font-size:11px;")
        root.addWidget(self._status)

        # Подключения сигналов
        self._btn_folder.clicked.connect(self._on_load_folder)
        self._btn_files.clicked.connect(self._on_load_files)
        self._btn_table.clicked.connect(self._on_load_table)
        self._btn_prev.clicked.connect(self._on_prev)
        self._btn_next.clicked.connect(self._on_next)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_autosave.toggled.connect(self._on_autosave)
        btn_add_lbl.clicked.connect(self._on_add_label)
        self._btn_apply.clicked.connect(self._on_apply_label)
        self._btn_del_ann.clicked.connect(self._on_del_ann)

        self._canvas.annotation_added.connect(self._on_ann_added)
        self._canvas.annotation_selected.connect(self._on_ann_selected)
        self._canvas.annotations_changed.connect(self._refresh_ann_list)

        self._ann_list.currentRowChanged.connect(self._canvas.set_selected)
        self._img_list.currentRowChanged.connect(self._load_by_index)
        self._model_combo.currentIndexChanged.connect(self._update_label_combo)

    # ── Список меток ───────────────────────────────────────────────────────────

    def _update_label_combo(self, _=None):
        mid = self._model_combo.currentData()
        self._label_combo.clear()
        if mid:
            labels = MODEL_DEFS[mid]["classes"]
        else:
            labels = ALL_LABELS
        for key, display in labels.items():
            self._label_combo.addItem(display, key)

    # ── Загрузка таблицы ───────────────────────────────────────────────────────

    def _on_load_table(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить таблицу", "",
            "Tables (*.xlsx *.xls *.csv *.tsv *.json *.db *.sqlite);;All (*)"
        )
        if not path:
            return
        try:
            tbl = read_table(path)
        except Exception as e:
            QMessageBox.warning(self, "Ошибка загрузки таблицы", str(e))
            return

        if not tbl.headers:
            QMessageBox.warning(self, "Ошибка", "Таблица пустая или не распознана")
            return

        dlg = _ColPickerDialog(tbl, self)
        if dlg.exec() != QDialog.Accepted:
            return

        file_col  = dlg.file_col
        label_col = dlg.label_col

        # Построение обратного словаря: display_name.lower() / key.lower() → label_key
        reverse: dict[str, str] = {}
        for key, display in ALL_LABELS.items():
            reverse[key.lower()]     = key
            reverse[display.lower()] = key

        # Построение словаря stem → label_key
        try:
            fc_idx = tbl.headers.index(file_col)
            lc_idx = tbl.headers.index(label_col)
        except ValueError:
            QMessageBox.warning(self, "Ошибка", "Колонка не найдена")
            return

        self._table_map = {}
        for row in tbl.rows:
            stem = os.path.splitext(os.path.basename(str(row[fc_idx]).strip()))[0].lower()
            raw  = str(row[lc_idx]).strip()
            key  = reverse.get(raw.lower(), raw)
            if stem:
                self._table_map[stem] = key

        matched = len(self._table_map)
        fname   = os.path.basename(path)
        self._table_lbl.setText(f"📊 {fname}  ({matched} записей)")
        self._set_status(f"Таблица загружена: {matched} записей сопоставлено")
        self._apply_table_label()

    def _apply_table_label(self):
        """Если текущее изображение найдено в таблице, установить соответствующую метку."""
        if not self._table_map or self._cur_idx < 0:
            return
        path = self._images[self._cur_idx]
        stem = os.path.splitext(os.path.basename(path))[0].lower()
        key  = self._table_map.get(stem)
        if not key:
            return

        # Найти совпадающий элемент в списке меток
        for i in range(self._label_combo.count()):
            if self._label_combo.itemData(i) == key:
                self._label_combo.setCurrentIndex(i)
                self._set_status(
                    f"{os.path.basename(path)}  →  метка из таблицы: "
                    f"{self._label_combo.itemText(i)}"
                )
                return
        # Ключ не найден в комбо (пользовательская метка из таблицы) — добавить временно
        display = ALL_LABELS.get(key, key)
        self._label_combo.addItem(display, key)
        self._label_combo.setCurrentIndex(self._label_combo.count() - 1)
        self._set_status(
            f"{os.path.basename(path)}  →  метка из таблицы: {display}"
        )

    # ── Загрузка ───────────────────────────────────────────────────────────────

    def _on_load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку с изображениями")
        if not folder:
            return
        files = sorted(
            str(p) for p in Path(folder).iterdir()
            if p.suffix.lower() in IMAGE_EXTS
        )
        if not files:
            QMessageBox.information(self, "Папка пуста", "Изображений не найдено")
            return
        self.load_images(files)

    def _on_load_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите изображения", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if files:
            self.load_images(files)

    def load_images(self, paths: list[str]):
        self._images = list(paths)
        self._img_list.clear()
        for p in self._images:
            self._img_list.addItem(os.path.basename(p))
        self.images_changed.emit(self._images)
        if self._images:
            self._load_by_index(0)

    def _load_by_index(self, idx: int):
        try:
            idx = int(idx)
        except Exception:
            return
        if not (0 <= idx < len(self._images)):
            return

        if self._autosave and self._ia is not None:
            self._do_save(silent=True)

        self._cur_idx = idx
        self._img_list.setCurrentRow(idx)
        path = self._images[idx]

        pix = QPixmap(path)
        if pix.isNull():
            self._set_status(f"Ошибка загрузки: {os.path.basename(path)}", error=True)
            return

        self._canvas.set_image(pix)
        self._ia = ann_io.load(path)
        self._ia.width  = pix.width()
        self._ia.height = pix.height()
        self._canvas.load_annotations(self._ia.annotations)
        self._refresh_ann_list()
        self._update_counter()
        self._set_status(os.path.basename(path))
        self._apply_table_label()   # автоустановка метки из таблицы при наличии

    # ── Навигация ──────────────────────────────────────────────────────────────

    def _on_prev(self):
        if self._images:
            self._load_by_index(max(0, self._cur_idx - 1))

    def _on_next(self):
        if self._images:
            self._load_by_index(min(len(self._images) - 1, self._cur_idx + 1))

    # ── Сохранение ─────────────────────────────────────────────────────────────

    def _on_save(self):
        self._do_save(silent=False)

    def _do_save(self, silent: bool = False):
        if self._cur_idx < 0 or self._ia is None:
            if not silent:
                QMessageBox.warning(self, "Ошибка", "Нет загруженного изображения")
            return
        self._ia.annotations = self._canvas.get_annotations()
        try:
            ann_io.save(self._ia)
            if not silent:
                self._set_status(f"Сохранено: {os.path.basename(self._ia.json_path)}")
                QMessageBox.information(self, "Сохранено",
                                        f"Аннотация сохранена:\n{self._ia.json_path}")
        except Exception as e:
            if not silent:
                QMessageBox.warning(self, "Ошибка сохранения", str(e))

    def _on_autosave(self, checked: bool):
        self._autosave = checked
        self._btn_autosave.setText(f"Авто-сохр: {'ВКЛ' if checked else 'ВЫКЛ'}")

    # ── Список аннотаций ───────────────────────────────────────────────────────

    def _refresh_ann_list(self):
        anns = self._canvas.get_annotations()
        self._ann_list.blockSignals(True)
        self._ann_list.clear()
        for i, a in enumerate(anns):
            has_bbox = a.bbox is not None
            bbox_str = (f"  ({a.bbox.x},{a.bbox.y}) {a.bbox.w}×{a.bbox.h}"
                        if has_bbox else "  [нет bbox]")
            item = QListWidgetItem(f"[{i}] {a.display_name}{bbox_str}")
            from app.ui.widgets.canvas import _COLORS
            c = _COLORS.get(a.label)
            item.setForeground(QBrush(c))
            self._ann_list.addItem(item)
        sel = self._canvas._selected
        if 0 <= sel < self._ann_list.count():
            self._ann_list.setCurrentRow(sel)
        self._ann_list.blockSignals(False)

    def _on_ann_added(self, idx: int):
        # Применить текущую выбранную метку
        lbl_key  = self._label_combo.currentData() or self._label_combo.currentText()
        model_id = self._model_combo.currentData() or ann_io.label_to_model(lbl_key)
        self._canvas.update_label(idx, lbl_key, model_id)
        self._refresh_ann_list()

    def _on_ann_selected(self, idx: int):
        self._ann_list.blockSignals(True)
        if 0 <= idx < self._ann_list.count():
            self._ann_list.setCurrentRow(idx)
        self._ann_list.blockSignals(False)

    def _on_apply_label(self):
        idx = self._ann_list.currentRow()
        if idx < 0:
            return
        lbl_key  = self._label_combo.currentData() or self._label_combo.currentText()
        model_id = self._model_combo.currentData() or ann_io.label_to_model(lbl_key)
        self._canvas.update_label(idx, lbl_key, model_id)
        self._refresh_ann_list()

    def _on_del_ann(self):
        idx = self._ann_list.currentRow()
        if idx >= 0:
            self._canvas.delete_annotation(idx)
            self._refresh_ann_list()

    def _on_add_label(self):
        text, ok = QInputDialog.getText(self, "Новая метка", "Название (ключ):")
        if ok and text.strip():
            self._label_combo.addItem(text.strip(), text.strip())
            self._label_combo.setCurrentText(text.strip())

    # ── Вспомогательные методы ─────────────────────────────────────────────────

    def _update_counter(self):
        total = len(self._images)
        cur   = self._cur_idx + 1 if self._cur_idx >= 0 else 0
        self._counter.setText(f"{cur} / {total}")

    def _set_status(self, msg: str, error: bool = False):
        color = "#e74c3c" if error else "#6272a4"
        self._status.setStyleSheet(f"color:{color}; font-size:11px;")
        self._status.setText(msg)

    # ── Публичный API (для главного окна / других вкладок) ─────────────────────

    @property
    def images(self) -> list[str]:
        return self._images

    @property
    def current_image(self) -> str | None:
        if 0 <= self._cur_idx < len(self._images):
            return self._images[self._cur_idx]
        return None
