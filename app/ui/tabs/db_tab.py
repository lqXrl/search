"""
Вкладка БД — авторазметка из табличных данных.

Рабочий процесс:
  1. Загрузить таблицу (.xlsx / .csv / .json / .db)
  2. Выбрать «колонку с именем файла» и «колонку с меткой»
  3. Нажать «Сопоставить» — сопоставляет строки таблицы с загруженными изображениями
  4. Просмотреть результаты в цветовой кодировке
  5. Нажать «Применить аннотации» — создаёт/обновляет .json файлы
  6. Опционально: экспорт в CSV
"""

from __future__ import annotations

import json
import os
import random

from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from app.utils.db_utils import read_table, match_filenames, TableData, MatchResult
from app.utils import annotation as ann_io
from app.utils.file_utils import collect_images_flat


_GREEN = QColor(40, 167, 80, 60)
_RED   = QColor(220, 53, 69, 60)
_GRAY  = QColor(108, 117, 125, 40)


class _SplitDialog(QDialog):
    """Запрашивает у пользователя процентное соотношение train/val/test."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Разбивка датасета")
        self.setMinimumWidth(280)

        form = QFormLayout()

        self._train = QSpinBox()
        self._train.setRange(1, 98)
        self._train.setValue(70)
        self._train.setSuffix(" %")

        self._val = QSpinBox()
        self._val.setRange(1, 98)
        self._val.setValue(15)
        self._val.setSuffix(" %")

        self._test_lbl = QLabel("15 %")
        self._test_lbl.setStyleSheet("color:#6272a4;")

        form.addRow("Train:", self._train)
        form.addRow("Val:", self._val)
        form.addRow("Test (авто):", self._test_lbl)

        self._warn = QLabel("")
        self._warn.setStyleSheet("color:#ff5555;")

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._check_accept)
        btns.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._warn)
        layout.addWidget(btns)

        self._train.valueChanged.connect(self._update_test)
        self._val.valueChanged.connect(self._update_test)

    def _update_test(self):
        t = 100 - self._train.value() - self._val.value()
        self._test_lbl.setText(f"{t} %")
        if t < 0:
            self._warn.setText("Сумма превышает 100 %!")
        else:
            self._warn.setText("")

    def _check_accept(self):
        if 100 - self._train.value() - self._val.value() < 0:
            return
        self.accept()

    def train_pct(self) -> int:
        return self._train.value()

    def val_pct(self) -> int:
        return self._val.value()


class _LoadWorker(QObject):
    done  = Signal(object)   # TableData
    error = Signal(str)

    def __init__(self, path: str):
        super().__init__()
        self._path = path

    @Slot()
    def run(self):
        try:
            self.done.emit(read_table(self._path))
        except Exception as e:
            self.error.emit(str(e))


class _MatchWorker(QObject):
    done  = Signal(object)   # MatchResult
    error = Signal(str)

    def __init__(self, table: TableData, images: list[str],
                 filename_col: str, mode: str):
        super().__init__()
        self._table        = table
        self._images       = images
        self._filename_col = filename_col
        self._mode         = mode

    @Slot()
    def run(self):
        try:
            self.done.emit(
                match_filenames(self._table, self._images,
                                self._filename_col, self._mode)
            )
        except Exception as e:
            self.error.emit(str(e))


class DBTab(QWidget):
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._table:        TableData | None  = None
        self._match:        MatchResult | None = None
        self._images:       list[str]         = []
        self._label_col:    str               = ""
        self._threads:      list[QThread]     = []
        self._workers:      list[QObject]     = []   # защита от GC

        self._build_ui()

    # ── Интерфейс ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Шаг 0: Загрузить изображения ─────────────────────────────────────
        grp0 = QGroupBox("Шаг 0 — Загрузить фотографии (если не загружены во вкладке «Аннотация»)")
        g0   = QHBoxLayout(grp0)
        self._images_lbl = QLabel("Изображений: 0")
        self._images_lbl.setStyleSheet("color:#6272a4;")
        btn_load_folder = QPushButton("📁 Папка с фото")
        btn_load_folder.setFixedWidth(130)
        btn_load_files  = QPushButton("🖼 Выбрать файлы")
        btn_load_files.setFixedWidth(130)
        g0.addWidget(btn_load_folder)
        g0.addWidget(btn_load_files)
        g0.addWidget(self._images_lbl, stretch=1)
        root.addWidget(grp0)

        # ── Шаг 1: Загрузить таблицу ─────────────────────────────────────────
        grp1 = QGroupBox("Шаг 1 — Загрузить таблицу")
        g1   = QHBoxLayout(grp1)
        self._path_lbl = QLabel("Файл не выбран")
        self._path_lbl.setStyleSheet("color:#6272a4;")
        btn_open = QPushButton("📂 Открыть")
        btn_open.setFixedWidth(100)
        g1.addWidget(self._path_lbl, stretch=1)
        g1.addWidget(btn_open)
        root.addWidget(grp1)

        # ── Шаг 2: Настройка колонок ──────────────────────────────────────────
        grp2 = QGroupBox("Шаг 2 — Настройка колонок")
        g2   = QHBoxLayout(grp2)

        g2.addWidget(QLabel("Колонка файлов:"))
        self._col_file  = QComboBox()
        self._col_file.setMinimumWidth(160)
        g2.addWidget(self._col_file)

        g2.addWidget(QLabel("Колонка метки:"))
        self._col_label = QComboBox()
        self._col_label.setMinimumWidth(160)
        g2.addWidget(self._col_label)

        g2.addWidget(QLabel("Режим:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["stem (без расш.)", "exact (точное)", "contains (вхождение)"])
        g2.addWidget(self._mode_combo)
        g2.addStretch()
        root.addWidget(grp2)

        # ── Шаг 3: Действия ───────────────────────────────────────────────────
        grp3  = QGroupBox("Шаг 3 — Действия")
        g3    = QHBoxLayout(grp3)
        self._btn_match  = QPushButton("🔗 Сопоставить")
        self._btn_match.setStyleSheet(
            "QPushButton{background:#2979ff;color:white;font-weight:bold;padding:5px 12px;}"
        )
        self._btn_apply  = QPushButton("✅ Применить аннотации")
        self._btn_apply.setEnabled(False)
        self._btn_export = QPushButton("💾 Экспорт CSV")
        self._btn_export.setEnabled(False)
        self._btn_save_db = QPushButton("🗄 Сохранить в SQLite")
        self._btn_save_db.setEnabled(False)
        self._btn_save_db.setToolTip(
            "Сохранить сопоставление в SQLite БД:\n"
            "таблица matched_annotations(filename, object)"
        )
        self._btn_dataset = QPushButton("📦 Создать датасет")
        self._btn_dataset.setToolTip(
            "Создать полноценный SQLite датасет из загруженных\n"
            "фотографий и их аннотаций (.json файлы рядом с фото).\n"
            "Таблицы: images / annotations / classes"
        )
        self._stats_lbl  = QLabel("—")
        self._stats_lbl.setStyleSheet("color:#f1fa8c; font-size:11px;")
        for w in (self._btn_match, self._btn_apply, self._btn_export,
                  self._btn_save_db, self._btn_dataset):
            w.setFixedHeight(30)
            g3.addWidget(w)
        g3.addWidget(self._stats_lbl)
        g3.addStretch()
        root.addWidget(grp3)

        # ── Таблица результатов ───────────────────────────────────────────────
        self._tbl = QTableWidget()
        self._tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._tbl.setAlternatingRowColors(True)
        self._tbl.horizontalHeader().setStretchLastSection(True)
        self._tbl.setStyleSheet("alternate-background-color: #282a36;")
        root.addWidget(self._tbl)

        # Подключения сигналов
        btn_load_folder.clicked.connect(self._on_load_folder)
        btn_load_files.clicked.connect(self._on_load_files)
        btn_open.clicked.connect(self._on_open)
        self._btn_match.clicked.connect(self._on_match)
        self._btn_apply.clicked.connect(self._on_apply)
        self._btn_export.clicked.connect(self._on_export)
        self._btn_save_db.clicked.connect(self._on_save_sqlite)
        self._btn_dataset.clicked.connect(self._on_create_dataset)

    # ── Загрузка изображений ───────────────────────────────────────────────────

    def _on_load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Папка с фотографиями")
        if not folder:
            return
        images = collect_images_flat(folder)
        if not images:
            QMessageBox.information(self, "Пусто", "Изображений не найдено в папке")
            return
        self.set_images(images)

    def _on_load_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Выбрать изображения", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if paths:
            self.set_images(paths)

    # ── Загрузка таблицы ───────────────────────────────────────────────────────

    def _on_open(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть таблицу", "",
            "Tables (*.xlsx *.xls *.csv *.tsv *.json *.db *.sqlite);;All (*)"
        )
        if not path:
            return
        self._path_lbl.setText(os.path.basename(path))
        self._run_async(
            _LoadWorker(path), "run",
            on_done=self._on_table_loaded,
            on_error=lambda e: QMessageBox.warning(self, "Ошибка", e)
        )

    def _on_table_loaded(self, tbl: TableData):
        self._table = tbl
        # Заполнить списки колонок
        for cb in (self._col_file, self._col_label):
            cb.clear()
            cb.addItems(tbl.headers)

        # Автоопределение колонок
        for i, h in enumerate(tbl.headers):
            hl = h.lower()
            if any(k in hl for k in ("file", "name", "path", "image", "фото")):
                self._col_file.setCurrentIndex(i)
                break
        for i, h in enumerate(tbl.headers):
            hl = h.lower()
            if any(k in hl for k in ("label", "class", "метка", "тип", "категория")):
                self._col_label.setCurrentIndex(i)
                break

        # Предпросмотр
        self._show_preview(tbl)
        self._stats_lbl.setText(
            f"Загружено: {tbl.source_format.upper()}  "
            f"{len(tbl.rows)} строк  {len(tbl.headers)} колонок"
        )

    def _show_preview(self, tbl: TableData):
        self._tbl.setColumnCount(len(tbl.headers))
        self._tbl.setHorizontalHeaderLabels(tbl.headers)
        rows = tbl.rows[:300]
        self._tbl.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, val in enumerate(row):
                self._tbl.setItem(r, c, QTableWidgetItem(str(val)))
        self._tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    # ── Сопоставление ─────────────────────────────────────────────────────────

    def set_images(self, images: list[str]):
        self._images = images
        self._images_lbl.setText(f"Изображений: {len(images)}")

    def _on_match(self):
        if self._table is None:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите таблицу")
            return
        if not self._images:
            QMessageBox.warning(self, "Ошибка",
                                "Нет изображений. Загрузите их в «Аннотации»")
            return

        mode_map = {0: "stem", 1: "exact", 2: "contains"}
        mode = mode_map[self._mode_combo.currentIndex()]
        worker = _MatchWorker(
            self._table, self._images,
            self._col_file.currentText(), mode
        )
        self._run_async(worker, "run",
                        on_done=self._on_matched,
                        on_error=lambda e: QMessageBox.warning(self, "Ошибка", e))

    def _on_matched(self, result: MatchResult):
        self._match = result
        self._label_col = self._col_label.currentText()

        nm = len(result.matched)
        nu = len(result.unmatched_db)
        nn = len(result.images_without_record)
        self._stats_lbl.setText(
            f"✓ Совпало: {nm}   ✗ Нет фото: {nu}   📷 Нет в БД: {nn}"
        )
        self._btn_apply.setEnabled(nm > 0)
        self._btn_export.setEnabled(True)
        self._btn_save_db.setEnabled(nm > 0)
        self._show_results(result)

    def _show_results(self, result: MatchResult):
        tbl = self._table
        lc  = self._col_label.currentText()
        cols = ["Статус", "Файл", lc] + tbl.headers
        self._tbl.setColumnCount(len(cols))
        self._tbl.setHorizontalHeaderLabels(cols)
        total = len(result.matched) + len(result.unmatched_db) + len(result.images_without_record)
        self._tbl.setRowCount(total)

        row_n = 0

        def _add_row(status, fname, label, row_data, bg):
            items = [status, fname, label] + [str(row_data.get(h, "")) for h in tbl.headers]
            for c, val in enumerate(items):
                it = QTableWidgetItem(val)
                it.setBackground(QBrush(bg))
                self._tbl.setItem(row_n, c, it)

        for idx, m in result.matched.items():
            fname  = os.path.basename(m["image_path"])
            label  = str(m["row"].get(lc, ""))
            _add_row("✓ Совпало", fname, label, m["row"], _GREEN)
            row_n += 1

        for idx, m in result.unmatched_db.items():
            label = str(m["row"].get(lc, ""))
            _add_row("✗ Нет фото", "", label, m["row"], _RED)
            row_n += 1

        for path in result.images_without_record:
            items = ["📷 Нет в БД", os.path.basename(path), ""] + [""] * len(tbl.headers)
            for c, val in enumerate(items):
                it = QTableWidgetItem(val)
                it.setBackground(QBrush(_GRAY))
                self._tbl.setItem(row_n, c, it)
            row_n += 1

        self._tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    # ── Применение аннотаций ───────────────────────────────────────────────────

    def _on_apply(self):
        if not self._match:
            return
        lc = self._label_col
        written, errors = 0, []

        for idx, m in self._match.matched.items():
            img_path = m["image_path"]
            label    = str(m["row"].get(lc, "object")).strip() or "object"
            try:
                ia = ann_io.load(img_path)
                from PIL import Image as _PIL
                with _PIL.open(img_path) as im:
                    ia.width, ia.height = im.size
                # Добавить аннотацию метки (всё изображение, без bbox)
                existing = {a.label for a in ia.annotations}
                if label not in existing:
                    from app.utils.annotation import label_to_model
                    ia.annotations.append(
                        ann_io.Annotation(
                            label=label,
                            model=label_to_model(label),
                            source="table",
                        )
                    )
                ann_io.save(ia)
                written += 1
            except Exception as e:
                errors.append(f"{os.path.basename(img_path)}: {e}")

        msg = f"Создано/обновлено: {written} аннотаций"
        if errors:
            msg += f"\nОшибок: {len(errors)}:\n" + "\n".join(errors[:5])
        QMessageBox.information(self, "Готово", msg)
        self.status_message.emit(msg)

    # ── Экспорт ────────────────────────────────────────────────────────────────

    def _on_export(self):
        if not self._match:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результат", "", "CSV (*.csv)"
        )
        if not path:
            return
        import csv
        tbl = self._table
        lc  = self._label_col
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["status", "image_file", lc] + tbl.headers)
            for _, m in self._match.matched.items():
                w.writerow(
                    ["matched", os.path.basename(m["image_path"]),
                     m["row"].get(lc, "")]
                    + [str(m["row"].get(h, "")) for h in tbl.headers]
                )
            for _, m in self._match.unmatched_db.items():
                w.writerow(
                    ["unmatched_db", "", m["row"].get(lc, "")]
                    + [str(m["row"].get(h, "")) for h in tbl.headers]
                )
            for p in self._match.images_without_record:
                w.writerow(["no_record", os.path.basename(p), ""] + [""] * len(tbl.headers))
        QMessageBox.information(self, "Экспорт", f"Сохранено:\n{path}")

    # ── Сохранение в SQLite ────────────────────────────────────────────────────

    def _on_save_sqlite(self):
        if not self._match or not self._match.matched:
            QMessageBox.warning(self, "Ошибка", "Нет сопоставленных данных")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить базу данных", "annotations.db",
            "SQLite Database (*.db *.sqlite);;All (*)"
        )
        if not path:
            return

        lc = self._label_col
        try:
            import sqlite3
            conn = sqlite3.connect(path)
            cur  = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS matched_annotations (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    object   TEXT NOT NULL
                )
            """)
            # Очистить существующие строки, если таблица уже была в файле
            cur.execute("DELETE FROM matched_annotations")

            rows = [
                (os.path.basename(m["image_path"]),
                 str(m["row"].get(lc, "")).strip())
                for m in self._match.matched.values()
            ]
            cur.executemany(
                "INSERT INTO matched_annotations (filename, object) VALUES (?, ?)",
                rows
            )
            conn.commit()
            conn.close()

            QMessageBox.information(
                self, "Готово",
                f"Сохранено {len(rows)} записей в:\n{path}\n\n"
                f"Таблица: matched_annotations\n"
                f"Колонки: filename | object"
            )
            self.status_message.emit(f"SQLite: сохранено {len(rows)} записей → {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось сохранить:\n{e}")

    # ── Создание датасета ──────────────────────────────────────────────────────

    def _on_create_dataset(self):
        if not self._images:
            QMessageBox.warning(self, "Ошибка",
                                "Нет изображений. Загрузите их в Шаге 0 или во вкладке «Аннотация».")
            return

        # Запросить соотношение разбивки
        dlg = _SplitDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        train_pct, val_pct = dlg.train_pct(), dlg.val_pct()
        test_pct = 100 - train_pct - val_pct

        # Запросить путь для сохранения
        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить датасет", "dataset.db",
            "SQLite Database (*.db *.sqlite);;All (*)"
        )
        if not path:
            return

        import sqlite3
        from app.utils import annotation as ann_io
        from config import ALL_LABELS, MODEL_DEFS

        try:
            conn = sqlite3.connect(path)
            cur  = conn.cursor()

            cur.executescript("""
                CREATE TABLE IF NOT EXISTS classes (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    name         TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL DEFAULT '',
                    model_id     TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS images (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    path     TEXT NOT NULL,
                    width    INTEGER NOT NULL DEFAULT 0,
                    height   INTEGER NOT NULL DEFAULT 0,
                    split    TEXT NOT NULL DEFAULT 'train'
                );
                CREATE TABLE IF NOT EXISTS annotations (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id   INTEGER NOT NULL REFERENCES images(id),
                    class_id   INTEGER REFERENCES classes(id),
                    label      TEXT NOT NULL DEFAULT '',
                    model      TEXT NOT NULL DEFAULT '',
                    bbox_x     INTEGER,
                    bbox_y     INTEGER,
                    bbox_w     INTEGER,
                    bbox_h     INTEGER,
                    source     TEXT NOT NULL DEFAULT 'manual',
                    confidence REAL
                );
            """)

            # Заполнить таблицу классов
            label_to_id: dict[str, int] = {}
            for model_id, mdef in MODEL_DEFS.items():
                for lname, ldisp in mdef["classes"].items():
                    cur.execute(
                        "INSERT OR IGNORE INTO classes (name, display_name, model_id) VALUES (?, ?, ?)",
                        (lname, ldisp, model_id)
                    )
            conn.commit()
            for row in cur.execute("SELECT id, name FROM classes"):
                label_to_id[row[1]] = row[0]

            # Назначить разбивку
            imgs = list(self._images)
            random.shuffle(imgs)
            n = len(imgs)
            n_train = round(n * train_pct / 100)
            n_val   = round(n * val_pct   / 100)
            splits: list[str] = (
                ["train"] * n_train +
                ["val"]   * n_val   +
                ["test"]  * (n - n_train - n_val)
            )

            n_images = 0
            n_ann    = 0
            n_no_ann = 0

            for img_path, split in zip(imgs, splits):
                ia = ann_io.load(img_path)

                # Прочитать размеры, если их нет в JSON
                w, h = ia.width, ia.height
                if w == 0 or h == 0:
                    try:
                        from PIL import Image as _PIL
                        with _PIL.open(img_path) as im:
                            w, h = im.size
                    except Exception:
                        pass

                cur.execute(
                    "INSERT INTO images (filename, path, width, height, split) VALUES (?, ?, ?, ?, ?)",
                    (os.path.basename(img_path), img_path, w, h, split)
                )
                image_id = cur.lastrowid
                n_images += 1

                if not ia.annotations:
                    n_no_ann += 1

                for a in ia.annotations:
                    cid = label_to_id.get(a.label)
                    if cid is None and a.label:
                        cur.execute(
                            "INSERT OR IGNORE INTO classes (name, display_name, model_id) VALUES (?, ?, ?)",
                            (a.label, ALL_LABELS.get(a.label, a.label), a.model)
                        )
                        conn.commit()
                        cid = cur.lastrowid or cur.execute(
                            "SELECT id FROM classes WHERE name=?", (a.label,)
                        ).fetchone()[0]
                        label_to_id[a.label] = cid

                    bx = by = bw = bh = None
                    if a.bbox:
                        bx, by, bw, bh = a.bbox.x, a.bbox.y, a.bbox.w, a.bbox.h

                    cur.execute(
                        "INSERT INTO annotations "
                        "(image_id, class_id, label, model, bbox_x, bbox_y, bbox_w, bbox_h, source, confidence) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (image_id, cid, a.label, a.model,
                         bx, by, bw, bh, a.source, a.confidence)
                    )
                    n_ann += 1

            conn.commit()
            conn.close()

            splits_stat = (
                f"train={splits.count('train')}  "
                f"val={splits.count('val')}  "
                f"test={splits.count('test')}"
            )
            QMessageBox.information(
                self, "Датасет создан",
                f"Файл:  {path}\n\n"
                f"Изображений:  {n_images}\n"
                f"Аннотаций:    {n_ann}\n"
                f"Без аннотаций: {n_no_ann}\n"
                f"Разбивка:  {splits_stat}\n\n"
                f"Таблицы:  images / annotations / classes"
            )
            self.status_message.emit(
                f"Датасет: {n_images} фото, {n_ann} аннотаций → {os.path.basename(path)}"
            )

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось создать датасет:\n{e}")

    # ── Вспомогательный метод для асинхронного выполнения ─────────────────────

    def _run_async(self, worker: QObject, slot: str,
                   on_done=None, on_error=None):
        thread = QThread()
        worker.moveToThread(thread)
        if on_done:
            worker.done.connect(on_done)
            worker.done.connect(lambda _: thread.quit())
        if on_error:
            worker.error.connect(on_error)
            worker.error.connect(lambda _: thread.quit())
        thread.started.connect(getattr(worker, slot))
        thread.finished.connect(
            lambda: self._threads.remove(thread) if thread in self._threads else None
        )
        thread.finished.connect(
            lambda: self._workers.remove(worker) if worker in self._workers else None
        )
        self._threads.append(thread)
        self._workers.append(worker)
        thread.start()
