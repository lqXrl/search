"""
Вкладка предсказания — инференс с обученными моделями.

Режимы:
  • Одна модель    — выбор конкретной модели
  • Все модели     — запустить все обученные и показать сводку
  • Batch (папка)  — обработать папку, сохранить CSV
"""

from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtGui import QPixmap, QFont
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel,
    QListWidget, QMessageBox, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
)

from app.core.predictor import Predictor
from app.models.registry import ModelRegistry
from config import MODEL_DEFS


# ── Рабочий поток ─────────────────────────────────────────────────────────────

class _Worker(QObject):
    done  = Signal(object)
    error = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn = fn

    @Slot()
    def run(self):
        try:
            self.done.emit(self._fn())
        except Exception as e:
            self.error.emit(str(e))


# ── Вкладка ────────────────────────────────────────────────────────────────────

class PredictTab(QWidget):
    def __init__(self, registry: ModelRegistry, parent=None):
        super().__init__(parent)
        self._registry    = registry
        self._image_path: str | None = None
        self._last_result: dict | None = None
        self._threads:    list[QThread] = []
        self._workers:    list         = []   # защита от GC

        self._build_ui()

    # ── Интерфейс ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # Выбор модели
        mdl_grp = QGroupBox("Выбор модели")
        mg = QHBoxLayout(mdl_grp)

        mg.addWidget(QLabel("Модель:"))
        self._model_cb = QComboBox()
        self._model_cb.setMinimumWidth(220)
        self._populate_models()
        mg.addWidget(self._model_cb)

        self._all_chk = QCheckBox("Запустить все обученные модели")
        self._all_chk.setChecked(True)
        mg.addWidget(self._all_chk)
        mg.addStretch()
        root.addWidget(mdl_grp)

        # Кнопки
        btn_row = QHBoxLayout()
        self._btn_load    = QPushButton("🖼 Загрузить фото")
        self._btn_predict = QPushButton("▶ Предсказать")
        self._btn_predict.setStyleSheet(
            "QPushButton{background:#2979ff;color:white;font-weight:bold;padding:5px 14px;}"
        )
        self._btn_batch  = QPushButton("📁 Batch (папка)")
        self._btn_export = QPushButton("💾 Экспорт")
        for b in (self._btn_load, self._btn_predict, self._btn_batch, self._btn_export):
            b.setFixedHeight(30)
            btn_row.addWidget(b)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # Баннер с результатом
        self._result_lbl = QLabel()
        self._result_lbl.setWordWrap(True)
        self._result_lbl.setTextFormat(Qt.RichText)
        self._result_lbl.setStyleSheet(
            "background:#282a36; border-radius:6px; padding:10px; "
            "color:#f8f8f2; font-size:13px; border:1px solid #44475a;"
        )
        self._result_lbl.setVisible(False)
        root.addWidget(self._result_lbl)

        # Изображение
        self._img_lbl = QLabel("Загрузите изображение")
        self._img_lbl.setAlignment(Qt.AlignCenter)
        self._img_lbl.setStyleSheet("background:#1e1e2e; border:1px solid #44475a;")
        self._img_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self._img_lbl, stretch=1)

        # Лог пакетной обработки
        self._batch_log = QListWidget()
        self._batch_log.setFont(QFont("Courier New", 9))
        self._batch_log.setMaximumHeight(130)
        self._batch_log.setVisible(False)
        root.addWidget(self._batch_log)

        # Подключения сигналов
        self._btn_load.clicked.connect(self._on_load)
        self._btn_predict.clicked.connect(self._on_predict)
        self._btn_batch.clicked.connect(self._on_batch)
        self._btn_export.clicked.connect(self._on_export)
        self._all_chk.toggled.connect(
            lambda v: self._model_cb.setEnabled(not v)
        )
        self._model_cb.setEnabled(not self._all_chk.isChecked())

    def _populate_models(self):
        self._model_cb.clear()
        for mid, mdef in MODEL_DEFS.items():
            trained = " ✓" if self._registry.is_trained(mid) else " ⚠"
            self._model_cb.addItem(
                f"{mdef['icon']}  {mdef['name']}{trained}", mid
            )

    # ── Загрузка ───────────────────────────────────────────────────────────────

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Загрузить изображение", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)"
        )
        if not path:
            return
        self._image_path = path
        pix = QPixmap(path)
        if not pix.isNull():
            self._show_pixmap(pix)
        self._result_lbl.setVisible(False)
        self._batch_log.setVisible(False)

    # ── Предсказание ───────────────────────────────────────────────────────────

    def _on_predict(self):
        if not self._image_path:
            QMessageBox.warning(self, "Ошибка", "Сначала загрузите изображение")
            return

        if self._all_chk.isChecked():
            model_ids = self._registry.list_trained()
            if not model_ids:
                QMessageBox.warning(self, "Нет обученных моделей",
                                    "Обучите хотя бы одну модель во вкладке «Обучение»")
                return
        else:
            model_ids = [self._model_cb.currentData()]
            if not self._registry.is_trained(model_ids[0]):
                QMessageBox.warning(self, "Модель не обучена",
                                    f"Модель «{MODEL_DEFS[model_ids[0]]['name']}» ещё не обучена")
                return

        path = self._image_path

        def _run():
            results = {}
            for mid in model_ids:
                model = self._registry.get(mid)
                pred  = Predictor(model, mid)
                results[mid] = pred.predict(path)
            return results

        self._run_async(
            _run,
            on_done=self._show_results,
            on_error=lambda e: QMessageBox.warning(self, "Ошибка", e),
        )

    def _show_results(self, results: dict):
        self._last_result = results

        lines = []
        for mid, r in results.items():
            mdef = MODEL_DEFS[mid]
            classes_str = "  ".join(
                f"<span style='color:#6272a4'>{mdef['classes'].get(k,k)}: "
                f"{v:.0%}</span>"
                for k, v in r["probs"].items()
            )
            conf_color = "#50fa7b" if r["confidence"] > 0.75 else (
                "#f1fa8c" if r["confidence"] > 0.5 else "#ff5555"
            )
            lines.append(
                f"<b>{mdef['icon']} {mdef['name']}</b>"
                f"&nbsp;&nbsp;→&nbsp;&nbsp;"
                f"<span style='font-size:14px; font-weight:bold;'>{r['class_label']}</span>"
                f"&nbsp;<span style='color:{conf_color}'>{r['confidence']:.1%}</span>"
                f"<br><span style='font-size:11px;'>{classes_str}</span>"
            )
        self._result_lbl.setText("<br><br>".join(lines))
        self._result_lbl.setVisible(True)

        # Наложение на изображение
        self._render_overlay(self._image_path, results)

    def _render_overlay(self, path: str, results: dict):
        try:
            img  = Image.open(path).convert("RGB")
            draw = ImageDraw.Draw(img, "RGBA")
            try:
                font_big = ImageFont.truetype("arial.ttf", 20)
                font_sm  = ImageFont.truetype("arial.ttf", 14)
            except Exception:
                font_big = ImageFont.load_default()
                font_sm  = font_big

            y = 10
            for mid, r in results.items():
                mdef = MODEL_DEFS[mid]
                conf = r["confidence"]
                txt  = f"{mdef['name']}: {r['class_label']}  {conf:.0%}"

                bb   = draw.textbbox((0, 0), txt, font=font_big)
                tw, th = bb[2] - bb[0] + 14, bb[3] - bb[1] + 8

                # Определить цвет по уверенности
                if conf > 0.75:
                    bg = (80, 250, 123, 210)
                elif conf > 0.5:
                    bg = (241, 250, 140, 210)
                else:
                    bg = (255, 85, 85, 210)

                draw.rounded_rectangle([8, y, 8 + tw, y + th], radius=6, fill=bg)
                draw.text((15, y + 4), txt, fill=(30, 30, 30), font=font_big)
                y += th + 6

            qim = ImageQt(img)
            pix = QPixmap.fromImage(qim)
            self._show_pixmap(pix)
        except Exception:
            pass

    def _show_pixmap(self, pix: QPixmap):
        self._img_lbl.setPixmap(
            pix.scaled(self._img_lbl.size(), Qt.KeepAspectRatio,
                       Qt.SmoothTransformation)
        )

    # ── Пакетная обработка ─────────────────────────────────────────────────────

    def _on_batch(self):
        folder = QFileDialog.getExistingDirectory(self, "Папка с изображениями")
        if not folder:
            return

        from app.utils.file_utils import collect_images_flat
        files = collect_images_flat(folder)
        if not files:
            QMessageBox.information(self, "Пусто", "Изображений не найдено")
            return

        model_ids = self._registry.list_trained()
        if not model_ids:
            QMessageBox.warning(self, "Ошибка", "Нет обученных моделей")
            return

        self._batch_log.clear()
        self._batch_log.setVisible(True)

        def _run():
            predictors = {mid: Predictor(self._registry.get(mid), mid)
                          for mid in model_ids}
            rows = []
            for p in files:
                row = {"file": os.path.basename(p)}
                for mid, pred in predictors.items():
                    try:
                        r = pred.predict(p)
                        row[f"{mid}_class"] = r["class_label"]
                        row[f"{mid}_conf"]  = round(r["confidence"], 4)
                    except Exception as e:
                        row[f"{mid}_class"] = "ERROR"
                        row[f"{mid}_conf"]  = 0.0
                rows.append(row)
            return rows

        def _done(rows):
            for row in rows:
                parts = [row["file"]]
                for mid in model_ids:
                    cls  = row.get(f"{mid}_class", "—")
                    conf = row.get(f"{mid}_conf",  0)
                    parts.append(f"{MODEL_DEFS[mid]['name']}: {cls} {conf:.0%}")
                self._batch_log.addItem("  |  ".join(parts))
            self._batch_log.scrollToBottom()
            self._last_result = {"batch": rows, "model_ids": model_ids}

        self._run_async(_run, on_done=_done,
                        on_error=lambda e: QMessageBox.warning(self, "Ошибка", e))

    # ── Экспорт ────────────────────────────────────────────────────────────────

    def _on_export(self):
        if not self._last_result:
            QMessageBox.warning(self, "Ошибка", "Нет результатов для экспорта")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить", "", "CSV (*.csv);;JSON (*.json)"
        )
        if not path:
            return

        if "batch" in self._last_result:
            rows      = self._last_result["batch"]
            model_ids = self._last_result.get("model_ids", [])
            if path.endswith(".json"):
                import json
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(rows, f, ensure_ascii=False, indent=2)
            else:
                import csv
                if not rows:
                    return
                with open(path, "w", newline="", encoding="utf-8-sig") as f:
                    w = csv.DictWriter(f, fieldnames=rows[0].keys())
                    w.writeheader()
                    w.writerows(rows)
        else:
            import json
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._last_result, f, ensure_ascii=False, indent=2)

        QMessageBox.information(self, "Экспорт", f"Сохранено:\n{path}")

    # ── Асинхронное выполнение ─────────────────────────────────────────────────

    def _run_async(self, fn, on_done=None, on_error=None):
        worker = _Worker(fn)
        thread = QThread()
        worker.moveToThread(thread)
        if on_done:
            worker.done.connect(on_done)
            worker.done.connect(lambda _: thread.quit())
        if on_error:
            worker.error.connect(on_error)
            worker.error.connect(lambda _: thread.quit())
        thread.started.connect(worker.run)
        thread.finished.connect(
            lambda: self._threads.remove(thread) if thread in self._threads else None
        )
        thread.finished.connect(
            lambda: self._workers.remove(worker) if worker in self._workers else None
        )
        self._threads.append(thread)
        self._workers.append(worker)
        thread.start()

    # ── Обновление ─────────────────────────────────────────────────────────────

    def refresh_models(self):
        self._populate_models()
