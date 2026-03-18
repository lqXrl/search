"""
Вкладка обучения — обучение модели с учителем и графиками в реальном времени.

Макет:
  ┌─ Конфигурация модели и данных ─────────────────────┐
  │  Выбор модели | Папка данных | Гиперпараметры       │
  └─────────────────────────────────────────────────────┘
  ┌─ Управление ────┐
  │ Старт | Стоп   │  Прогресс: [батч]  Эпоха N/M     │
  └─────────────────┘
  ┌─ Графики (loss + acc) ──────────────────────────────┐
  │  MetricsChart                                       │
  └─────────────────────────────────────────────────────┘
  ┌─ Лог ───────────────────────────────────────────────┐
  └─────────────────────────────────────────────────────┘
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QMessageBox, QProgressBar, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from app.core.trainer import TrainerWorker
from app.models.registry import ModelRegistry
from app.ui.widgets.chart import MetricsChart
from config import (
    DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_VAL_SPLIT,
    MODEL_DEFS,
)


class TrainTab(QWidget):
    def __init__(self, registry: ModelRegistry, parent=None):
        super().__init__(parent)
        self._registry = registry
        self._thread:  QThread | None     = None
        self._worker:  TrainerWorker | None = None

        self._build_ui()

    # ── Интерфейс ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Группа конфигурации ───────────────────────────────────────────────
        cfg_grp = QGroupBox("Конфигурация")
        cfg_l   = QVBoxLayout(cfg_grp)

        # Модель
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Модель:"))
        self._model_cb = QComboBox()
        for mid, mdef in MODEL_DEFS.items():
            trained = " ✓" if self._registry.is_trained(mid) else ""
            self._model_cb.addItem(
                f"{mdef.get('icon','')}  {mdef['name']}{trained}", mid
            )
        row1.addWidget(self._model_cb)
        self._model_info = QLabel()
        self._model_info.setStyleSheet("color:#6272a4; font-size:11px;")
        row1.addWidget(self._model_info)
        row1.addStretch()
        cfg_l.addLayout(row1)
        self._model_cb.currentIndexChanged.connect(self._on_model_change)
        self._on_model_change(0)

        # Папка с данными / файл БД
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Данные:"))
        self._data_line = QLineEdit()
        self._data_line.setPlaceholderText("Папка с .json аннотациями  или  файл датасета .db")
        btn_browse = QPushButton("📁 Папка")
        btn_browse.setFixedWidth(80)
        btn_browse.clicked.connect(self._browse_folder)
        btn_browse_db = QPushButton("🗄 База .db")
        btn_browse_db.setFixedWidth(90)
        btn_browse_db.clicked.connect(self._browse_db)
        row2.addWidget(self._data_line)
        row2.addWidget(btn_browse)
        row2.addWidget(btn_browse_db)
        cfg_l.addLayout(row2)

        # Гиперпараметры
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Эпохи:"))
        self._ep_spin = QSpinBox()
        self._ep_spin.setRange(1, 1000)
        self._ep_spin.setValue(DEFAULT_EPOCHS)
        row3.addWidget(self._ep_spin)

        row3.addWidget(QLabel("Batch:"))
        self._bs_spin = QSpinBox()
        self._bs_spin.setRange(1, 512)
        self._bs_spin.setValue(DEFAULT_BATCH_SIZE)
        row3.addWidget(self._bs_spin)

        row3.addWidget(QLabel("LR:"))
        self._lr_spin = QDoubleSpinBox()
        self._lr_spin.setDecimals(6)
        self._lr_spin.setRange(1e-7, 1.0)
        self._lr_spin.setSingleStep(1e-5)
        self._lr_spin.setValue(DEFAULT_LR)
        row3.addWidget(self._lr_spin)

        row3.addWidget(QLabel("Val:"))
        self._val_spin = QDoubleSpinBox()
        self._val_spin.setRange(0.05, 0.5)
        self._val_spin.setSingleStep(0.05)
        self._val_spin.setValue(DEFAULT_VAL_SPLIT)
        row3.addWidget(self._val_spin)

        row3.addWidget(QLabel("Backbone:"))
        self._backbone_cb = QComboBox()
        self._backbone_cb.addItems(["resnet18", "resnet34", "efficientnet_b0"])
        row3.addWidget(self._backbone_cb)
        row3.addStretch()
        cfg_l.addLayout(row3)

        root.addWidget(cfg_grp)

        # ── Управление ────────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        self._btn_run  = QPushButton("▶ Запустить обучение")
        self._btn_run.setStyleSheet(
            "QPushButton{background:#27ae60;color:white;font-weight:bold;padding:6px 14px;}"
            "QPushButton:disabled{background:#555;}"
        )
        self._btn_stop = QPushButton("■ Остановить")
        self._btn_stop.setEnabled(False)
        self._epoch_lbl = QLabel("")
        self._epoch_lbl.setStyleSheet("color:#f8f8f2; font-size:12px; font-weight:bold;")
        ctrl.addWidget(self._btn_run)
        ctrl.addWidget(self._btn_stop)
        ctrl.addWidget(self._epoch_lbl)
        ctrl.addStretch()
        root.addLayout(ctrl)

        # Прогресс-бары
        pb_row = QHBoxLayout()
        self._epoch_bar = QProgressBar()
        self._epoch_bar.setFixedHeight(16)
        self._epoch_bar.setVisible(False)
        self._batch_bar = QProgressBar()
        self._batch_bar.setFixedHeight(10)
        self._batch_bar.setVisible(False)
        pb_row.addWidget(QLabel("Эпохи:"))
        pb_row.addWidget(self._epoch_bar, stretch=1)
        pb_row.addWidget(QLabel("Батчи:"))
        pb_row.addWidget(self._batch_bar, stretch=2)
        root.addLayout(pb_row)

        # ── Графики ───────────────────────────────────────────────────────────
        self._chart = MetricsChart()
        self._chart.setMinimumHeight(220)
        root.addWidget(self._chart)

        # ── Лог ───────────────────────────────────────────────────────────────
        root.addWidget(QLabel("Лог обучения:"))
        self._log = QListWidget()
        self._log.setFont(QFont("Courier New", 9))
        self._log.setMaximumHeight(160)
        root.addWidget(self._log)

        # Подключения сигналов
        self._btn_run.clicked.connect(self._on_run)
        self._btn_stop.clicked.connect(self._on_stop)

    # ── Информация о модели ────────────────────────────────────────────────────

    def _on_model_change(self, _=None):
        mid  = self._model_cb.currentData()
        mdef = MODEL_DEFS.get(mid, {})
        classes = mdef.get("classes", {})
        trained = self._registry.is_trained(mid)
        cls_names = " | ".join(classes.values())
        self._model_info.setText(
            f"{mdef.get('description', '')}  ·  "
            f"Классов {len(classes)}: [{cls_names}]  ·  "
            f"{'✓ Обучена' if trained else '⚠ Не обучена'}"
        )
        # Обновить также текст в комбо
        for i in range(self._model_cb.count()):
            m = self._model_cb.itemData(i)
            md = MODEL_DEFS.get(m, {})
            tr = " ✓" if self._registry.is_trained(m) else ""
            self._model_cb.setItemText(i, f"{md.get('icon','')}  {md['name']}{tr}")

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Папка с данными")
        if folder:
            self._data_line.setText(folder)

    def _browse_db(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть датасет", "",
            "SQLite Dataset (*.db *.sqlite);;All (*)"
        )
        if path:
            self._data_line.setText(path)

    # ── Запуск / Остановка ─────────────────────────────────────────────────────

    def _on_run(self):
        data_dir = self._data_line.text().strip()
        if not data_dir:
            QMessageBox.warning(self, "Ошибка",
                                "Выберите папку с данными или файл датасета .db")
            return

        model_id = self._model_cb.currentData()
        epochs   = self._ep_spin.value()
        batch    = self._bs_spin.value()
        lr       = self._lr_spin.value()
        val_spl  = self._val_spin.value()
        backbone = self._backbone_cb.currentText()

        self._chart.reset()
        self._log.clear()
        self._btn_run.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._epoch_bar.setRange(0, epochs)
        self._epoch_bar.setValue(0)
        self._epoch_bar.setVisible(True)
        self._batch_bar.setVisible(True)

        self._worker = TrainerWorker(
            model_id=model_id, data_dir=data_dir,
            registry=self._registry, epochs=epochs,
            batch_size=batch, lr=lr,
            val_split=val_spl, backbone=backbone,
        )
        self._thread = QThread()
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._on_log)
        self._worker.epoch_done.connect(self._on_epoch)
        self._worker.batch_done.connect(self._on_batch)
        self._worker.finished.connect(self._on_done)
        self._thread.finished.connect(self._worker.deleteLater)

        self._thread.start()

    def _on_stop(self):
        if self._worker:
            self._worker.request_stop()

    # ── Слоты ──────────────────────────────────────────────────────────────────

    @Slot(str)
    def _on_log(self, msg: str):
        self._log.addItem(msg)
        self._log.scrollToBottom()

    @Slot(int, int, dict)
    def _on_epoch(self, epoch: int, total: int, metrics: dict):
        self._epoch_bar.setValue(epoch)
        self._epoch_lbl.setText(
            f"Эпоха {epoch}/{total}  "
            f"loss {metrics['train_loss']:.4f}→{metrics['val_loss']:.4f}  "
            f"acc {metrics['train_acc']:.1%}→{metrics['val_acc']:.1%}"
        )
        self._chart.add_epoch(epoch, metrics)
        # Сбросить прогресс батчей для следующей эпохи
        self._batch_bar.setValue(0)
        # Обновить информацию о модели (может стать обученной)
        self._on_model_change()

    @Slot(int, int)
    def _on_batch(self, batch: int, total: int):
        self._batch_bar.setRange(0, total)
        self._batch_bar.setValue(batch)

    @Slot(bool, str)
    def _on_done(self, success: bool, msg: str):
        self._btn_run.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._epoch_bar.setVisible(False)
        self._batch_bar.setVisible(False)
        self._epoch_lbl.setText("Готово ✓" if success else "Остановлено")
        if self._thread:
            self._thread.quit()
            self._thread.wait(3000)
        self._on_model_change()
