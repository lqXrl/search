"""
MainWindow — оболочка приложения.

Вкладки:
  🏷  Аннотация    — ручная разметка
  🗄  База данных  — авторазметка из таблицы
  ⚙   Обучение    — обучение с учителем с графиками в реальном времени
  🔍  Предсказание — инференс
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QLabel, QMainWindow, QStatusBar, QTabWidget, QWidget,
)

from app.models.registry import ModelRegistry
from app.ui.tabs.annotate_tab import AnnotateTab
from app.ui.tabs.db_tab import DBTab
from app.ui.tabs.train_tab import TrainTab
from app.ui.tabs.predict_tab import PredictTab
from config import APP_TITLE, APP_VERSION


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_TITLE}  v{APP_VERSION}")
        self.resize(1340, 840)
        self.setMinimumSize(1000, 700)

        self._registry = ModelRegistry()
        self._build_ui()
        self._apply_style()

    # ── Построение ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Вкладки
        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.North)
        self._tabs.setFont(QFont("Segoe UI", 10))
        self.setCentralWidget(self._tabs)

        # Создание вкладок
        self._ann_tab   = AnnotateTab()
        self._db_tab    = DBTab()
        self._train_tab = TrainTab(self._registry)
        self._pred_tab  = PredictTab(self._registry)

        self._tabs.addTab(self._ann_tab,   "🏷   Аннотация")
        self._tabs.addTab(self._db_tab,    "🗄   База данных")
        self._tabs.addTab(self._train_tab, "⚙   Обучение")
        self._tabs.addTab(self._pred_tab,  "🔍  Предсказание")

        # Строка состояния
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Готов")

        # Связи между вкладками
        self._ann_tab.images_changed.connect(self._db_tab.set_images)
        self._db_tab.status_message.connect(self._status.showMessage)
        self._train_tab._worker  # существует после _build_ui вкладки обучения

        # Обновить вкладку предсказания после завершения обучения
        self._train_tab._btn_run.clicked.connect(
            lambda: None  # только для отслеживания потока
        )
        # При завершении обучения обновить метки моделей в предсказании
        self._tabs.currentChanged.connect(self._on_tab_changed)

    def _on_tab_changed(self, idx: int):
        # Обновить список моделей в предсказании при переключении на эту вкладку
        if self._tabs.widget(idx) is self._pred_tab:
            self._pred_tab.refresh_models()

    # ── Стиль ──────────────────────────────────────────────────────────────────

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #f8f8f2;
                font-family: "Segoe UI", Arial, sans-serif;
            }
            QTabWidget::pane {
                border: 1px solid #44475a;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #282a36;
                color: #6272a4;
                padding: 8px 16px;
                border: 1px solid #44475a;
                border-bottom: none;
                border-radius: 4px 4px 0 0;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background: #44475a;
                color: #f8f8f2;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected { background: #383a4a; color: #f8f8f2; }
            QPushButton {
                background: #44475a;
                color: #f8f8f2;
                border: 1px solid #6272a4;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
            }
            QPushButton:hover  { background: #6272a4; }
            QPushButton:pressed{ background: #bd93f9; color: #1e1e2e; }
            QPushButton:disabled{ background: #282a36; color: #555; border-color: #44475a; }
            QComboBox {
                background: #282a36;
                color: #f8f8f2;
                border: 1px solid #44475a;
                border-radius: 4px;
                padding: 3px 6px;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #282a36;
                color: #f8f8f2;
                selection-background-color: #44475a;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background: #282a36;
                color: #f8f8f2;
                border: 1px solid #44475a;
                border-radius: 4px;
                padding: 3px 6px;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border-color: #bd93f9;
            }
            QListWidget, QTableWidget {
                background: #282a36;
                color: #f8f8f2;
                border: 1px solid #44475a;
                border-radius: 4px;
                gridline-color: #44475a;
            }
            QListWidget::item:selected, QTableWidget::item:selected {
                background: #44475a;
            }
            QHeaderView::section {
                background: #383a4a;
                color: #f8f8f2;
                border: none;
                padding: 4px;
                font-size: 11px;
            }
            QGroupBox {
                color: #bd93f9;
                border: 1px solid #44475a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 6px;
                font-weight: bold;
                font-size: 11px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QProgressBar {
                background: #282a36;
                border: 1px solid #44475a;
                border-radius: 3px;
                text-align: center;
                color: #f8f8f2;
                font-size: 10px;
            }
            QProgressBar::chunk { background: #bd93f9; border-radius: 2px; }
            QSplitter::handle { background: #44475a; }
            QStatusBar { background: #282a36; color: #6272a4; font-size: 11px; }
            QCheckBox { color: #f8f8f2; }
            QCheckBox::indicator:checked { background: #bd93f9; border-radius: 2px; }
            QScrollBar:vertical {
                background: #282a36;
                width: 10px;
            }
            QScrollBar::handle:vertical {
                background: #44475a;
                border-radius: 4px;
                min-height: 20px;
            }
        """)

    def closeEvent(self, event):
        # Остановить работающий поток обучения, если есть
        if hasattr(self._train_tab, "_thread") and self._train_tab._thread:
            t = self._train_tab._thread
            if t.isRunning():
                if self._train_tab._worker:
                    self._train_tab._worker.request_stop()
                t.quit()
                t.wait(4000)
        event.accept()
