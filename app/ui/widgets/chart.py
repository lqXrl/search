"""
MetricsChart — визуализация метрик обучения в реальном времени.

Встраивает matplotlib-фигуру (2 × 1 подграфика: Loss | Accuracy) в Qt-виджет.
Вызывайте add_epoch(metrics_dict) после каждой эпохи для обновления.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as ticker

from PySide6.QtWidgets import QVBoxLayout, QWidget

_DARK_BG  = "#1e1e2e"
_AXES_BG  = "#282a36"
_GRID     = "#44475a"
_TEXT     = "#f8f8f2"
_TRAIN    = "#8be9fd"
_VAL      = "#ff79c6"


class MetricsChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._epochs:      list[int]   = []
        self._train_loss:  list[float] = []
        self._val_loss:    list[float] = []
        self._train_acc:   list[float] = []
        self._val_acc:     list[float] = []

        self.fig = Figure(figsize=(9, 3.5), facecolor=_DARK_BG, tight_layout=True)
        self.ax_loss = self.fig.add_subplot(1, 2, 1)
        self.ax_acc  = self.fig.add_subplot(1, 2, 2)
        self._style_axes()

        self.canvas = FigureCanvas(self.fig)
        self.canvas.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    # ── Стиль ──────────────────────────────────────────────────────────────────

    def _style_axes(self):
        for ax in (self.ax_loss, self.ax_acc):
            ax.set_facecolor(_AXES_BG)
            ax.tick_params(colors=_TEXT, labelsize=8)
            ax.title.set_color(_TEXT)
            ax.xaxis.label.set_color(_TEXT)
            ax.yaxis.label.set_color(_TEXT)
            for spine in ax.spines.values():
                spine.set_edgecolor(_GRID)
            ax.grid(True, color=_GRID, linewidth=0.5, linestyle="--")

        self.ax_loss.set_title("Loss", fontsize=10, fontweight="bold")
        self.ax_loss.set_xlabel("Эпоха")
        self.ax_loss.set_ylabel("Loss")

        self.ax_acc.set_title("Accuracy", fontsize=10, fontweight="bold")
        self.ax_acc.set_xlabel("Эпоха")
        self.ax_acc.set_ylabel("Accuracy")
        self.ax_acc.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

    # ── Публичные методы ───────────────────────────────────────────────────────

    def add_epoch(self, epoch: int, metrics: dict):
        self._epochs.append(epoch)
        self._train_loss.append(metrics.get("train_loss", 0))
        self._val_loss.append(metrics.get("val_loss", 0))
        self._train_acc.append(metrics.get("train_acc", 0))
        self._val_acc.append(metrics.get("val_acc", 0))
        self._redraw()

    def reset(self):
        self._epochs.clear()
        self._train_loss.clear()
        self._val_loss.clear()
        self._train_acc.clear()
        self._val_acc.clear()
        for ax in (self.ax_loss, self.ax_acc):
            ax.cla()
        self._style_axes()
        self.canvas.draw()

    # ── Приватные методы ───────────────────────────────────────────────────────

    def _redraw(self):
        ep = self._epochs

        self.ax_loss.cla()
        self.ax_acc.cla()
        self._style_axes()

        kw = dict(linewidth=2, markersize=4, marker="o")
        self.ax_loss.plot(ep, self._train_loss, color=_TRAIN, label="Train", **kw)
        self.ax_loss.plot(ep, self._val_loss,   color=_VAL,   label="Val",   **kw)
        self.ax_loss.legend(facecolor=_AXES_BG, labelcolor=_TEXT, fontsize=8)

        self.ax_acc.plot(ep, self._train_acc, color=_TRAIN, label="Train", **kw)
        self.ax_acc.plot(ep, self._val_acc,   color=_VAL,   label="Val",   **kw)
        self.ax_acc.legend(facecolor=_AXES_BG, labelcolor=_TEXT, fontsize=8)
        self.ax_acc.set_ylim(0, 1.05)

        # Аннотация последней точки
        if self._epochs:
            last_e = self._epochs[-1]
            for ax, series, color in [
                (self.ax_loss, self._train_loss, _TRAIN),
                (self.ax_loss, self._val_loss,   _VAL),
                (self.ax_acc,  self._train_acc,  _TRAIN),
                (self.ax_acc,  self._val_acc,    _VAL),
            ]:
                val = series[-1]
                fmt = f"{val:.4f}" if ax is self.ax_loss else f"{val:.1%}"
                ax.annotate(
                    fmt, xy=(last_e, val),
                    xytext=(3, 3), textcoords="offset points",
                    fontsize=7, color=color
                )

        self.canvas.draw()
