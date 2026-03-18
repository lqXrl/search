"""
Асинхронный тренер — выполняется как QObject в QThread.

Сигналы:
  log(str)                      — текстовое сообщение в лог
  epoch_done(int, int, dict)    — (эпоха, всего_эпох, словарь_метрик)
  batch_done(int, int)          — (индекс_батча, всего_батчей) для прогресс-бара
  finished(bool, str)           — (успех, сообщение)

Ключи словаря метрик: train_loss, train_acc, val_loss, val_acc
"""

from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PySide6.QtCore import QObject, Signal, Slot

from config import DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_VAL_SPLIT
from app.models.torch_model import build_model, get_device
from app.models.registry import ModelRegistry
from app.core.dataset import build_datasets


class TrainerWorker(QObject):
    log        = Signal(str)
    epoch_done = Signal(int, int, dict)   # эпоха, всего, метрики
    batch_done = Signal(int, int)         # батч, всего_батчей
    finished   = Signal(bool, str)        # успех, сообщение

    def __init__(
        self,
        model_id:   str,
        data_dir:   str,
        registry:   ModelRegistry,
        epochs:     int   = DEFAULT_EPOCHS,
        batch_size: int   = DEFAULT_BATCH_SIZE,
        lr:         float = DEFAULT_LR,
        val_split:  float = DEFAULT_VAL_SPLIT,
        backbone:   str   = "resnet18",
        fine_tune:  bool  = True,
    ):
        super().__init__()
        self.model_id   = model_id
        self.data_dir   = data_dir
        self.registry   = registry
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.val_split  = val_split
        self.backbone   = backbone
        self.fine_tune  = fine_tune
        self._stop      = False

    def request_stop(self):
        self._stop = True

    @Slot()
    def run(self):
        try:
            self._run()
        except Exception as e:
            self.log.emit(f"[ERROR] {e}")
            self.finished.emit(False, str(e))

    def _run(self):
        device = get_device()
        self.log.emit(f"Устройство: {device}")

        # ── Построение датасетов ──────────────────────────────────────────────
        self.log.emit("Загрузка датасета…")
        train_ds, val_ds = build_datasets(
            folder=self.data_dir,
            model_id=self.model_id,
            val_split=self.val_split,
        )
        self.log.emit(
            f"Датасет: {len(train_ds)} train / {len(val_ds)} val  "
            f"| {train_ds.dataset.num_classes} классов: "
            f"{train_ds.dataset.class_names}"
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=0
        )

        # ── Построение модели ─────────────────────────────────────────────────
        num_classes = train_ds.dataset.num_classes
        model = build_model(num_classes=num_classes, backbone=self.backbone)

        if self.fine_tune:
            model.freeze_backbone()
            model.unfreeze_last_n(2)

        model = model.to(device)
        self.log.emit(f"Модель: {self.backbone}  {num_classes} классов")

        # ── Оптимизатор и планировщик ─────────────────────────────────────────
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-6
        )
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0

        for epoch in range(1, self.epochs + 1):
            if self._stop:
                self.log.emit("Обучение остановлено пользователем.")
                break

            t0 = time.time()

            # ── Обучение ──────────────────────────────────────────────────────
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for bi, (imgs, labels) in enumerate(train_loader):
                if self._stop:
                    break
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                out  = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                train_loss    += loss.item() * imgs.size(0)
                preds          = out.argmax(dim=1)
                train_correct += (preds == labels).sum().item()
                train_total   += imgs.size(0)
                self.batch_done.emit(bi + 1, len(train_loader))

            # ── Валидация ─────────────────────────────────────────────────────
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    out  = model(imgs)
                    loss = criterion(out, labels)
                    val_loss    += loss.item() * imgs.size(0)
                    preds        = out.argmax(dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total   += imgs.size(0)

            scheduler.step()

            t_train = train_loss / train_total if train_total else 0
            a_train = train_correct / train_total if train_total else 0
            t_val   = val_loss / val_total if val_total else 0
            a_val   = val_correct / val_total if val_total else 0
            elapsed = time.time() - t0

            metrics = {
                "train_loss": round(t_train, 4),
                "train_acc":  round(a_train, 4),
                "val_loss":   round(t_val, 4),
                "val_acc":    round(a_val, 4),
                "lr":         round(scheduler.get_last_lr()[0], 6),
                "epoch_time": round(elapsed, 1),
            }
            self.epoch_done.emit(epoch, self.epochs, metrics)
            self.log.emit(
                f"Эп {epoch:>3}/{self.epochs}  "
                f"loss {t_train:.4f}→{t_val:.4f}  "
                f"acc {a_train:.1%}→{a_val:.1%}  "
                f"lr {metrics['lr']}  {elapsed:.1f}s"
            )

            # ── Сохранение лучшей модели ──────────────────────────────────────
            if a_val >= best_val_acc:
                best_val_acc = a_val
                self.registry.save(self.model_id, model, {
                    "best_val_acc": best_val_acc,
                    "epoch":        epoch,
                    "class_names":  train_ds.dataset.class_names,
                })
                self.log.emit(f"  ✓ Сохранена лучшая модель  val_acc={a_val:.1%}")

        self.log.emit(f"Обучение завершено. Лучший val_acc: {best_val_acc:.1%}")
        self.finished.emit(True, f"best_val_acc={best_val_acc:.1%}")
