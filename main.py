"""
Точка входа — Десктопное приложение Space Vision.

Запуск:  python main.py
"""

import sys
import traceback
from pathlib import Path

# Добавить корневую директорию проекта в sys.path
sys.path.insert(0, str(Path(__file__).parent))

_LOG = Path(__file__).parent / "crash.log"

def _excepthook(exc_type, exc_value, exc_tb):
    msg = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(msg, flush=True)
    with open(_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

sys.excepthook = _excepthook

from PySide6.QtWidgets import QApplication, QMessageBox
from PySide6.QtCore import Qt

from app.ui.main_window import MainWindow
from config import APP_TITLE


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_TITLE)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    try:
        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception:
        msg = traceback.format_exc()
        with open(_LOG, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        QMessageBox.critical(None, "Критическая ошибка", msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
