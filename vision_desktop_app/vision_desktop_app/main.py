import sys
from PySide6 import QtWidgets
from PySide6.QtWidgets import QMessageBox
from vision_desktop_app.ui.main_window import MainWindow
import subprocess
import webbrowser
import threading
from pathlib import Path


def main():
    # show a simple dialog that asks for Desktop or Web
    app = QtWidgets.QApplication(sys.argv)
    choice = QMessageBox()
    choice.setWindowTitle('Vision App: Choose mode')
    choice.setText('Выберите режим работы:')
    choice.setStandardButtons(QMessageBox.Cancel)
    desktop = choice.addButton('Desktop', QMessageBox.YesRole)
    web = choice.addButton('Web (localhost)', QMessageBox.NoRole)
    choice.exec()
    if choice.clickedButton() == web:
        # start web app in separate thread and open browser
        def run_server():
            from vision_desktop_app.web.app import app as webapp
            webapp.run(host='127.0.0.1', port=8000)
        t = threading.Thread(target=run_server, daemon=True)
        t.start()
        webbrowser.open('http://127.0.0.1:8000')
        # show a small message box
        QMessageBox.information(None, 'Web mode', 'Web server запущен, откройте http://127.0.0.1:8000')
        sys.exit(0)
    else:
        print('Starting Vision Desktop App (Desktop mode)...')
        main_window = MainWindow()
        main_window.show()
        print('Main window shown.')
        sys.exit(app.exec())


if __name__ == '__main__':
    main()
