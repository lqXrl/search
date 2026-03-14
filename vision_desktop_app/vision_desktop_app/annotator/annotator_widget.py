from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QFileDialog, QListWidget, QMessageBox, QComboBox, QMenu, QApplication
from PySide6.QtGui import QPixmap, QMouseEvent, QPainter, QPen, QColor, QAction
from PySide6.QtCore import Qt, QRect, QPoint, Signal

import json
import os

class ImageLabel(QLabel):
    """ QLabel that supports drawing bounding boxes and mapping coordinates to original image size """
    rectAdded = Signal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pixmap_orig = None
        self.rects = []
        self.drawing = False
        self.start_pos = None
        self.current_rect = None

    def setPixmap(self, pixmap: QPixmap):
        super().setPixmap(pixmap)
        self.pixmap_orig = pixmap

    def mousePressEvent(self, event: QMouseEvent):
        if not self.pixmap():
            return
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_pos = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.drawing:
            return
        current_pos = event.position().toPoint()
        self.current_rect = QRect(self.start_pos, current_pos).normalized()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self.drawing:
            return
        if event.button() == Qt.LeftButton:
            self.drawing = False
            end_pos = event.position().toPoint()
            rect = QRect(self.start_pos, end_pos).normalized()
            self.rects.append(rect)
            # emit a signal so that AnnotatorWidget can record label for this rect
            try:
                self.rectAdded.emit(rect)
            except Exception:
                pass
            self.current_rect = None
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.pixmap():
            return
        painter = QPainter(self)
        # Pen for saved rects
        pen = QPen(QColor(0, 200, 0), 2)
        painter.setPen(pen)
        for r in self.rects:
            painter.drawRect(r)
        if self.current_rect:
            pen = QPen(QColor(200, 0, 0), 2)
            painter.setPen(pen)
            painter.drawRect(self.current_rect)
        painter.end()

    def clear(self):
        self.rects = []
        self.update()


class AnnotatorWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)

        self.image_label = ImageLabel('')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.image_label)

        # list of loaded images and navigation
        self.images_list_widget = QListWidget()
        self.images_list_widget.setMaximumHeight(120)
        self.layout.addWidget(self.images_list_widget)
        # CSV toolbar (filter)
        csv_toolbar = QHBoxLayout()
        csv_toolbar.addWidget(QLabel('CSV:'))
        self.csv_filter = QComboBox()
        self.csv_filter.addItems(['All', 'Present', 'Missing'])
        csv_toolbar.addWidget(self.csv_filter)
        csv_toolbar.addStretch()
        self.layout.addLayout(csv_toolbar)
        # CSV entries list (to map CSV rows to images)
        self.csv_list_widget = QListWidget()
        self.csv_list_widget.setMaximumHeight(120)
        # enable context menu
        self.csv_list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.layout.addWidget(self.csv_list_widget)

        controls_layout = QHBoxLayout()
        self.load_btn = QPushButton('Загрузить изображение')
        self.prev_btn = QPushButton('◀')
        self.next_btn = QPushButton('▶')
        self.save_annotation_btn = QPushButton('Сохранить аннотацию')
        self.label_line = QLineEdit()
        self.label_line.setPlaceholderText('Название метки')
        controls_layout.addWidget(self.load_btn)
        controls_layout.addWidget(self.prev_btn)
        controls_layout.addWidget(self.next_btn)
        controls_layout.addWidget(self.label_line)
        controls_layout.addWidget(self.save_annotation_btn)

        self.layout.addLayout(controls_layout)

        # Connect
        self.load_btn.clicked.connect(self.on_load_image)
        self.prev_btn.clicked.connect(self.on_prev_image)
        self.next_btn.clicked.connect(self.on_next_image)
        self.save_annotation_btn.clicked.connect(self.on_save_annotation)
        # connect rectAdded signal to map label to rect
        self.image_label.rectAdded.connect(self._on_rect_added)

        # State for annotation
        self.pixmap = None
        self.image_path = None
        self.label_map = {}  # rect str -> label
        self.images = []
        self.current_index = -1
        # CSV entries and mapping
        self._csv_entries = []
        self._csv_resolved = []
        # mapping from csv row index -> resolved image path or None
        self._csv_to_image = {}
        # connect filter change and context menu
        self.csv_filter.currentIndexChanged.connect(self._apply_csv_filter)
        self.csv_list_widget.customContextMenuRequested.connect(self._on_csv_context_menu)

    def on_load_image(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, 'Выберите изображение(я)', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if filenames:
            self.load_images(filenames)

    # load_image is defined below (kept there)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.pixmap:
            self.image_label.setPixmap(self.pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # on_save_annotation is implemented below (kept there)

    def load_image(self, filename):
        self.image_path = filename
        self.pixmap = QPixmap(filename)
        if self.pixmap.isNull():
            QMessageBox.warning(self, 'Ошибка', 'Не удалось загрузить изображение')
            return
        # set to label
        scaled = self.pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.label_map = {}
        self.image_label.clear()
        self.update()

    def load_images(self, filenames):
        # store absolute paths
        self.images = [str(f) for f in filenames]
        self.images_list_widget.clear()
        for f in self.images:
            self.images_list_widget.addItem(f)
        # connect click
        self.images_list_widget.currentRowChanged.connect(self.load_image_by_index)
        # load first
        if self.images:
            self.load_image_by_index(0)

    def load_csv_list(self, entries, resolved_paths=None):
        """Populate CSV list widget. `entries` is list of original CSV strings.
        `resolved_paths` is list of same length with absolute path or None if missing.
        Clicking a CSV row will open the corresponding image if resolved.
        """
        self._csv_entries = list(entries)
        self._csv_resolved = list(resolved_paths) if resolved_paths is not None else [None] * len(entries)
        self._csv_to_image = {i: (self._csv_resolved[i] if i < len(self._csv_resolved) else None) for i in range(len(self._csv_entries))}
        # populate according to filter
        self._apply_csv_filter()
        # connect click
        self.csv_list_widget.itemClicked.connect(self._on_csv_item_clicked)

    def _apply_csv_filter(self):
        # rebuild list based on filter selection
        mode = self.csv_filter.currentText()
        self.csv_list_widget.clear()
        for i, text in enumerate(self._csv_entries):
            resolved = self._csv_resolved[i] if i < len(self._csv_resolved) else None
            ok = bool(resolved)
            if mode == 'Present' and not ok:
                continue
            if mode == 'Missing' and ok:
                continue
            display = f"{text}"
            item = self.csv_list_widget.addItem(display)
            # color items: green for present, red for missing
            qitem = self.csv_list_widget.item(self.csv_list_widget.count() - 1)
            if ok:
                qitem.setForeground(QColor(0, 128, 0))
            else:
                qitem.setForeground(QColor(200, 0, 0))
            # store mapping in widget's data
            self._csv_to_image[self.csv_list_widget.count() - 1] = resolved

    def _on_csv_item_clicked(self, item):
        row = self.csv_list_widget.row(item)
        # Need to map displayed row index back to original entry index
        # We stored mapping in _csv_to_image using display index; retrieve target
        target = self._csv_to_image.get(row)
        if not target:
            QMessageBox.information(self, 'Файл не найден', 'Соответствующее изображение не найдено на диске')
            return
        # if target is in loaded images, load that index, else add it and load
        try:
            idx = self.images.index(target)
        except ValueError:
            # add to images list
            self.images.append(target)
            self.images_list_widget.addItem(target)
            idx = len(self.images) - 1
        self.load_image_by_index(idx)

    def _on_csv_context_menu(self, pos):
        item = self.csv_list_widget.itemAt(pos)
        if item is None:
            return
        row = self.csv_list_widget.row(item)
        mapped = self._csv_to_image.get(row)
        menu = QMenu(self)
        act_open = QAction('Open image', self)
        act_show = QAction('Show in Explorer', self)
        act_copy = QAction('Copy path', self)
        act_remove = QAction('Remove entry', self)
        menu.addAction(act_open)
        menu.addAction(act_show)
        menu.addAction(act_copy)
        menu.addSeparator()
        menu.addAction(act_remove)

        def do_open():
            if not mapped:
                QMessageBox.information(self, 'Файл не найден', 'Соответствующее изображение не найдено на диске')
                return
            try:
                if mapped in self.images:
                    idx = self.images.index(mapped)
                else:
                    self.images.append(mapped)
                    self.images_list_widget.addItem(mapped)
                    idx = len(self.images) - 1
                self.load_image_by_index(idx)
            except Exception as e:
                QMessageBox.warning(self, 'Ошибка', str(e))

        def do_show():
            if not mapped:
                QMessageBox.information(self, 'Файл не найден', 'Соответствующее изображение не найдено на диске')
                return
            try:
                # platform-specific: use os.startfile on Windows, QDesktopServices otherwise
                try:
                    os.startfile(mapped)
                except Exception:
                    from PySide6.QtGui import QDesktopServices
                    from PySide6.QtCore import QUrl
                    QDesktopServices.openUrl(QUrl.fromLocalFile(mapped))
            except Exception as e:
                QMessageBox.warning(self, 'Ошибка', str(e))

        def do_copy():
            cb = QApplication.clipboard()
            cb.setText(mapped or '')

        def do_remove():
            # find original index in entries
            disp_idx = row
            # remove from internal lists if possible
            try:
                # find corresponding original index by matching entry text
                entry_text = item.text()
                # remove first matching entry
                for i, t in enumerate(self._csv_entries):
                    if entry_text.endswith(t) or entry_text == t:
                        del self._csv_entries[i]
                        if i < len(self._csv_resolved):
                            del self._csv_resolved[i]
                        break
            except Exception:
                pass
            # refresh
            self._apply_csv_filter()

        act_open.triggered.connect(do_open)
        act_show.triggered.connect(do_show)
        act_copy.triggered.connect(do_copy)
        act_remove.triggered.connect(do_remove)

        menu.exec(self.csv_list_widget.mapToGlobal(pos))

    def load_image_by_index(self, idx):
        try:
            idx = int(idx)
        except Exception:
            return
        if idx < 0 or idx >= len(self.images):
            return
        self.current_index = idx
        filename = self.images[idx]
        self.images_list_widget.setCurrentRow(idx)
        self.load_image(filename)

    def on_prev_image(self):
        if not self.images:
            return
        idx = max(0, (self.current_index or 0) - 1)
        self.load_image_by_index(idx)

    def on_next_image(self):
        if not self.images:
            return
        idx = min(len(self.images) - 1, (self.current_index or 0) + 1)
        self.load_image_by_index(idx)

    def _on_rect_added(self, rect: QRect):
        # associate current label for the newly added rect
        key = f"{rect.x()}_{rect.y()}_{rect.width()}_{rect.height()}"
        label = self.label_line.text().strip() or 'object'
        self.label_map[key] = label

    def on_save_annotation(self):
        if not self.image_path:
            QMessageBox.warning(self, 'Ошибка', 'Нет загруженного изображения')
            return
        ann = []
        for r in self.image_label.rects:
            key = f"{r.x()}_{r.y()}_{r.width()}_{r.height()}"
            label = self.label_map.get(key, self.label_line.text().strip() or 'object')
            # сохраняем как метку, так и явное название объекта, заданное учителем
            ann.append({
                'label': label,
                'object_name': label,
                'x': r.x(),
                'y': r.y(),
                'w': r.width(),
                'h': r.height()
            })
        out_path = os.path.splitext(self.image_path)[0] + '.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump({'image': os.path.basename(self.image_path), 'annotations': ann}, f, ensure_ascii=False, indent=2)
        QMessageBox.information(self, 'Сохранено', f'Аннотация сохранена: {out_path}')
