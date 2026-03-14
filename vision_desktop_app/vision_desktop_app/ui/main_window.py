from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QHBoxLayout, QFileDialog, QListWidget, QTabWidget, QSpinBox, QLineEdit, QMessageBox
from PySide6.QtCore import Qt, QThread, Signal, QObject, Slot
from PySide6.QtGui import QPixmap
import os

from vision_desktop_app.annotator.annotator_widget import AnnotatorWidget
from vision_desktop_app.training.trainer import Trainer
from vision_desktop_app.models.model_manager import ModelManager
from vision_desktop_app.utils.bulk_import import sort_images_by_model
from vision_desktop_app.utils.csv_utils import csv_first_column_as_paths

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Vision Desktop App — распознавание объектов')
        self.resize(1100, 700)
        # enforce a reasonable minimum size to avoid platform geometry warnings
        try:
            self.setMinimumSize(800, 600)
        except Exception:
            pass

        self.central = QWidget()
        self.setCentralWidget(self.central)
        self.layout = QVBoxLayout(self.central)

        # Top controls
        controls_layout = QHBoxLayout()
        self.model_select = QComboBox()
        self.model_manager = ModelManager(root_dir='models')
        self.model_select.addItems(self.model_manager.list_models())
        controls_layout.addWidget(QLabel('Модель:'))
        controls_layout.addWidget(self.model_select)

        self.load_image_btn = QPushButton('Загрузить фото')
        controls_layout.addWidget(self.load_image_btn)
        self.load_csv_btn = QPushButton('Загрузить список (CSV)')
        controls_layout.addWidget(self.load_csv_btn)
        self.auto_annotate_btn = QPushButton('Авто-аннотации (CSV)')
        controls_layout.addWidget(self.auto_annotate_btn)

        self.train_btn = QPushButton('Обучить модель')
        controls_layout.addWidget(self.train_btn)

        self.export_btn = QPushButton('Экспорт предсказаний')
        controls_layout.addWidget(self.export_btn)

        self.bulk_import_btn = QPushButton('Bulk Import')
        controls_layout.addWidget(self.bulk_import_btn)

        controls_layout.addStretch()
        self.layout.addLayout(controls_layout)

        # Main tabs
        self.tabs = QTabWidget()
        self.annotator_tab = AnnotatorWidget()
        self.training_info_tab = QWidget()
        self.prediction_tab = QWidget()

        self.tabs.addTab(self.annotator_tab, 'Аннотация')
        self.tabs.addTab(self.training_info_tab, 'Обучение')
        self.tabs.addTab(self.prediction_tab, 'Предсказание')

        self.layout.addWidget(self.tabs)

        # Training controls
        t_layout = QVBoxLayout(self.training_info_tab)
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel('Эпохи:'))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        params_layout.addWidget(self.epochs_spin)
        params_layout.addWidget(QLabel('Batch:'))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        params_layout.addWidget(self.batch_spin)
        t_layout.addLayout(params_layout)

        # model dir
        m_layout = QHBoxLayout()
        m_layout.addWidget(QLabel('Save model dir:'))
        self.model_dir_line = QLineEdit('models/' + self.model_select.currentText())
        def on_model_change(idx):
            m = self.model_select.currentText()
            self.model_dir_line.setText('models/' + m)
            cfg = self.model_manager.get_model_config(m)
            if cfg:
                self.model_params_label.setText(f"Model params: type={cfg.get('type')}, num_classes={cfg.get('num_classes')}")
            else:
                self.model_params_label.setText('Model params: unknown')
        self.model_select.currentIndexChanged.connect(on_model_change)
        m_layout.addWidget(self.model_dir_line)
        t_layout.addLayout(m_layout)
        # model parameters readout
        self.model_params_label = QLabel('Model params:')
        t_layout.addWidget(self.model_params_label)

        self.start_train_btn = QPushButton('Запустить обучение')
        t_layout.addWidget(self.start_train_btn)

        self.train_log = QListWidget()
        t_layout.addWidget(self.train_log)

        # Bindings
        self.load_image_btn.clicked.connect(self.open_image)
        self.load_csv_btn.clicked.connect(self.on_load_csv)
        self.auto_annotate_btn.clicked.connect(self.on_auto_annotate_csv)
        self.start_train_btn.clicked.connect(self.on_start_train)
        self.bulk_import_btn.clicked.connect(self.on_bulk_import)

        # thread references dictionary to avoid premature GC
        self._threads = []

        # keep explicit reference to trainer thread if used
        self._trainer_thread = None

    # Generic async runner
    def run_in_thread(self, fn, args=(), kwargs=None, on_done=None, on_error=None):
        if kwargs is None:
            kwargs = {}

        class Worker(QObject):
            finished = Signal(object)
            error = Signal(str)

            @Slot()
            def run(self):
                try:
                    res = fn(*args, **kwargs)
                    self.finished.emit(res)
                except Exception as e:
                    self.error.emit(str(e))

        thread = QThread()
        worker = Worker()
        worker.moveToThread(thread)
        worker.finished.connect(lambda res: (on_done(res) if on_done else None, thread.quit()))
        worker.error.connect(lambda e: (on_error(e) if on_error else None, thread.quit()))
        thread.started.connect(worker.run)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(lambda: self._threads.remove(thread) if thread in self._threads else None)
        self._threads.append(thread)
        thread.start()
        return thread

    def open_image(self):
        filenames, _ = QFileDialog.getOpenFileNames(self, 'Выберите изображение(я)', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if filenames:
            try:
                self.annotator_tab.load_images(filenames)
            except Exception as e:
                QMessageBox.warning(self, 'Ошибка', f'Не удалось загрузить изображения: {e}')

    def on_load_csv(self):
        csv_path, _ = QFileDialog.getOpenFileName(self, 'Выберите CSV файл со списком изображений', '', 'CSV Files (*.csv);;All files (*)')
        if not csv_path:
            return
        # parse CSV in background
        def _read_csv(path):
            return csv_first_column_as_paths(path)

        def _on_done(res):
            paths, enc = res
            # Resolve relative paths relative to CSV file
            base = os.path.dirname(csv_path)
            resolved = []
            missing = []
            for p in paths:
                if not p:
                    continue
                candidate = p
                if not os.path.isabs(candidate):
                    candidate = os.path.join(base, candidate)
                if os.path.exists(candidate):
                    resolved.append(candidate)
                else:
                    missing.append(p)
            if not resolved:
                QMessageBox.warning(self, 'Ошибка', f'Ни одного файла не найдено. Пропущено: {len(missing)} строк')
                return
            # notify encoding used
            self.train_log.addItem(f'CSV loaded ({enc}), {len(resolved)} files, {len(missing)} missing')
            try:
                self.annotator_tab.load_images(resolved)
            except Exception as e:
                QMessageBox.warning(self, 'Ошибка', f'Не удалось загрузить изображения из CSV: {e}')
            # build aligned resolved list (same order as paths)
            resolved_aligned = []
            for p in paths:
                if not p:
                    resolved_aligned.append(None)
                    continue
                candidate = p
                if not os.path.isabs(candidate):
                    candidate = os.path.join(base, candidate)
                if os.path.exists(candidate):
                    resolved_aligned.append(candidate)
                else:
                    resolved_aligned.append(None)
            try:
                self.annotator_tab.load_csv_list(paths, resolved_paths=resolved_aligned)
            except Exception:
                pass

        def _on_error(e):
            QMessageBox.warning(self, 'Ошибка', f'Не удалось прочитать CSV: {e}')

        self.run_in_thread(_read_csv, args=(csv_path,), on_done=_on_done, on_error=_on_error)

    def on_auto_annotate_csv(self):
        csv_path, _ = QFileDialog.getOpenFileName(self, 'Выберите CSV файл с аннотациями', '', 'CSV Files (*.csv);;All files (*)')
        if not csv_path:
            return
        # Run annotation parsing and writing in background
        from vision_desktop_app.utils.csv_utils import read_annotations_csv

        def _do_annotate(path):
            ann_rows, enc = read_annotations_csv(path)
            base = os.path.dirname(path)
            written = 0
            missing = 0
            errors = []
            import json
            for r in ann_rows:
                fname = r.get('filename')
                if not fname:
                    continue
                if not os.path.isabs(fname):
                    candidate = os.path.join(base, fname)
                else:
                    candidate = fname
                if not os.path.exists(candidate):
                    missing += 1
                    continue
                ann = r.get('label') or ''
                x = r.get('x')
                y = r.get('y')
                w = r.get('w')
                h = r.get('h')
                ann_obj = {'label': ann}
                if None not in (x, y, w, h):
                    ann_obj.update({'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)})
                json_path = os.path.splitext(candidate)[0] + '.json'
                try:
                    if os.path.exists(json_path):
                        with open(json_path, 'r', encoding='utf-8') as fh:
                            data = json.load(fh)
                        annotations = data.get('annotations', [])
                    else:
                        annotations = []
                    annotations.append(ann_obj)
                    out = {'image': os.path.basename(candidate), 'annotations': annotations}
                    with open(json_path, 'w', encoding='utf-8') as fh:
                        json.dump(out, fh, ensure_ascii=False, indent=2)
                    written += 1
                except Exception as e:
                    errors.append((json_path, str(e)))
            return {'written': written, 'missing': missing, 'errors': errors, 'encoding': enc}

        def _on_done(res):
            self.train_log.addItem(f"Auto-annotate finished: written={res.get('written')}, missing_images={res.get('missing')}, encoding={res.get('encoding')}")
            for p, e in res.get('errors', []):
                self.train_log.addItem(f'Failed write {p}: {e}')
            QMessageBox.information(self, 'Готово', f"Авто-аннотирование завершено: записано {res.get('written')} файлов, пропущено {res.get('missing')} изображений")

        def _on_error(e):
            QMessageBox.warning(self, 'Ошибка', f'Не удалось прочитать CSV: {e}')

        self.run_in_thread(_do_annotate, args=(csv_path,), on_done=_on_done, on_error=_on_error)

    def on_start_train(self):
        model_name = self.model_select.currentText()
        epochs = int(self.epochs_spin.value())
        batch = int(self.batch_spin.value())
        model_dir = self.model_dir_line.text().strip() or 'models/default'
        # model types mapping
        if model_name in ['inside_outside', 'ground_surface', 'space_objects']:
            model_type = 'classification'
        else:
            model_type = 'detection'
        # open folder dialog for train data
        folder = QFileDialog.getExistingDirectory(self, 'Выберите папку с данными для обучения')
        if not folder:
            return
        self.train_log.addItem(f'Start training {model_name} on {folder} for {epochs} epochs')
        # Launch trainer in a thread to avoid UI block
        self.trainer_thread = QThread()
        self._trainer_thread = self.trainer_thread
        cfg = self.model_manager.get_model_config(model_name)
        num_classes = cfg.get('num_classes', 2) if cfg else 2
        self.trainer_worker = TrainerWorker(model_type=model_type, model_dir=model_dir, input_shape=(128,128,3), num_classes=num_classes)
        self.trainer_worker.moveToThread(self.trainer_thread)
        self.trainer_thread.started.connect(lambda: self.trainer_worker.run(folder, epochs, batch))
        self.trainer_worker.log_signal.connect(self.on_train_log)
        self.trainer_thread.start()

        # prediction tab controls
        p_layout = QVBoxLayout(self.prediction_tab)
        p_btn_layout = QHBoxLayout()
        self.load_pred_image_btn = QPushButton('Загрузить фото для предсказания')
        self.run_pred_btn = QPushButton('Сделать предсказание')
        self.export_pred_btn = QPushButton('Экспорт с метками')
        p_btn_layout.addWidget(self.load_pred_image_btn)
        p_btn_layout.addWidget(self.run_pred_btn)
        p_btn_layout.addWidget(self.export_pred_btn)
        p_layout.addLayout(p_btn_layout)

        self.pred_img_label = QLabel('No image')
        self.pred_img_label.setAlignment(Qt.AlignCenter)
        p_layout.addWidget(self.pred_img_label)

        self.load_pred_image_btn.clicked.connect(self.on_load_pred_image)
        self.run_pred_btn.clicked.connect(self.on_run_prediction)
        self.export_pred_btn.clicked.connect(self.on_export_prediction)

        self.pred_image_path = None
        self.prediction_result = None

    def on_load_pred_image(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Выберите изображение', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if filename:
            self.pred_image_path = filename
            pixmap = QPixmap(filename)
            if pixmap:
                self.pred_img_label.setPixmap(pixmap.scaled(self.pred_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def on_run_prediction(self):
        if not self.pred_image_path:
            return
        # load model
        model_name = self.model_select.currentText()
        model_dir = 'models/' + model_name
        # run prediction in background to avoid UI blocking
        from vision_desktop_app.training.predictor import load_model, predict_image, render_prediction

        def _predict(path):
            model = load_model(os.path.join(model_dir, 'saved_model'))
            if not model:
                raise FileNotFoundError(f'Model not found: {model_dir}')
            res = predict_image(model, path, input_size=(128,128))
            img = render_prediction(path, res)
            return {'res': res, 'img': img}

        def _on_done(res):
            try:
                self.train_log.addItem('Prediction finished')
                self.prediction_result = res.get('res')
                out = res.get('img')
                from PIL.ImageQt import ImageQt
                qim = ImageQt(out)
                pix = QPixmap.fromImage(qim)
                self.pred_img_label.setPixmap(pix.scaled(self.pred_img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            except Exception as e:
                # show and log any errors that happen while handling the result
                self.train_log.addItem(f'Prediction display error: {e}')
                QMessageBox.warning(self, 'Ошибка', f'Prediction display error: {e}')

        def _on_error(e):
            # worker-level errors
            self.train_log.addItem(f'Prediction error: {e}')
            QMessageBox.warning(self, 'Ошибка', str(e))

        self.train_log.addItem('Starting prediction...')
        self.run_in_thread(_predict, args=(self.pred_image_path,), on_done=_on_done, on_error=_on_error)

    def on_export_prediction(self):
        if not self.prediction_result or not self.pred_image_path:
            QMessageBox.warning(self, 'Ошибка', 'Нет предсказания для экспорта')
            return
        save_path, _ = QFileDialog.getSaveFileName(self, 'Сохранить результат как', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if not save_path:
            return
        from vision_desktop_app.training.predictor import render_prediction
        json_path = os.path.splitext(save_path)[0] + '.json'
        render_prediction(self.pred_image_path, self.prediction_result, output_path=save_path, save_json=True, json_path=json_path)
        QMessageBox.information(self, 'Готово', f'Экспорт сохранен: {save_path}')

    def on_bulk_import(self):
        # choose folder to import
        folder = QFileDialog.getExistingDirectory(self, 'Выберите папку с фотографиями для bulk import')
        if not folder:
            return
        # model and saved path
        model_name = self.model_select.currentText()
        model_dir = os.path.join('models', model_name, 'saved_model')
        self.train_log.addItem(f'Bulk import: sorting images from {folder} using model {model_name}...')
        def _do_bulk(src, mdir):
            return sort_images_by_model(src, mdir, move_files=True, output_root='data/sorted')

        def _on_done(res):
            n_ok = sum(1 for v in res.values() if 'error' not in v)
            n_err = sum(1 for v in res.values() if 'error' in v)
            self.train_log.addItem(f'Bulk import finished. {n_ok} OK, {n_err} errors.')

        def _on_error(e):
            QMessageBox.warning(self, 'Ошибка', f'Bulk import failed: {e}')

        self.run_in_thread(_do_bulk, args=(folder, model_dir), on_done=_on_done, on_error=_on_error)

    def on_train_log(self, msg):
        self.train_log.addItem(msg)

    def closeEvent(self, event):
        """Attempt to cleanly stop background threads before closing the window.
        This avoids 'QThread: Destroyed while thread is still running' warnings.
        """
        # stop threads started with run_in_thread
        for thread in list(self._threads):
            try:
                if thread.isRunning():
                    try:
                        thread.quit()
                    except Exception:
                        pass
                    # wait a short time
                    thread.wait(2000)
                    if thread.isRunning():
                        try:
                            thread.terminate()
                        except Exception:
                            pass
                        thread.wait(500)
            except Exception:
                pass

        # stop trainer thread if running
        try:
            t = getattr(self, 'trainer_thread', None) or getattr(self, '_trainer_thread', None)
            if t is not None and isinstance(t, QThread):
                if t.isRunning():
                    try:
                        t.quit()
                    except Exception:
                        pass
                    t.wait(2000)
                    if t.isRunning():
                        try:
                            t.terminate()
                        except Exception:
                            pass
                        t.wait(500)
        except Exception:
            pass

        event.accept()


class TrainerWorker(QObject):
    log_signal = Signal(str)

    def __init__(self, model_type='classification', model_dir='models/default', input_shape=(128,128,3), num_classes=2):
        super().__init__()
        self.model_type = model_type
        self.model_dir = model_dir
        self.input_shape = input_shape
        self.num_classes = num_classes

    def run(self, folder, epochs, batch):
        self.log_signal.emit('Building model...')
        try:
            tr = Trainer(model_type=self.model_type, model_dir=self.model_dir, input_shape=self.input_shape, num_classes=self.num_classes, logger=self.log_signal.emit)
            tr.build_model()
            self.log_signal.emit('Begin training...')
            tr.train(folder, epochs=epochs, batch_size=batch)
            self.log_signal.emit('Training finished')
        except Exception as e:
            # emit error to log_signal so UI can show it
            try:
                self.log_signal.emit(f'Training error: {e}')
            except Exception:
                pass
