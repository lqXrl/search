from flask import Flask, request, render_template, jsonify, send_from_directory
from pathlib import Path
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

from vision_desktop_app.training.predictor import load_model, predict_image, render_prediction
from vision_desktop_app.utils.bulk_import import sort_images_by_model

app = Flask(__name__, static_folder='static', template_folder='templates')

# пул потоков для выполнения тяжёлых операций без блокировки основного потока
_executor = ThreadPoolExecutor(max_workers=4)

# Create static and templates if not exists
Path('web/static').mkdir(parents=True, exist_ok=True)
Path('web/templates').mkdir(parents=True, exist_ok=True)


@app.route('/')
def index():
    # show simple UI: home with upload and model selection
    models_dir = Path('models')
    models = [p.name for p in models_dir.iterdir() if p.is_dir()]
    return render_template('index.html', models=models)


@app.route('/api/upload', methods=['POST'])
async def upload_file():
    """Асинхронная загрузка файла и запуск предсказания в пуле потоков."""
    model_name = request.form.get('model')
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Файл не выбран'}), 400

    upload_dir = Path('web_uploads')
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / file.filename
    file.save(path)

    model_dir = os.path.join('models', model_name, 'saved_model')

    def _predict_sync():
        model = load_model(model_dir)
        if not model:
            return None
        pred = predict_image(model, str(path), input_size=(128, 128))
        render_prediction(str(path), pred, output_path=str(path.with_name('pred_' + path.name)))
        return True

    loop = asyncio.get_running_loop()
    ok = await loop.run_in_executor(_executor, _predict_sync)
    if not ok:
        return jsonify({'error': 'Модель не найдена'}), 404
    return jsonify({'success': True, 'result': 'Готово'})


@app.route('/api/bulk_import', methods=['POST'])
async def api_bulk_import():
    """Асинхронный bulk-import, сортировка изображений выполняется в пуле потоков."""
    folder = request.form.get('folder')
    model_name = request.form.get('model')
    if not folder or not model_name:
        return jsonify({'error': 'отсутствуют параметры (folder/model)'}, 400)

    model_dir = os.path.join('models', model_name, 'saved_model')

    def _bulk_sync():
        return sort_images_by_model(folder, model_dir, move_files=True, output_root='data/sorted')

    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(_executor, _bulk_sync)
    return jsonify({'summary': res, 'message': 'Готово'})


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('web/static', filename)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=False)
