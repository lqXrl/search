from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory
from pathlib import Path
import os
from vision_desktop_app.training.predictor import load_model, predict_image, render_prediction
from vision_desktop_app.utils.bulk_import import sort_images_by_model

app = Flask(__name__, static_folder='static', template_folder='templates')
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
def upload_file():
    model_name = request.form.get('model')
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Файл не выбран'}), 400
    upload_dir = Path('web_uploads')
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / file.filename
    file.save(path)
    # run prediction
    model = load_model(os.path.join('models', model_name, 'saved_model'))
    if not model:
        return jsonify({'error': 'Модель не найдена'}), 404
    pred = predict_image(model, str(path), input_size=(128,128))
    # optionally render and save
    out_img = render_prediction(str(path), pred, output_path=str(path.with_name('pred_' + path.name)))
    return jsonify({'success': True, 'result': 'Готово'})

@app.route('/api/bulk_import', methods=['POST'])
def api_bulk_import():
    folder = request.form.get('folder')
    model_name = request.form.get('model')
    if not folder or not model_name:
        return jsonify({'error': 'отсутствуют параметры (folder/model)'}, 400)
    model_dir = os.path.join('models', model_name, 'saved_model')
    res = sort_images_by_model(folder, model_dir, move_files=True, output_root='data/sorted')
    return jsonify({'summary': res, 'message': 'Готово'})

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('web/static', filename)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000, debug=False)
