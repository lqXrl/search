import os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


def load_model(model_dir):
    if os.path.exists(model_dir):
        return tf.keras.models.load_model(model_dir)
    return None


def predict_image(model, image_path, input_size=(128,128)):
    img = Image.open(image_path).convert('RGB').resize(input_size)
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, 0)
    res = model.predict(arr)
    return res


def render_prediction(image_path, prediction, output_path=None, save_json=False, json_path=None):
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    if isinstance(prediction, list) or isinstance(prediction, tuple):
        # detection: [class_probs, bbox]
        class_probs, bbox = prediction
        class_idx = int(np.argmax(class_probs[0]))
        conf = float(np.max(class_probs[0]))
        # bbox here is normalized x,y,w,h
        w, h = img.size
        x = bbox[0][0] * w
        y = bbox[0][1] * h
        ww = bbox[0][2] * w
        hh = bbox[0][3] * h
        draw.rectangle([x, y, x+ww, y+hh], outline=(255,0,0), width=3)
        draw.text((x, y-12), f'class:{class_idx}, conf:{conf:.2f}', fill=(255,0,0))
    else:
        # classification
        probs = prediction[0]
        idx = int(np.argmax(probs))
        conf = float(np.max(probs))
        draw.text((10, 10), f'class:{idx}, conf:{conf:.2f}', fill=(255,0,0))
    if output_path:
        img.save(output_path)
    if save_json and json_path:
        # save predicted boxes as json
        if isinstance(prediction, list) or isinstance(prediction, tuple):
            class_probs, bbox = prediction
            class_idx = int(np.argmax(class_probs[0]))
            conf = float(np.max(class_probs[0]))
            j = {'predictions': [{'class': int(class_idx), 'confidence': conf, 'bbox': [float(bbox[0][0]), float(bbox[0][1]), float(bbox[0][2]), float(bbox[0][3])] }]}
        else:
            probs = prediction[0]
            idx = int(np.argmax(probs))
            conf = float(np.max(probs))
            j = {'predictions': [{'class': idx, 'confidence': conf}]}
        import json
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(j, f, ensure_ascii=False, indent=2)
    return img
