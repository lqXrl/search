import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image


def load_annotation_json(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def parse_dataset_from_folder(folder, image_size=(128, 128), classification=False):
    """
    Finds images and their JSON annotations (image.json) and returns a tf.data.Dataset

    classification: if True, returns dataset for image-level classification (labels by folder or annotation); else detection dataset returns images and (class, bbox)
    """
    image_paths = []
    ann_paths = {}
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                path = os.path.join(root, file)
                image_paths.append(path)
                ann_file = os.path.splitext(path)[0] + '.json'
                if os.path.exists(ann_file):
                    ann_paths[path] = ann_file

    # build label mapping
    label_set = set()
    for p, a in ann_paths.items():
        data = load_annotation_json(a)
        for ann in data.get('annotations', []):
            label_set.add(ann['label'])
    label_list = sorted(list(label_set))
    label_to_idx = {l: i for i, l in enumerate(label_list)}

    def gen():
        for img_path in image_paths:
            # get original size to normalize bbox
            from PIL import Image
            img_pil = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img_pil.size
            arr = tf.io.read_file(img_path)
            img = tf.image.decode_image(arr, channels=3)
            img = tf.image.resize(img, image_size)
            img = tf.cast(img, tf.float32)
            if classification:
                # try load json to get class label (first annotation label)
                ann = ann_paths.get(img_path)
                label = 0
                if ann:
                    data = load_annotation_json(ann)
                    if data['annotations']:
                        label_name = data['annotations'][0]['label']
                        label = label_to_idx.get(label_name, 0)
                yield img, label
            else:
                ann = ann_paths.get(img_path)
                if ann:
                    data = load_annotation_json(ann)
                    if data['annotations']:
                        a = data['annotations'][0]
                        # normalized by original image size cannot be computed here without re-reading original image size.
                        x = a['x']
                        y = a['y']
                        w = a['w']
                        h = a['h']
                        # Normalize by final image size for training
                        bbox = [x / orig_w, y / orig_h, w / orig_w, h / orig_h]
                        label = label_to_idx.get(a['label'], 0)
                        yield img, (label, bbox)
                else:
                    # no annotation; skip
                    continue

    if classification:
        ds = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(*image_size, 3), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int64)))
    else:
        ds = tf.data.Dataset.from_generator(gen, output_signature=(tf.TensorSpec(shape=(*image_size, 3), dtype=tf.float32), (tf.TensorSpec(shape=(), dtype=tf.int64), tf.TensorSpec(shape=(4,), dtype=tf.float32))))
    return ds


if __name__ == '__main__':
    # Simple self-test: generate dataset (if missing) and load a sample
    import argparse
    from pathlib import Path
    import sys
    # When running the file directly, ensure parent of package is on sys.path so absolute imports work
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from vision_desktop_app.training.sample_data_generator import generate_sample
    except Exception as e:
        print('Failed to import project package modules:', e)
        print('Make sure you run the script from the project root or use `python -m vision_desktop_app.training.dataset_utils`')
        raise

    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='data/sample_dataset')
    p.add_argument('--count', type=int, default=10)
    p.add_argument('--mode', choices=['classification', 'detection'], default='detection')
    args = p.parse_args()

    ds_dir = Path(args.dataset)
    if not ds_dir.exists():
        print('Dataset not found — generating sample dataset at', ds_dir)
        generate_sample(str(ds_dir), n_images=args.count)

    print('Parsing dataset:', ds_dir, 'mode:', args.mode)
    ds = parse_dataset_from_folder(str(ds_dir), image_size=(128, 128), classification=(args.mode == 'classification'))
    ds = ds.batch(2)
    print('Printing 2 batches from dataset...')
    for i, batch in enumerate(ds.take(2)):
        print('Batch', i, '-> types:', type(batch), 'shapes:')
        try:
            # If detection: (img, (label, bbox))
            if args.mode == 'detection':
                imgs, (labels, bboxes) = batch
                print(' imgs:', imgs.shape, 'labels:', labels.shape, 'bboxes:', bboxes.shape)
            else:
                imgs, labels = batch
                print(' imgs:', imgs.shape, 'labels:', labels.shape)
        except Exception as e:
            print('Error printing batch:', e)
        if i >= 1:
            break
    print('Done.')
