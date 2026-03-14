import os
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, Tuple

from vision_desktop_app.training.predictor import load_model, predict_image


def ensure_dir(path: Path):
    os.makedirs(path, exist_ok=True)


def get_label_from_prediction(prediction) -> Tuple[int, float]:
    """Return (class_idx, confidence) for classification or detection prediction
    prediction can be: 
     - classification: probs (1, num_classes)
     - detection: [class_probs, bbox]
    """
    if isinstance(prediction, (tuple, list)):
        class_probs = prediction[0]
        class_idx = int(class_probs.argmax(axis=1)[0])
        conf = float(class_probs.max(axis=1)[0])
        return class_idx, conf
    else:
        probs = prediction
        class_idx = int(probs.argmax(axis=1)[0])
        conf = float(probs.max(axis=1)[0])
        return class_idx, conf


def sort_images_by_model(source_dir: str, model_dir: str, move_files: bool = True, output_root: str = 'data/sorted', input_extensions: Optional[list] = None, model_loader=None) -> Dict[str, Dict]:
    """
    Process images in `source_dir`, run a model for prediction, and move/copy them to directories: output_root/<model_name>/<class_idx>/
    Params:
      - source_dir: folder containing images (and possibly .json annotations)
      - model_dir: path to saved model directory (folder or saved_model path)
      - move_files: if True => move; False => copy
      - output_root: base directory where sorted files go
      - input_extensions: list of allowed image extensions
      - model_loader: optional callable to load model, defaults to `load_model` from predictor

    Returns a dictionary with results per file: {filename: {class, confidence, dest}}
    """
    if input_extensions is None:
        input_extensions = ['.png', '.jpg', '.jpeg', '.bmp']

    model_name = os.path.basename(os.path.normpath(model_dir))
    results = {}

    # Load model
    if model_loader is None:
        model_loader = load_model
    model = model_loader(model_dir)
    if model is None:
        raise FileNotFoundError(f'Model not found at {model_dir}')

    # enumerate files
    p_source = Path(source_dir)
    if not p_source.exists():
        raise FileNotFoundError(f'Source folder not found: {source_dir}')

    files = [p for p in p_source.iterdir() if p.is_file() and p.suffix.lower() in input_extensions]
    for f in files:
        try:
            res = predict_image(model, str(f), input_size=(128, 128))
            label_idx, conf = get_label_from_prediction(res)
            dest_dir = Path(output_root) / model_name / str(label_idx)
            ensure_dir(dest_dir)
            dest_file = dest_dir / f.name
            if move_files:
                shutil.move(str(f), str(dest_file))
                # move corresponding json if exists
                json_src = f.with_suffix('.json')
                if json_src.exists():
                    shutil.move(str(json_src), str(dest_dir / json_src.name))
            else:
                shutil.copy2(str(f), str(dest_file))
                json_src = f.with_suffix('.json')
                if json_src.exists():
                    shutil.copy2(str(json_src), str(dest_dir / json_src.name))
            results[str(f)] = {'class': int(label_idx), 'confidence': float(conf), 'dest': str(dest_file)}
        except Exception as e:
            results[str(f)] = {'error': str(e)}
    # write a summary.json
    summary_file = Path(output_root) / model_name / 'summary.json'
    ensure_dir(summary_file.parent)
    with open(summary_file, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)

    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='incoming', help='Source directory with images to sort')
    p.add_argument('--model_dir', default='models/inside_outside/saved_model', help='Saved model folder')
    p.add_argument('--move', action='store_true', help='Move files instead of copy')
    p.add_argument('--out', default='data/sorted', help='Output root for sorted files')
    args = p.parse_args()

    print('Sorting images:', args.source, '->', args.out)
    res = sort_images_by_model(args.source, args.model_dir, move_files=args.move, output_root=args.out)
    print('Sorted', len([k for k in res.keys() if 'error' not in res[k]]), 'files. Summary written to', args.out)
