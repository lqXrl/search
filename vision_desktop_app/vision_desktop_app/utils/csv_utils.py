import csv
from typing import List, Tuple, Optional

def read_csv_autoencoding(path: str, delimiter: str = ',') -> Tuple[List[List[str]], Optional[str]]:
    """Read CSV trying several encodings. Returns (rows, detected_encoding).

    Tries: utf-8-sig, utf-8, cp1251, latin1. Falls back to latin1.
    """
    encodings = ['utf-8-sig', 'utf-8', 'cp1251', 'latin1']
    last_exc = None
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc, errors='strict') as fh:
                reader = csv.reader(fh, delimiter=delimiter)
                rows = [row for row in reader]
            return rows, enc
        except Exception as e:
            last_exc = e
            continue
    # as a last resort try latin1 with replacement
    try:
        with open(path, 'r', encoding='latin1', errors='replace') as fh:
            reader = csv.reader(fh, delimiter=delimiter)
            rows = [row for row in reader]
        return rows, 'latin1-replace'
    except Exception:
        raise last_exc if last_exc is not None else RuntimeError('Failed to read CSV')


def csv_first_column_as_paths(path: str, delimiter: str = ',') -> Tuple[List[str], Optional[str]]:
    """Read CSV and return first column values (useful for lists of filenames)."""
    rows, enc = read_csv_autoencoding(path, delimiter=delimiter)
    out = []
    for r in rows:
        if not r:
            continue
        out.append(r[0].strip())
    return out, enc


def read_annotations_csv(path: str, delimiter: str = ',') -> Tuple[List[dict], Optional[str]]:
    """Read CSV that contains annotation rows and return list of dicts:
    Expected columns (any of these header names accepted):
      filename, file, image, path  -> image path
      label, class                 -> annotation label
      x, xmin, left                -> x coordinate (int)
      y, ymin, top                 -> y coordinate (int)
      w, width                     -> width (int)
      h, height                    -> height (int)

    If no header is present, function will try to parse rows as
    [filename, label, x, y, w, h]. Missing numeric columns will be
    left as None (caller can decide how to handle).
    Returns (rows_list, detected_encoding)
    """
    rows, enc = read_csv_autoencoding(path, delimiter=delimiter)
    if not rows:
        return [], enc

    # detect header
    header = None
    first = rows[0]
    lower = [c.strip().lower() for c in first]
    has_header = any(h in lower for h in ('filename', 'file', 'image', 'path', 'label', 'class'))
    start_idx = 1 if has_header else 0
    if has_header:
        header = lower

    out = []
    for r in rows[start_idx:]:
        if not r or all((not c.strip()) for c in r):
            continue
        data = {'source_row': r}
        if header:
            # map by header
            mapping = {h: i for i, h in enumerate(header)}
            # filename
            for key in ('filename', 'file', 'image', 'path'):
                if key in mapping:
                    data['filename'] = r[mapping[key]].strip()
                    break
            # label
            for key in ('label', 'class'):
                if key in mapping:
                    data['label'] = r[mapping[key]].strip()
                    break
            # coords
            def get_int(keys):
                for k in keys:
                    if k in mapping:
                        try:
                            return int(float(r[mapping[k]].strip()))
                        except Exception:
                            return None
                return None
            data['x'] = get_int(('x', 'xmin', 'left'))
            data['y'] = get_int(('y', 'ymin', 'top'))
            data['w'] = get_int(('w', 'width'))
            data['h'] = get_int(('h', 'height'))
        else:
            # assume order filename,label,x,y,w,h
            data['filename'] = r[0].strip() if len(r) > 0 else ''
            data['label'] = r[1].strip() if len(r) > 1 else ''
            def parse_i(i):
                try:
                    return int(float(r[i].strip()))
                except Exception:
                    return None
            data['x'] = parse_i(2) if len(r) > 2 else None
            data['y'] = parse_i(3) if len(r) > 3 else None
            data['w'] = parse_i(4) if len(r) > 4 else None
            data['h'] = parse_i(5) if len(r) > 5 else None

        out.append(data)

    return out, enc
