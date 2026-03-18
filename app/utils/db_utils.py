"""
Утилиты для чтения таблиц.

Поддерживаемые форматы: .csv .tsv .xlsx .xls .json .db .sqlite .sqlite3

read_table(path) → TableData
match_filenames(table, image_paths, ...) → MatchResult
"""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TableData:
    headers:       list[str]
    rows:          list[list[Any]]
    source_path:   str = ""
    source_format: str = ""

    def column(self, name: str) -> list[Any]:
        try:
            idx = self.headers.index(name)
            return [r[idx] if idx < len(r) else None for r in self.rows]
        except ValueError:
            return []

    def row_dict(self, i: int) -> dict:
        r = self.rows[i]
        return {h: (r[j] if j < len(r) else None)
                for j, h in enumerate(self.headers)}


@dataclass
class MatchResult:
    matched:               dict[int, dict] = field(default_factory=dict)
    unmatched_db:          dict[int, dict] = field(default_factory=dict)
    images_without_record: list[str]       = field(default_factory=list)


# ─── Считыватели ───────────────────────────────────────────────────────────────

def _encodings(path: str, fn):
    for enc in ("utf-8-sig", "utf-8", "cp1251", "cp1252", "latin1"):
        try:
            return fn(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return fn("latin1")


def _read_csv(path: str) -> TableData:
    def attempt(enc):
        with open(path, "r", encoding=enc, newline="") as fh:
            sample = fh.read(8192)
        counts = {d: sample.count(d) for d in (",", ";", "\t")}
        delim  = max(counts, key=counts.get)
        rows   = []
        with open(path, "r", encoding=enc, newline="") as fh:
            for r in csv.reader(fh, delimiter=delim):
                rows.append(r)
        return rows

    rows = _encodings(path, attempt)
    if not rows:
        return TableData([], [], path, "csv")
    headers = [h.strip() for h in rows[0]]
    data    = [[c.strip() for c in r] for r in rows[1:] if any(c.strip() for c in r)]
    return TableData(headers, data, path, "csv")


def _read_excel(path: str) -> TableData:
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        all_rows = [
            [str(c.value) if c.value is not None else "" for c in row]
            for row in ws.iter_rows()
        ]
        wb.close()
    except ImportError:
        import xlrd  # type: ignore
        book  = xlrd.open_workbook(path)
        sheet = book.sheet_by_index(0)
        all_rows = [
            [str(sheet.cell_value(r, c)) for c in range(sheet.ncols)]
            for r in range(sheet.nrows)
        ]
    if not all_rows:
        return TableData([], [], path, "excel")
    headers = [str(h).strip() for h in all_rows[0]]
    data    = [[str(c).strip() for c in r] for r in all_rows[1:]
               if any(str(c).strip() for c in r)]
    return TableData(headers, data, path, "excel")


def _read_json(path: str) -> TableData:
    def load(enc):
        with open(path, "r", encoding=enc, errors="strict") as fh:
            return json.load(fh)

    data = _encodings(path, load)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        headers = list(data[0].keys())
        rows    = [[str(r.get(h, "")) for h in headers] for r in data]
        return TableData(headers, rows, path, "json")
    if isinstance(data, dict):
        if "headers" in data and "rows" in data:
            return TableData(
                [str(h) for h in data["headers"]],
                [[str(c) for c in r] for r in data["rows"]],
                path, "json"
            )
        headers = ["key", "value"]
        rows    = [[str(k), str(v)] for k, v in data.items()]
        return TableData(headers, rows, path, "json")
    return TableData([], [], path, "json")


def _read_sqlite(path: str) -> TableData:
    conn = sqlite3.connect(path)
    cur  = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    if not tables:
        conn.close()
        return TableData([], [], path, "sqlite")
    cur.execute(f'SELECT * FROM "{tables[0]}"')  # noqa: S608
    headers = [d[0] for d in cur.description]
    rows    = [[str(c) if c is not None else "" for c in row]
               for row in cur.fetchall()]
    conn.close()
    return TableData(headers, rows, path, "sqlite")


# ─── Публичные функции ─────────────────────────────────────────────────────────

def read_table(path: str) -> TableData:
    ext = Path(path).suffix.lower()
    readers = {
        ".csv": _read_csv, ".tsv": _read_csv, ".txt": _read_csv,
        ".xlsx": _read_excel, ".xls": _read_excel,
        ".json": _read_json,
        ".db": _read_sqlite, ".sqlite": _read_sqlite, ".sqlite3": _read_sqlite,
    }
    return readers.get(ext, _read_csv)(path)


def _stem(s: str) -> str:
    return os.path.splitext(os.path.basename(str(s).strip()))[0].lower()


def match_filenames(
    table:        TableData,
    image_paths:  list[str],
    filename_col: str,
    mode:         str = "stem",   # stem | exact | contains
) -> MatchResult:
    result = MatchResult()

    try:
        col_idx = table.headers.index(filename_col)
    except ValueError:
        result.images_without_record = list(image_paths)
        return result

    # Построение индекса изображений
    img_index: dict[str, str] = {}
    for p in image_paths:
        key = _stem(p) if mode != "exact" else os.path.basename(p).lower()
        img_index[key] = p

    matched_paths: set[str] = set()

    for i, row in enumerate(table.rows):
        cell  = row[col_idx] if col_idx < len(row) else ""
        key   = _stem(cell) if mode != "exact" else cell.strip().lower()

        found = None
        if mode == "contains":
            for img_key, img_path in img_index.items():
                if key and key in img_key:
                    found = img_path
                    break
        else:
            found = img_index.get(key)

        if found:
            result.matched[i] = {"image_path": found, "row": table.row_dict(i)}
            matched_paths.add(found)
        else:
            result.unmatched_db[i] = {"row": table.row_dict(i)}

    result.images_without_record = [p for p in image_paths if p not in matched_paths]
    return result
