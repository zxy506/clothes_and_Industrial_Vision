"""SQLite history for locally saved image-level predictions."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from src.services.inference_service import ROOT


DB_PATH = ROOT / "database" / "history.db"


def _connect(path=DB_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("""CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        detected_at TEXT NOT NULL,
        image_name TEXT NOT NULL,
        class_code TEXT NOT NULL,
        class_zh TEXT NOT NULL,
        confidence REAL NOT NULL,
        probabilities_json TEXT NOT NULL,
        model_name TEXT NOT NULL,
        model_version TEXT NOT NULL
    )""")
    return connection


def save_predictions(rows, *, mode: str, seed: int, version: str, path=DB_PATH,
                     model_family: str = "NEUCNN"):
    if model_family not in {"NEUCNN", "FashionCNN"}:
        raise ValueError("未知模型类型")
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    values = [(timestamp, row["image_name"], row["class_code"], row["class_zh"],
               row["confidence"], json.dumps(row["probabilities"]),
               f"{model_family}-{mode}-seed{seed}", version) for row in rows]
    with closing(_connect(path)) as connection:
        with connection:
            connection.executemany("""INSERT INTO detections
                (detected_at,image_name,class_code,class_zh,confidence,
                 probabilities_json,model_name,model_version)
                VALUES (?,?,?,?,?,?,?,?)""", values)
    return len(values)


def query_history(*, class_code: str | None = None, limit: int = 1000, path=DB_PATH,
                  model_family: str | None = None):
    if not 1 <= limit <= 10000:
        raise ValueError("limit 超出范围")
    if model_family not in {None, "NEUCNN", "FashionCNN"}:
        raise ValueError("未知模型类型")
    sql = "SELECT * FROM detections WHERE 1=1"
    params = []
    if class_code:
        sql += " AND class_code = ?"
        params.append(class_code)
    if model_family:
        sql += " AND model_name LIKE ?"
        params.append(model_family + "-%")
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    with closing(_connect(path)) as connection:
        return pd.read_sql_query(sql, connection, params=params)


def to_csv_bytes(frame: pd.DataFrame):
    return frame.to_csv(index=False).encode("utf-8-sig")
