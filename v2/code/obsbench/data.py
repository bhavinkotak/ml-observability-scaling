"""Download (once) and cache the public OpenML datasets; record provenance in data/SOURCES.md."""

from __future__ import annotations

import datetime as dt
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from .config import DATA_RAW

DATASETS = {
    "electricity": {
        "data_id": 151,
        "numeric": ["nswprice", "nswdemand", "vicprice", "vicdemand", "transfer", "period"],
        "categorical": ["day"],
    },
    "adult": {
        "data_id": 1590,
        "numeric": ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"],
        "categorical": ["workclass", "occupation", "relationship", "race"],
    },
    "bank-marketing": {
        "data_id": 1461,
        "numeric": ["V1", "V6", "V12", "V13", "V14", "V15"],
        "categorical": ["V2", "V3", "V11", "V16"],
        "note": "UCI column names: V1 age, V6 balance, V12 duration, V13 campaign, V14 pdays, V15 previous; "
                "V2 job, V3 marital, V11 month, V16 poutcome",
    },
}


def load(name: str) -> pd.DataFrame:
    spec = DATASETS[name]
    bunch = fetch_openml(data_id=spec["data_id"], as_frame=True, data_home=str(DATA_RAW), parser="liac-arff")
    return bunch.frame


def load_details(name: str) -> dict:
    bunch = fetch_openml(data_id=DATASETS[name]["data_id"], as_frame=True, data_home=str(DATA_RAW),
                         parser="liac-arff")
    return dict(bunch.details)


def feature_arrays(name: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Numeric features as float arrays (NaN dropped); categoricals as int codes with missing as its own code."""
    df = load(name)
    spec = DATASETS[name]
    numeric = {}
    for col in spec["numeric"]:
        x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        numeric[col] = x[~np.isnan(x)]
    categorical = {}
    for col in spec["categorical"]:
        s = df[col].astype("object").where(df[col].notna(), "__missing__").astype(str)
        categorical[col] = pd.factorize(s, sort=True)[0].astype(np.int64)
    return numeric, categorical


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sources() -> None:
    """Fetch all datasets (cached after the first call) and write data/SOURCES.md."""
    lines = ["# Data sources", "",
             "All datasets are fetched from OpenML with `sklearn.datasets.fetch_openml` and cached under "
             "`data/raw/openml/`. Retrieval date is the date `run_all.py` first populated the cache.", ""]
    stamp = DATA_RAW / "retrieved.json"
    retrieved = json.loads(stamp.read_text()) if stamp.exists() else {}
    for name, spec in DATASETS.items():
        det = load_details(name)
        retrieved.setdefault(name, dt.date.today().isoformat())
        files = sorted(p for p in (DATA_RAW / "openml").rglob("*")
                       if p.is_file() and (p.name == f"{spec['data_id']}.gz"
                                           or f"/download/{det.get('file_id')}/" in str(p)))
        lines += [f"## {name} (OpenML data_id {spec['data_id']}, version {det.get('version')})", "",
                  f"- URL: https://www.openml.org/d/{spec['data_id']}",
                  f"- Original name: {det.get('name')}",
                  f"- License (OpenML `licence` field): {det.get('licence', 'not stated')}",
                  f"- Retrieved: {retrieved[name]}",
                  f"- Features used (numeric): {', '.join(spec['numeric'])}",
                  f"- Features used (categorical): {', '.join(spec['categorical'])}",
                  *([f"- Column meaning: {spec['note']}"] if "note" in spec else []),
                  "- Cached files (sha256):"]
        lines += [f"  - `{p.relative_to(DATA_RAW.parent)}`: {_sha256(p)}" for p in files]
        lines.append("")
    stamp.write_text(json.dumps(retrieved, indent=1))
    (DATA_RAW.parent / "SOURCES.md").write_text("\n".join(lines))
