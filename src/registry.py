"""
@File - registry.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 28/07/2026
"""

from dataclasses import dataclass, field
from typing import Optional
import joblib
import json


@dataclass
class ModelEntry:
    key: str
    display_name: str
    description: str
    status: str
    target_range: tuple = (0, 9)
    unit: str = ""
    pipeline_path: Optional[str] = None
    schema_path: Optional[str] = None
    _pipeline: object = field(default=None, repr=False)
    _schema: object = field(default=None, repr=False)

    def load(self):
        if self._pipeline is not None:
            return self._pipeline
        if not self.pipeline_path or not self.schema_path:
            raise ValueError(f"{self.key}: pipeline_path/schema_path not defined")
        self._pipeline = joblib.load(self.pipeline_path)

        with open(self.schema_path) as f:
            self._schema = json.load(f)

        return self._pipeline

    @property
    def numeric(self):
        if self._schema is None:
            self.load()
        return self._schema["features"]["numeric"]

    @property
    def categorical(self):
        if self._schema is None:
            self.load()
        return self._schema["features"]["categorical"]

    @property
    def feature_order(self):
        if self._schema is None:
            self.load()
        return self._schema["features"]["feature_cols_order"]


REGISTRY = {
    "oa": ModelEntry(
        key="oa",
        display_name="Overall Appearance Score",
        description="Predicts overall appearance (0–9) from trait measurements.",
        status="live",
        target_range=(0, 9),
        unit="/ 9",
        pipeline_path="model/oa/pipeline.joblib",
        schema_path="model/oa/model_schema.json",
    ),
    "yield": ModelEntry(
        key="yield",
        display_name="Yield Prediction",
        description="Predicts tuber yield (t/ha) across environments. In development.",
        status="coming_soon",
        unit="t/ha",
    ),
    "cooking": ModelEntry(
        key="cooking",
        display_name="Cooking Traits",
        description="Predicts cooking-quality traits. In development.",
        status="coming_soon",
    ),
}


def verify_schema_alignment(entry, input_model):

    if entry.status != "live":
        return
    schema_feats = set(entry.numeric) | set(entry.categorical)

    pyd_feats = set()
    for n, f in input_model.model_fields.items():
        pyd_feats.add(f.alias or n)
    missing = schema_feats - pyd_feats
    extra = pyd_feats - schema_feats
    if missing or extra:
        raise RuntimeError(
            f"{entry.key}: form/schema mismatch. "
            f"in schema not in form: {missing}; in form not in schema: {extra}"
        )
