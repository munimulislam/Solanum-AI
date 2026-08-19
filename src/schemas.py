"""
@File - schemas.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 29/07/2026
"""

from pydantic import BaseModel, Field
from typing import Optional, Literal

CHOICE_LABELS = {
    "env_type": {
        "MED": "Mediterranean",
        "NE": "North Europe",
    },
    "trial_type": {"PROCESS": "Process", "WARE": "Ware", "UNKNOWN": "Please Select"},
    "soil_type": {"CLAY": "Clay", "SAND": "Sand", "UNKNOWN": "Please Select"},
}


class OAScoreInput(BaseModel):

    model_config = {"populate_by_name": True}

    env_type: Literal["MED", "NE"] = Field(..., description="Environment")
    trial_type: Optional[Literal["UNKNOWN", "PROCESS", "WARE"]] = Field(
        "UNKNOWN", description="Trial type"
    )
    soil_type: Optional[Literal["UNKNOWN", "CLAY", "SAND"]] = Field(
        "UNKNOWN", description="Soil type"
    )

    tubersize: Optional[float] = Field(None, ge=0, description="Tuber size")
    eveness: Optional[float] = Field(None, ge=0, le=9, description="Evenness (0–9)")
    appearance: Optional[float] = Field(
        None, ge=0, le=9, description="Appearance (0–9)"
    )
    tubnumbers: Optional[float] = Field(None, ge=0, description="Tuber count")
    eyedepth: Optional[float] = Field(None, ge=0, le=9, description="Eye depth (0–9)")

    uniformity: Optional[float] = Field(
        None, ge=0, le=9, description="Uniformity (0–9)"
    )
    yield_: Optional[float] = Field(
        None, ge=0, alias="yield", description="Yield (t/ha)"
    )
    ffscab: Optional[float] = Field(None, ge=0, le=9, description="Scab (0–9)")
    ffdefects: Optional[float] = Field(None, ge=0, le=9, description="Defects (0–9)")
    ffhollowh: Optional[float] = Field(
        None, ge=0, le=9, description="Hollow heart (0–9)"
    )
    ff_irs: Optional[float] = Field(
        None, ge=0, le=9, description="Internal rust spot (0–9)"
    )


INPUT_SCHEMAS = {
    "oa": OAScoreInput,
    # "yield": YieldInput,
    # "cooking": CookingInput,
}


def form_fields(model_cls) -> list[dict]:
    sch = model_cls.model_json_schema()
    required = set(sch.get("required", []))
    fields = []
    for name, spec in sch["properties"].items():
        choices, lo, hi = None, None, None
        for src in [spec, *spec.get("anyOf", [])]:
            if "enum" in src:
                choices = src["enum"]
            if "minimum" in src:
                lo = src["minimum"]
            if "maximum" in src:
                hi = src["maximum"]
                choice_pairs = None
        if choices:
            lm = CHOICE_LABELS.get(name, {})
            choice_pairs = [(c, lm.get(c, c)) for c in choices]
        fields.append(
            {
                "name": name,
                "kind": "select" if choices else "number",
                "choices": choice_pairs,
                "min": lo,
                "max": hi,
                "required": name in required,
                "desc": spec.get("description", ""),
            }
        )
    return fields
