"""
@File - explain.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 03/08/2026
"""

import numpy as np
import pandas as pd

_EXPLAINERS = {}

NICE_NAME = {
    "appearance": "appearance rating",
    "uniformity": "uniformity",
    "eveness": "evenness",
    "tubersize": "tuber size",
    "tubnumbers": "tuber count",
    "eyedepth": "eye depth",
    "yield_": "yield",
    "yield": "yield",
    "ffscab": "scab level",
    "ffdefects": "defect level",
    "ffhollowh": "hollow heart",
    "ff_irs": "internal rust spot",
    "env_type": "environment",
    "soil_type": "soil type",
    "trial_type": "trial type",
}


def _nice(f):
    return NICE_NAME.get(f, f.replace("_", " "))


def build_explainer(
    model_key, pipeline, feature_order, sample_df, *, numeric_cols, n_bg=30
):
    import shap

    feats = feature_order
    bg = (
        sample_df[feats]
        .sample(min(n_bg, len(sample_df)), random_state=0)
        .reset_index(drop=True)
    )

    def f(data):
        d = pd.DataFrame(data, columns=feats)
        for c in numeric_cols:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        return pipeline.predict(d)

    expl = shap.KernelExplainer(f, bg)
    _EXPLAINERS[model_key] = (expl, feats, float(expl.expected_value), numeric_cols)


def has_explainer(model_key):
    return model_key in _EXPLAINERS


def explain_instance(model_key, row_df, *, target_range=(0, 9), nsamples=100, top_k=4):
    expl, feats, base, numeric = _EXPLAINERS[model_key]
    sv = np.array(
        expl.shap_values(row_df[feats], nsamples=nsamples, silent=True)
    ).ravel()

    contribs = sorted(
        zip(feats, sv, [row_df.iloc[0][f] for f in feats]), key=lambda t: -abs(t[1])
    )
    pred = float(np.clip(base + sv.sum(), *target_range))

    top = [c for c in contribs if abs(c[1]) > 0.01][:top_k]
    ups = [c for c in top if c[1] > 0]
    downs = [c for c in top if c[1] < 0]

    def phrase(items, verb):
        if not items:
            return ""
        parts = [f"{_nice(f)} ({'+' if v > 0 else ''}{v:.2f})" for f, v, _ in items]
        joined = (
            parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + f" and {parts[-1]}"
        )
        return f"{joined} {verb} the score"

    body = "; ".join(t for t in [phrase(ups, "raised"), phrase(downs, "lowered")] if t)
    if not body:
        body = (
            "no single trait strongly moved this prediction; "
            "the score sits near the dataset average"
        )

    text = (
        f"The model predicted {pred:.2f} (dataset average is {base:.2f}). "
        f"For this variety, {body}. These are the traits that most "
        f"influenced this particular prediction."
    )

    bars = [{"feature": _nice(f), "value": float(v)} for f, v, _ in top]
    return {
        "prediction": round(pred, 2),
        "base": round(base, 2),
        "text": text,
        "bars": bars,
    }


def svg_bars(bars, width=380, row_h=34):
    if not bars:
        return ""
    maxabs = max(abs(b["value"]) for b in bars) or 1.0
    mid = width // 2
    h = row_h * len(bars) + 10
    rows = []
    for i, b in enumerate(bars):
        y = 8 + i * row_h
        w = int(abs(b["value"]) / maxabs * (mid - 70))
        pos = b["value"] > 0
        x = mid if pos else mid - w
        colour = "#16a34a" if pos else "#dc2626"
        lx = mid + (w + 4 if pos else -w - 4)
        rows.append(
            f'<text x="4" y="{y+16}" font-size="11" fill="currentColor">{b["feature"]}</text>'
            f'<rect x="{x}" y="{y+4}" width="{w}" height="16" rx="3" fill="{colour}"/>'
            f'<text x="{lx}" y="{y+16}" font-size="10" fill="currentColor" '
            f'text-anchor="{"start" if pos else "end"}">{b["value"]:+.2f}</text>'
        )
    axis = f'<line x1="{mid}" y1="0" x2="{mid}" y2="{h}" stroke="#ccc"/>'
    return (
        f'<svg viewBox="0 0 {width} {h}" width="100%" style="max-width:{width}px">'
        f'{axis}{"".join(rows)}</svg>'
    )
