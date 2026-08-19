"""
@File - app.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 28/07/2026
"""

"""Potato trait-prediction platform. Pydantic-driven forms, schema-driven features."""
from pathlib import Path
import io
import re
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from pydantic import ValidationError

from registry import REGISTRY, verify_schema_alignment
from schemas import INPUT_SCHEMAS, form_fields
import store
import explain

BASE = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(BASE / "templates"))

app = FastAPI(title="Teagasc Potato Prediction Platform")
app.add_middleware(SessionMiddleware, secret_key="change-me-in-prod")

DEMO_PASSWORD = "teagasc"
store.init_db()


def clean_column_name(name: str) -> str:
    c = str(name).strip().lower()
    c = re.sub(r"\s+", " ", c)

    c = c.replace(">=", "gte")
    c = c.replace("<=", "lte")
    c = c.replace(">", "gt")
    c = c.replace("<", "lt")
    c = c.replace("%", "percent")
    c = c.replace("#", "hash")

    c = re.sub(r"[^a-z0-9]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")

    if not c or not c[0].isalpha():
        c = f"col_{c}" if c else "col"

    return c


@app.on_event("startup")
def _startup():
    store.init_db()
    for key, entry in REGISTRY.items():
        if entry.status == "live" and key in INPUT_SCHEMAS:
            verify_schema_alignment(entry, INPUT_SCHEMAS[key])

            try:
                sample = pd.read_csv(f"model/{key}/sample.csv")
                explain.build_explainer(
                    key,
                    entry.load(),
                    entry.feature_order,
                    sample,
                    numeric_cols=entry.numeric,
                )
            except Exception as e:
                print(f"[warn] explainer for {key} not built: {e}")


# ---------- auth ----------
def require_login(request: Request):
    if not request.session.get("user"):
        raise HTTPException(
            status_code=status.HTTP_303_SEE_OTHER, headers={"Location": "/login"}
        )


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
async def login(request: Request):
    form = await request.form()
    if form.get("password") == DEMO_PASSWORD:
        request.session["user"] = form.get("username", "guest")
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(
        request, "login.html", {"error": "Wrong password"}, status_code=200
    )


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


# ---------- home ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request, _=Depends(require_login)):
    return templates.TemplateResponse(
        request,
        "home.html",
        {"models": list(REGISTRY.values()), "user": request.session.get("user")},
    )


# ---------- model page ----------
@app.get("/model/{model_key}", response_class=HTMLResponse)
def model_page(
    request: Request, model_key: str, tab: str = "predict", _=Depends(require_login)
):
    entry = REGISTRY.get(model_key)
    if not entry:
        raise HTTPException(404, "Unknown model")
    if entry.status == "coming_soon":
        return templates.TemplateResponse(request, "coming_soon.html", {"model": entry})
    ctx = {"model": entry, "tab": tab, "fields": form_fields(INPUT_SCHEMAS[model_key])}
    if tab == "performance":
        ctx["stats"] = store.monitoring(model_key)
        ctx["history"] = store.recent(model_key, limit=25)
    return templates.TemplateResponse(request, "model_page.html", ctx)


def _predict_frame(entry, data: dict):
    """Build the pipeline input frame in the schema's exact feature order."""
    order = entry.feature_order
    row = {}
    for c in order:
        v = data.get(c, None)
        if c in entry.numeric:
            row[c] = float(v) if v not in (None, "") else np.nan
        else:
            row[c] = v if v not in (None, "") else None
    return pd.DataFrame([row])[order]


# ---------- predict (single, HTMX fragment) ----------
@app.post("/model/{model_key}/predict", response_class=HTMLResponse)
async def predict(request: Request, model_key: str, _=Depends(require_login)):
    entry = REGISTRY.get(model_key)
    if not entry or entry.status != "live":
        raise HTTPException(404)
    form = await request.form()
    raw = {k: (v if v != "" else None) for k, v in form.items()}

    # validate through Pydantic -> clean field errors, no crashes
    try:
        model_in = INPUT_SCHEMAS[model_key](**raw)
    except ValidationError as e:
        errs = [f"{err['loc'][0]}: {err['msg']}" for err in e.errors()]
        return templates.TemplateResponse(
            request, "_result.html", {"model": entry, "errors": errs}
        )

    data = model_in.model_dump(by_alias=True)
    X = _predict_frame(entry, data)
    expl = None
    if explain.has_explainer(model_key):
        expl = explain.explain_instance(model_key, X, target_range=entry.target_range)
        pred = expl["prediction"]
    else:
        lo, hi = entry.target_range
        pred = float(np.clip(entry.load().predict(X)[0], lo, hi))

    n_missing = sum(1 for c in entry.numeric if pd.isna(X.iloc[0][c]))
    pred_id = store.log_prediction(model_key, data, pred, n_missing)

    return templates.TemplateResponse(
        request,
        "_result.html",
        {
            "model": entry,
            "score": round(pred, 2),
            "n_missing": n_missing,
            "pred_id": pred_id,
            "explain_text": expl["text"] if expl else None,
            "explain_svg": explain.svg_bars(expl["bars"]) if expl else None,
        },
    )


# ---------- predict (Excel batch) ----------
@app.post("/model/{model_key}/predict-excel")
async def predict_excel(
    model_key: str, file: UploadFile = File(...), _=Depends(require_login)
):
    entry = REGISTRY.get(model_key)
    if not entry or entry.status != "live":
        raise HTTPException(404)

    content = await file.read()
    try:
        df = pd.read_excel(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(400, f"Could not read Excel: {e}")

    clean_col_names = [clean_column_name(c) for c in df.columns]
    df.columns = clean_col_names

    needed = entry.numeric + entry.categorical
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            400, f"Uploaded file is missing required columns: {missing_cols}"
        )

    X = df.reindex(columns=entry.feature_order)
    for c in entry.numeric:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    lo, hi = entry.target_range
    preds = np.clip(entry.load().predict(X), lo, hi)

    df["prediction"] = np.round(preds, 2)
    n_missing = X[entry.numeric].isna().sum(axis=1)
    df["explainability"] = [
        f"coming soon ({len(entry.numeric)-m}/{len(entry.numeric)} traits given)"
        for m in n_missing
    ]

    out = io.BytesIO()
    df.to_excel(out, index=False)
    out.seek(0)
    fname = f"predictions_{model_key}.xlsx"
    return StreamingResponse(
        out,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ---------- feedback ----------
@app.post("/feedback/{pred_id}/{value}", response_class=HTMLResponse)
def feedback(request: Request, pred_id: str, value: str, _=Depends(require_login)):
    if value not in ("like", "dislike"):
        raise HTTPException(400)
    store.set_feedback(pred_id, value)
    return HTMLResponse(
        '<span class="text-success font-semibold">👍Thank you for your feedback </span>'
        if value == "like"
        else '<span class="text-success font-semibold">👎Thank you for your feedback. </span>'
    )
