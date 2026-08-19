"""
@File - store.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 28/07/2026
"""

import sqlite3, json, uuid, datetime

DB_PATH = "app.db"


def _con():
    return sqlite3.connect(DB_PATH)


def init_db():
    con = _con()
    con.execute("""CREATE TABLE IF NOT EXISTS predictions(
        id TEXT PRIMARY KEY, model_key TEXT, created_at TEXT,
        inputs TEXT, prediction REAL, n_missing INTEGER,
        feedback TEXT, actual REAL)""")
    con.commit()
    con.close()


def log_prediction(model_key, inputs, pred, n_missing):
    pid = str(uuid.uuid4())[:8]
    con = _con()
    con.execute(
        "INSERT INTO predictions VALUES (?,?,?,?,?,?,?,?)",
        (
            pid,
            model_key,
            datetime.datetime.now().isoformat(),
            json.dumps(inputs, default=str),
            pred,
            n_missing,
            None,
            None,
        ),
    )
    con.commit()
    con.close()
    return pid


def set_feedback(pred_id, feedback):
    con = _con()
    con.execute("UPDATE predictions SET feedback=? WHERE id=?", (feedback, pred_id))
    con.commit()
    con.close()


def recent(model_key=None, limit=50):
    con = _con()
    con.row_factory = sqlite3.Row
    if model_key:
        rows = con.execute(
            "SELECT * FROM predictions WHERE model_key=? ORDER BY created_at DESC LIMIT ?",
            (model_key, limit),
        ).fetchall()
    else:
        rows = con.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def monitoring(model_key):
    con = _con()
    total = con.execute(
        "SELECT count(*) FROM predictions WHERE model_key=?", (model_key,)
    ).fetchone()[0]
    likes = con.execute(
        "SELECT count(*) FROM predictions WHERE model_key=? AND feedback='like'",
        (model_key,),
    ).fetchone()[0]
    dislikes = con.execute(
        "SELECT count(*) FROM predictions WHERE model_key=? AND feedback='dislike'",
        (model_key,),
    ).fetchone()[0]
    avg_pred = con.execute(
        "SELECT avg(prediction) FROM predictions WHERE model_key=?", (model_key,)
    ).fetchone()[0]
    incomplete = con.execute(
        "SELECT count(*) FROM predictions WHERE model_key=? AND n_missing>0",
        (model_key,),
    ).fetchone()[0]
    con.close()
    return {
        "total": total,
        "likes": likes,
        "dislikes": dislikes,
        "avg_pred": round(avg_pred, 2) if avg_pred else None,
        "pct_incomplete": round(100 * incomplete / total, 1) if total else 0.0,
    }
