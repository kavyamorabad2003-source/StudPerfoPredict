"""Student Performance Prediction - Flask web app.

Run:  python app.py   then open http://127.0.0.1:5000
"""
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from sample_data import make_sample

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
REAL_CSV = DATA_DIR / "student-mat.csv"
UPLOADED_CSV = DATA_DIR / "uploaded.csv"
TARGET = "G3"
DEFAULTS = {"use_prev": False, "drop_zero": False, "test_size": 0.2}
# On a public deployment set READ_ONLY=1 so visitors cannot replace the shared dataset
READ_ONLY = os.environ.get("READ_ONLY", "0") == "1"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
STATE = {}


# ---------------------------------------------------------------- data
def read_csv(source):
    """Read a CSV that may be ';' separated (UCI) or ',' separated."""
    raw = source.read() if hasattr(source, "read") else Path(source).read_bytes()
    text = raw.decode("utf-8-sig", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep=";")
    if df.shape[1] == 1:
        df = pd.read_csv(io.StringIO(text), sep=",")
    df.columns = [str(c).strip().strip('"') for c in df.columns]
    return df


def validate(df):
    if TARGET not in df.columns:
        raise ValueError(f"CSV must contain a '{TARGET}' column (final grade, 0-20).")
    if len(df) < 30:
        raise ValueError("Need at least 30 rows to train.")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    return df.dropna(subset=[TARGET]).reset_index(drop=True)


def load_dataset():
    if UPLOADED_CSV.exists():
        return validate(read_csv(UPLOADED_CSV)), "Uploaded CSV"
    if REAL_CSV.exists():
        return validate(read_csv(REAL_CSV)), "UCI Student Performance (student-mat.csv)"
    return make_sample(), "Built-in SYNTHETIC sample (same columns as UCI) - add the real CSV for real results"


# ---------------------------------------------------------------- modelling
def make_pipeline(model, scale, num_cols, cat_cols):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    prep = ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(drop="if_binary", handle_unknown="ignore"))]), cat_cols),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def aggregate_by_feature(names, values, cat_cols):
    """Sum one-hot importances back to the original column."""
    out = {}
    cats = sorted(cat_cols, key=len, reverse=True)
    for n, v in zip(names, values):
        base = n.split("__", 1)[1]
        if n.startswith("cat__"):
            base = next((c for c in cats if base == c or base.startswith(c + "_")), base)
        out[base] = out.get(base, 0.0) + float(v)
    return out


def top(d, k=12):
    items = sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]
    return [{"name": n, "value": round(v, 4)} for n, v in items]


def train(settings):
    df = STATE["df"].copy()
    if settings["drop_zero"]:
        df = df[df[TARGET] > 0]
    drop = [TARGET] + ([] if settings["use_prev"] else [c for c in ("G1", "G2") if c in df.columns])
    X, y = df.drop(columns=drop), df[TARGET]
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    specs = {
        "Linear Regression": (LinearRegression(), True),
        "Decision Tree": (DecisionTreeRegressor(max_depth=5, min_samples_leaf=5, random_state=42), False),
        "Random Forest": (RandomForestRegressor(n_estimators=200, min_samples_leaf=2,
                                                random_state=42, n_jobs=-1), False),
    }
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=settings["test_size"], random_state=42)

    models, results, importance, preds = {}, [], {}, {}
    for name, (est, scale) in specs.items():
        pipe = make_pipeline(est, scale, num_cols, cat_cols)
        pipe.fit(X_tr, y_tr)
        p = np.clip(pipe.predict(X_te), 0, 20)
        cv = cross_val_score(make_pipeline(est.__class__(**est.get_params()), scale, num_cols, cat_cols),
                             X, y, cv=5, scoring="r2").mean()
        results.append({"model": name,
                        "mae": round(float(mean_absolute_error(y_te, p)), 3),
                        "rmse": round(float(np.sqrt(mean_squared_error(y_te, p))), 3),
                        "r2": round(float(r2_score(y_te, p)), 3),
                        "cv_r2": round(float(cv), 3)})
        # refit on all data for the Predict tab
        pipe.fit(X, y)
        models[name] = pipe
        preds[name] = p
        names = pipe.named_steps["prep"].get_feature_names_out()
        inner = pipe.named_steps["model"]
        if name == "Linear Regression":
            coefs = {n.split("__", 1)[1]: float(c) for n, c in zip(names, inner.coef_)}
            importance[name] = {"kind": "coefficient (per 1 std for numeric features)", "items": top(coefs)}
        else:
            agg = aggregate_by_feature(names, inner.feature_importances_, cat_cols)
            importance[name] = {"kind": "importance (sums to 1)", "items": top(agg)}

    best = max(results, key=lambda r: r["r2"])["model"]
    features = []
    for c in X.columns:
        if c in num_cols:
            s = X[c].dropna()
            is_int = bool((s % 1 == 0).all())
            features.append({"name": c, "type": "number", "min": float(s.min()), "max": float(s.max()),
                             "default": float(s.median()), "step": 1 if is_int else 0.1})
        else:
            s = X[c].dropna().astype(str)
            features.append({"name": c, "type": "select", "options": sorted(s.unique().tolist()),
                             "default": s.mode().iloc[0]})

    STATE.update(models=models, columns=X.columns.tolist(), num_cols=num_cols, settings=settings,
                 results=results, importance=importance, features=features,
                 scatter={"model": best, "actual": y_te.tolist(),
                          "pred": [round(float(v), 2) for v in preds[best]]})


def payload():
    df = STATE["df"]
    hist = np.bincount(df[TARGET].round().clip(0, 20).astype(int), minlength=21).tolist()
    return {
        "source": STATE["source"], "rows": int(len(df)), "cols": int(df.shape[1]),
        "settings": STATE["settings"], "results": STATE["results"],
        "importance": STATE["importance"], "scatter": STATE["scatter"],
        "features": STATE["features"], "hist": hist,
        "has_prev": "G1" in df.columns and "G2" in df.columns,
        "preview_cols": df.columns.tolist(),
        "preview": json.loads(df.head(8).to_json(orient="records")),
        "target_mean": round(float(df[TARGET].mean()), 2),
        "read_only": READ_ONLY,
    }


def init(settings=None):
    df, source = load_dataset()
    STATE.update(df=df, source=source)
    train(settings or dict(DEFAULTS))


# ---------------------------------------------------------------- routes
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/state")
def api_state():
    return jsonify(payload())


@app.post("/api/train")
def api_train():
    body = request.get_json(force=True) or {}
    settings = {
        "use_prev": bool(body.get("use_prev", False)) and "G1" in STATE["df"].columns,
        "drop_zero": bool(body.get("drop_zero", False)),
        "test_size": min(0.5, max(0.1, float(body.get("test_size", 0.2)))),
    }
    train(settings)
    return jsonify(payload())


@app.post("/api/upload")
def api_upload():
    if READ_ONLY:
        return jsonify(error="Uploads are disabled on this deployment."), 403
    f = request.files.get("file")
    if not f:
        return jsonify(error="No file received."), 400
    try:
        df = validate(read_csv(f.stream))
    except Exception as e:  # noqa: BLE001
        return jsonify(error=str(e)), 400
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(UPLOADED_CSV, index=False)
    init()
    return jsonify(payload())


@app.post("/api/reset")
def api_reset():
    if READ_ONLY:
        return jsonify(error="Reset is disabled on this deployment."), 403
    UPLOADED_CSV.unlink(missing_ok=True)
    init()
    return jsonify(payload())


@app.post("/api/predict")
def api_predict():
    body = request.get_json(force=True) or {}
    values = body.get("values", {})
    row = {}
    for c in STATE["columns"]:
        v = values.get(c)
        row[c] = (float(v) if v not in (None, "") else np.nan) if c in STATE["num_cols"] else v
    X = pd.DataFrame([row])[STATE["columns"]]
    out = {n: round(float(np.clip(m.predict(X)[0], 0, 20)), 2) for n, m in STATE["models"].items()}
    return jsonify(predictions=out)


init()

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  Student Performance Predictor -> http://{host}:{port}\n")
    app.run(host=host, port=port, debug=False)
