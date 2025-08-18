# app.py
import os, re, difflib
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib

# ────────────────────────────────────────────────────────────────────────────────
# Paths
# ────────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
PRICE_MIN_YEAR = 2016  # only use recent price years for history/forecast

def find_file(filename, subfolder=None):
    if subfolder:
        p = BASE_DIR / subfolder / filename
        if p.exists(): return p
    p = BASE_DIR / filename
    if p.exists(): return p
    for folder in ("models", "data"):
        if subfolder and folder == subfolder:
            continue
        p = BASE_DIR / folder / filename
        if p.exists(): return p
    return None

MODEL_PATH = (find_file("crop_yield.joblib", "models") or
              find_file("crop_yield.pkl",   "models") or
              find_file("crop_yield.joblib")          or
              find_file("crop_yield.pkl")             or
              find_file("crop_model.pkl"))

PRICE_PATH = find_file("market_price_cleaned.csv", "data") or find_file("market_price_cleaned.csv")
APY_PATH   = find_file("apy_.csv",                 "data") or find_file("apy_.csv")

# ────────────────────────────────────────────────────────────────────────────────
# Remote dataset URLs (kept out of code): read from Streamlit secrets or env vars
# Put these in .streamlit/secrets.toml (see note at bottom)
# ────────────────────────────────────────────────────────────────────────────────
def _drive_direct_url(url):
    """Convert a Google Drive share link to a direct download URL; returns None if url falsy."""
    if not url:
        return None
    m = re.search(r"/d/([a-zA-Z0-9_-]+)", url) or re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    return url

def _get_secret(name):
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name)

DATA_URL_PRICE = _drive_direct_url(_get_secret("DATA_URL_PRICE"))
DATA_URL_APY   = _drive_direct_url(_get_secret("DATA_URL_APY"))

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Crop Price & Revenue Forecaster (India prototype)",
                   page_icon="🌾", layout="wide")
st.markdown("<h1 style='margin-bottom:0'>Crop Price & Revenue Forecaster</h1>", unsafe_allow_html=True)
st.caption("India prototype · Holt (damped) prices with fallback + backend XGBoost yield + APY trend blend → revenue · market→district→state fallbacks")
st.divider()

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def _tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(s).lower()))

def _extract_year(series: pd.Series) -> pd.Series:
    y = series.astype(str).str.extract(r"(\d{4})", expand=False)
    return pd.to_numeric(y, errors="coerce").astype("Int64")

def _to_numeric_clean(series: pd.Series) -> pd.Series:
    return (series.astype(str)
                  .str.replace(",", "", regex=False)
                  .str.replace("₹", "", regex=False)
                  .str.strip()
                  .replace({"": None})
                  .pipe(pd.to_numeric, errors="coerce"))

def _fix_names(s: pd.Series) -> pd.Series:
    alias = {"Bellary": "Ballari", "Gadag ": "Gadag"}
    return (s.astype(str).str.strip()
                     .str.replace(r"\s+", " ", regex=True)
                     .str.title()
                     .replace(alias))

# ────────────────────────────────────────────────────────────────────────────────
# Loaders
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_price() -> pd.DataFrame:
    df = None
    # 1) Remote first
    if DATA_URL_PRICE:
        try:
            df = pd.read_csv(DATA_URL_PRICE)
            st.caption("Price source: remote file")
        except Exception as e:
            st.warning(f"Could not load price data from remote; falling back to local. ({type(e).__name__})")
    # 2) Local fallback
    if df is None:
        if not PRICE_PATH or not PRICE_PATH.exists():
            st.error("Price CSV not found (remote URL not set/failed, and local file missing).")
            return pd.DataFrame()
        df = pd.read_csv(PRICE_PATH)
        st.caption("Price source: local file")

    # — cleaning —
    df.columns = df.columns.str.strip()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    rename_map = {
        "state": "State",
        "district": "District",
        "marketname": "Market Name",
        "market name": "Market Name",
        "commodityname": "Commodity Name",
        "commodity name": "Commodity Name",
    }
    n2c = {_norm(c): c for c in df.columns}
    for k, v in rename_map.items():
        if k in n2c: df.rename(columns={n2c[k]: v}, inplace=True)

    if "Year" not in df.columns:
        for c in df.columns:
            if "year" in _norm(c):
                df.rename(columns={c: "Year"}, inplace=True); break
    if "Year" in df.columns:
        df["Year"] = _extract_year(df["Year"])

    for col, toks in {"Min_Price": ["min","price"], "Max_Price": ["max","price"], "Modal_Price": ["modal","price"]}.items():
        if col not in df.columns:
            for c in df.columns:
                if all(t in _norm(c) for t in toks):
                    df.rename(columns={c: col}, inplace=True); break

    for c in ["Min_Price","Max_Price","Modal_Price"]:
        if c in df.columns: df[c] = _to_numeric_clean(df[c])

    before = len(df)
    df = df.dropna(subset=[c for c in ["Year","Commodity Name"] if c in df.columns])
    if "Modal_Price" in df.columns:
        df = df[(df["Modal_Price"] >= 0) & (df["Modal_Price"] <= 1_000_000)]

    if "State" in df.columns: df["State"] = _fix_names(df["State"])
    if "District" in df.columns: df["District"] = _fix_names(df["District"])

    df = df[df["Year"].astype("Int64") >= PRICE_MIN_YEAR]
    st.caption(f"Price rows kept: {len(df):,} (dropped {before-len(df):,})")
    return df

@st.cache_data(show_spinner=False)
def load_apy() -> pd.DataFrame:
    df = None
    # 1) Remote first
    if DATA_URL_APY:
        try:
            df = pd.read_csv(DATA_URL_APY)
            st.caption("APY source: remote file")
        except Exception as e:
            st.warning(f"Could not load APY data from remote; falling back to local. ({type(e).__name__})")
    # 2) Local fallback
    if df is None:
        if not APY_PATH or not APY_PATH.exists():
            st.warning("APY CSV not found (remote URL not set/failed, and local file missing) — continuing without APY.")
            return pd.DataFrame()
        df = pd.read_csv(APY_PATH)
        st.caption("APY source: local file")

    # — cleaning —
    df.columns = df.columns.str.strip()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    ren = {"state":"State","district":"District","crop":"Crop",
           "cropyear":"Crop_Year","crop_year":"Crop_Year","crop year":"Crop_Year","year":"Crop_Year",
           "yield":"Yield","season":"Season","area":"Area","production":"Production"}
    n2c = {_norm(c): c for c in df.columns}
    for k, v in ren.items():
        if k in n2c: df.rename(columns={n2c[k]: v}, inplace=True)

    if "Crop_Year" in df.columns: df["Crop_Year"] = _extract_year(df["Crop_Year"])
    for c in ["Yield","Area","Production"]:
        if c in df.columns: df[c] = _to_numeric_clean(df[c])

    need = [c for c in ["State","District","Crop","Crop_Year"] if c in df.columns]
    if need: df = df.dropna(subset=need)

    if "State" in df.columns: df["State"] = _fix_names(df["State"])
    if "District" in df.columns: df["District"] = _fix_names(df["District"])

    st.caption(f"APY rows kept: {len(df):,}")
    return df

# ────────────────────────────────────────────────────────────────────────────────
# Model + wrapping
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH or not MODEL_PATH.exists():
        st.error(f"Yield model file not found at: {MODEL_PATH}")
        return None
    return joblib.load(MODEL_PATH)

@st.cache_resource(show_spinner=False)
def wrap_bare_xgb_with_encoder_if_needed(_model, _apy_df: pd.DataFrame):
    if isinstance(_model, Pipeline):
        return _model, "pipeline"
    try:
        from xgboost.sklearn import XGBRegressor as _XGB
        if isinstance(_model, _XGB):
            try:
                if bool(_model.get_xgb_params().get("enable_categorical", False)):
                    return _model, "xgb_categorical"
            except Exception:
                pass
    except Exception:
        pass
    if _apy_df is None or _apy_df.empty:
        raise RuntimeError("Loaded model requires preprocessing but APY data is missing to fit encoder.")

    cat_cols = [c for c in ["State","District","Crop","Season"] if c in _apy_df.columns]
    num_cols = [c for c in ["Crop_Year","Area","Production"] if c in _apy_df.columns]
    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
         ("num", "passthrough", num_cols)],
        remainder="drop",
        sparse_threshold=1.0
    )
    pre.fit(_apy_df[[*cat_cols, *num_cols]].copy())
    return Pipeline([("pre", pre), ("reg", _model)]), "wrapped"

# ────────────────────────────────────────────────────────────────────────────────
# Crop mapper
# ────────────────────────────────────────────────────────────────────────────────
def _norm_crop(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def best_match(name: str, candidates: list[str], floor: float = 0.82) -> str:
    n = _norm_crop(name)
    best, score = None, 0.0
    for cand in candidates:
        r = difflib.SequenceMatcher(None, n, _norm_crop(cand)).ratio()
        if r > score:
            best, score = cand, r
    return best if score >= floor else name

@st.cache_data(show_spinner=False)
def build_crop_map(apy_df: pd.DataFrame, price_crops: list[str]) -> dict:
    manual = {
        "alsandegram": "Alsandikai",
        "alsandigram": "Alsandikai",
        "arhardal": "Arhar Dal(Tur Dal)",
        "arhar(tur/redgram)(whole)": "Arhar (Tur/Red Gram)(Whole)",
        "bajra(pearlmillet/cumbu)": "Bajra(Pearl Millet/Cumbu)",
        "blackgram(urdbeans)(whole)": "Black Gram (Urd Beans)(Whole)",
        "greenchilli": "Green Chilli",
        "jowar(sorghum)": "Jowar(Sorghum)",
    }
    if apy_df is None or apy_df.empty or "Crop" not in apy_df.columns:
        return {c: c for c in price_crops}

    apy_crops = sorted(apy_df["Crop"].dropna().astype(str).unique().tolist())
    apy_norm = {_norm_crop(x): x for x in apy_crops}

    mapping = {}
    for raw in price_crops:
        key = _norm_crop(raw)
        if key in manual:
            mapping[raw] = manual[key]
        elif key in apy_norm:
            mapping[raw] = apy_norm[key]
        else:
            ptoks = _tokens(raw)
            best_c, score = None, 0.0
            for c in apy_crops:
                atoks = _tokens(c)
                j = len(ptoks & atoks) / max(1, len(ptoks | atoks))
                if j > score:
                    score, best_c = j, c
            mapping[raw] = best_c if score >= 0.5 else best_match(raw, apy_crops, floor=0.8)
    return mapping

# ────────────────────────────────────────────────────────────────────────────────
# Defaults for Season/Area/Production
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_defaults(apy_df: pd.DataFrame):
    if apy_df is None or apy_df.empty:
        return pd.DataFrame(columns=["State","District","Crop","Season","Area","Production"])
    def _mode(s):
        s = s.dropna()
        return s.mode().iloc[0] if not s.empty else np.nan
    cols = {}
    if "Season" in apy_df.columns: cols["Season"] = _mode
    if "Area" in apy_df.columns:   cols["Area"]   = "median"
    if "Production" in apy_df.columns: cols["Production"] = "median"
    return (apy_df.groupby(["State","District","Crop"], dropna=False).agg(cols).reset_index())

def get_defaults(defaults_df, state, district, crop):
    for cond in [
        ((defaults_df["State"]==state)&(defaults_df["District"]==district)&(defaults_df["Crop"]==crop)),
        ((defaults_df["State"]==state)&(defaults_df["Crop"]==crop)),
        (defaults_df["Crop"]==crop),
        (pd.Series([True]*len(defaults_df))),
    ]:
        hit = defaults_df[cond]
        if not hit.empty:
            r = hit.iloc[0]
            season = r.get("Season", "Kharif")
            area   = r.get("Area",   np.nan)
            prod   = r.get("Production", np.nan)
            return (season if pd.notna(season) else "Kharif",
                    float(area) if pd.notna(area) else np.nan,
                    float(prod) if pd.notna(prod) else np.nan)
    return "Kharif", np.nan, np.nan

# ────────────────────────────────────────────────────────────────────────────────
# Price forecasting
# ────────────────────────────────────────────────────────────────────────────────
def choose_sarima(ts: pd.Series):
    best_aic, best_order = np.inf, (1,1,1)
    for p in (0,1,2):
        for q in (0,1,2):
            try:
                res = SARIMAX(ts, order=(p,1,q), trend="c", enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                if res.aic < best_aic: best_aic, best_order = res.aic, (p,1,q)
            except Exception:
                continue
    return best_order

def holt_forecast(ts: pd.Series, steps: int) -> np.ndarray:
    model = ExponentialSmoothing(ts, trend="add", damped_trend=True, seasonal=None)
    res = model.fit(optimized=True)
    fc = res.forecast(steps)
    return np.asarray(fc, dtype=float)

def arima_drift_forecast(ts_log1p: pd.Series, order, steps: int) -> np.ndarray:
    res = SARIMAX(ts_log1p, order=order, trend="c",
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = res.get_forecast(steps=steps).predicted_mean.values
    return np.expm1(fc)

def cap_forecast(fc: np.ndarray, hist: np.ndarray) -> np.ndarray:
    hist = np.asarray(hist, dtype=float)
    hist = hist[np.isfinite(hist)]
    if hist.size == 0:
        return np.maximum(fc, 0.0)
    q95 = np.quantile(hist, 0.95)
    cap = max(q95 * 1.8, np.median(hist) * 2.5)
    return np.clip(fc, 0.0, cap)

def log_arima_or_holt(hist_yearly: pd.DataFrame, horizon: int) -> pd.DataFrame:
    ydf = (hist_yearly.dropna()
           .groupby("Year", as_index=False)["Modal_Price"].mean()
           .sort_values("Year"))
    ydf = ydf[ydf["Year"] >= PRICE_MIN_YEAR]
    if len(ydf) < 3:
        return pd.DataFrame(columns=["Year", "Forecast_Price"])

    ts = ydf.set_index("Year")["Modal_Price"].astype(float)

    # forecasts at 2026
    steps_display = int(horizon)
    last_year = int(ts.index.max())
    start_year = max(2026, last_year + 1)         # required anchor
    lead_in_gap = max(0, start_year - (last_year + 1))
    steps_total = steps_display + lead_in_gap

    # Try Holt
    try:
        fc_total = holt_forecast(ts, steps_total)
    except Exception:
        fc_total = None

    # ARIMA with drift on log1p
    if fc_total is None or not np.isfinite(fc_total).all():
        ts_log = np.log1p(ts.clip(lower=1))
        order = choose_sarima(ts_log)
        try:
            fc_total = arima_drift_forecast(ts_log, order, steps_total)
        except Exception:
            fc_total = np.full(steps_total, float(ts.iloc[-1]))

    # Cap extremes
    fc_total = cap_forecast(np.asarray(fc_total, dtype=float), ts.values)
    start_idx = lead_in_gap
    fc = fc_total[start_idx:start_idx + steps_display]
    fut_years = np.arange(start_year, start_year + steps_display, dtype=int)

    return pd.DataFrame({"Year": fut_years, "Forecast_Price": fc})

def yearly_price_with_fallback(df: pd.DataFrame, state, district, market, crop):
    paths = [
        (df[(df["State"]==state)&(df["District"]==district)&(df["Market Name"]==market)&(df["Commodity Name"]==crop)], "market"),
        (df[(df["State"]==state)&(df["District"]==district)&(df["Commodity Name"]==crop)], "district"),
        (df[(df["State"]==state)&(df["Commodity Name"]==crop)], "state"),
    ]
    for sub, lvl in paths:
        sub = sub[["Year","Modal_Price"]].dropna()
        sub = sub[sub["Year"] >= PRICE_MIN_YEAR]
        if sub["Year"].nunique() >= 3:
            return sub, lvl
    return pd.DataFrame(columns=["Year","Modal_Price"]), "none"

# ────────────────────────────────────────────────────────────────────────────────
# Yield prediction
# ────────────────────────────────────────────────────────────────────────────────
def yield_drift_from_history(apy_df: pd.DataFrame, state: str, district: str, crop: str):
    """
    Return (g_per_year, source_tag) where g_per_year is an annual multiplicative growth rate
    estimated from APY history. Fallbacks:
      district+crop → state+crop → district all crops → state all crops.
    """
    if apy_df is None or apy_df.empty or not {"Crop_Year","Yield","State","District","Crop"}.issubset(apy_df.columns):
        return 0.0, "none"

    candidates = [
        ((apy_df["State"]==state)&(apy_df["District"]==district)&(apy_df["Crop"]==crop), "district-crop"),
        ((apy_df["State"]==state)&(apy_df["Crop"]==crop), "state-crop"),
        ((apy_df["State"]==state)&(apy_df["District"]==district), "district-all"),
        ((apy_df["State"]==state), "state-all"),
    ]
    for cond, tag in candidates:
        sub = apy_df.loc[cond, ["Crop_Year","Yield"]].dropna()
        if sub.empty:
            continue
        sub = sub.groupby("Crop_Year", as_index=False)["Yield"].mean().sort_values("Crop_Year")
        if len(sub) < 3:
            continue

        x = sub["Crop_Year"].astype(float).to_numpy()
        y = sub["Yield"].astype(float).to_numpy()
        x = x - x.mean()
        slope = float(np.polyfit(x, y, 1)[0])           # dy/dyear
        mean_y = float(max(sub["Yield"].mean(), 1e-9))  # avoid divide-by-0
        g = slope / mean_y                              # fractional growth per year
        g = float(np.clip(g, -0.10, 0.10))              # cap ±10%/yr
        return g, tag

    return 0.0, "none"

def predict_yield(model, model_mode, defaults_df, apy_df, crop_map: dict,
                  state, district, price_crop, years, apy_blend=0.30):
    mapped_crop = crop_map.get(price_crop, price_crop) if (apy_df is not None and not apy_df.empty) else price_crop
    season, area, prod = get_defaults(defaults_df, state, district, mapped_crop)

    Xp = pd.DataFrame({
        "State": [state]*len(years),
        "District": [district]*len(years),
        "Crop": [mapped_crop]*len(years),
        "Crop_Year": years,
        "Season": [season]*len(years),
        "Area": [area]*len(years),
        "Production": [prod]*len(years),
    })
    if model_mode == "xgb_categorical":
        for c in ["State","District","Crop","Season"]:
            if c in Xp.columns: Xp[c] = Xp[c].astype("category")

    # Base ML prediction
    yhat = np.asarray(model.predict(Xp), dtype=float)
    yhat = np.clip(yhat, 0.0, 50.0)

    # APY drift factors (always applied; 0 if no APY)
    g, src = yield_drift_from_history(apy_df, state, district, mapped_crop)
    if apy_blend and len(years) > 0:
        steps = np.arange(1, len(years)+1, dtype=float)
        factors = np.exp(g * steps)
        factors = np.clip(factors, 0.80, 1.25)  # guardrails over short horizons
        yhat = yhat * ((1.0 - apy_blend) + apy_blend * factors)

    return np.clip(yhat, 0.0, 50.0), mapped_crop, src

# ────────────────────────────────────────────────────────────────────────────────
# Load assets
# ────────────────────────────────────────────────────────────────────────────────
price_df = load_price()
apy_df   = load_apy()
raw_model = load_model()
if price_df.empty or raw_model is None:
    st.stop()

try:
    model, model_mode = wrap_bare_xgb_with_encoder_if_needed(raw_model, apy_df)
except Exception as e:
    st.error(f"Yield model needs preprocessing adjustments: {e}")
    st.stop()

defaults_df = build_defaults(apy_df)
price_unique_crops = (price_df["Commodity Name"].dropna().astype(str).unique().tolist()
                      if "Commodity Name" in price_df.columns else [])
crop_map = build_crop_map(apy_df, price_unique_crops)

# ────────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.subheader("Settings")
    horizon = st.slider("Forecast horizon (years)", 2, 6, 3, 1)
    apy_blend_pct = st.slider("Blend in APY yield trend (%, per year drift)", 0, 50, 30, 5,
                              help="0 = ML yield only · 50 = half driven by APY trend")
    apy_blend = apy_blend_pct / 100.0
    st.caption(f"Prices: using {PRICE_MIN_YEAR}+ history. If market has <3 yrs, we fallback to district/state.")

    st.subheader("Filters (India)")
    states = sorted(price_df["State"].dropna().unique().tolist()) if "State" in price_df.columns else []
    if not states:
        st.error("No 'State' in price data."); st.stop()
    sel_state = st.selectbox("State", states, index=0)

    districts = sorted(price_df.loc[price_df["State"]==sel_state, "District"].dropna().unique().tolist())
    sel_district = st.selectbox("District", districts, index=0)

    markets = sorted(price_df.loc[
        (price_df["State"]==sel_state) & (price_df["District"]==sel_district),
        "Market Name"
    ].dropna().unique().tolist())
    sel_market = st.selectbox("Market", markets, index=0)

    crops_all = sorted(price_df.loc[
        (price_df["State"]==sel_state)&
        (price_df["District"]==sel_district)&
        (price_df["Market Name"]==sel_market),
        "Commodity Name"
    ].dropna().unique().tolist())
    sel_crops = st.multiselect("Choose crops (up to 10)", crops_all, default=crops_all[: min(6, len(crops_all))])

    run = st.button("▶️ Run forecasts", use_container_width=True)

if not sel_crops:
    st.info("Pick at least one crop."); st.stop()
if not run:
    st.info("Adjust filters and click **Run forecasts**."); st.stop()

# ────────────────────────────────────────────────────────────────────────────────
# Forecasts & Revenue
# ────────────────────────────────────────────────────────────────────────────────
[tab1] = st.tabs(["📈 Forecasts & Revenue"])

with tab1:
    all_rows = []

    for crop in sel_crops:
        hist_df, level_used = yearly_price_with_fallback(price_df, sel_state, sel_district, sel_market, crop)
        if hist_df.empty:
            continue

        fc_df = log_arima_or_holt(hist_df, horizon=horizon)
        if fc_df.empty:
            continue

        years = fc_df["Year"].astype(int).tolist()
        yhat, mapped_crop, drift_src = predict_yield(
            model, model_mode, defaults_df, apy_df, crop_map,
            sel_state, sel_district, crop, years, apy_blend=apy_blend
        )
        revenue = fc_df["Forecast_Price"].values * yhat

        tmp = pd.DataFrame({
            "State": sel_state, "District": sel_district, "Market Name": sel_market,
            "Crop": crop, "APY_Crop": mapped_crop, "Year": years,
            "Forecast_Price": fc_df["Forecast_Price"].values.astype(float),
            "Pred_Yield": yhat.astype(float),
            "Projected_Revenue": revenue.astype(float),
            "Price_Level": level_used,
            "Yield_Drift_Source": drift_src
        })
        all_rows.append(tmp)

    if not all_rows:
        st.warning("Not enough history for the selected choices (even after district/state fallback).")
        st.stop()

    result = pd.concat(all_rows, ignore_index=True)

    # KPI row
    k1, k2 = st.columns(2)
    k1.metric("Crops forecasted", f"{result['Crop'].nunique()}")
    k2.metric("Forecast years", f"{int(result['Year'].min())}–{int(result['Year'].max())}")

    # Tables 
    st.subheader("Projected revenue (price × predicted yield)")
    show_df = result.copy()
    for c in ["State","District","Market Name","Crop","APY_Crop","Price_Level","Yield_Drift_Source"]:
        show_df[c] = show_df[c].astype(str)
    show_df["Year"] = show_df["Year"].astype(int)
    st.dataframe(
        show_df.sort_values(["Crop","Year"]).reset_index(drop=True),
        use_container_width=True, height=420,
        column_config={"Year": st.column_config.NumberColumn(format="%d", step=1)}
    )

    st.subheader("🌱 Recommended crops (by total projected revenue)")
    rec = (result.groupby(["Crop","APY_Crop","Price_Level"], as_index=False)
                 .agg(Total_Projected_Revenue=("Projected_Revenue","sum"),
                      Avg_Forecast_Price=("Forecast_Price","mean"),
                      Avg_Pred_Yield=("Pred_Yield","mean"))
                 .sort_values("Total_Projected_Revenue", ascending=False))
    st.dataframe(rec.head(10).reset_index(drop=True), use_container_width=True,
                 column_config={"Avg_Forecast_Price": st.column_config.NumberColumn(format="%.0f"),
                                "Avg_Pred_Yield": st.column_config.NumberColumn(format="%.3f")})

    with st.expander("Show all crops grown in this market (from price data)"):
        st.dataframe(pd.DataFrame({"Crop": sorted(crops_all)}), use_container_width=True, height=240)

    csv = result.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download forecasts (CSV)", data=csv,
                       file_name=f"forecasts_{sel_state}_{sel_district}_{sel_market}.csv",
                       mime="text/csv")

st.caption("Note: recommendations are a heuristic (price × predicted yield). Validate units/context before operational use.")