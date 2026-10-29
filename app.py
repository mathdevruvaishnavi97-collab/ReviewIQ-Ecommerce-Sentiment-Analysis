"""
ReviewIQ - E-commerce Review Sentiment Intelligence
---------------------------------------------------
Streamlit application for the Product_Review_Analysis_Ecommerce project.

Run with:   python -m streamlit run app.py

Everything is read from files that live next to this script:
    models/svm_model.joblib, models/tfidf_vectorizer.joblib
    results/model_comparison.csv (+ confusion matrix PNGs)
    dataset/Reviews.csv | processed_reviews.csv | preprocessed_reviews.csv
Missing files never crash the app; each page explains what is missing.
"""

from __future__ import annotations

import ast
import html
import importlib.util
import re
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="ReviewIQ - Review Sentiment Intelligence",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Paths (relative to this file, never hard-coded Windows paths)
# --------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DATASET_DIR = APP_DIR / "dataset"
MODELS_DIR = APP_DIR / "models"
RESULTS_DIR = APP_DIR / "results"

SVM_PATH = MODELS_DIR / "svm_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
METRICS_PATH = RESULTS_DIR / "model_comparison.csv"
DATASET_FILES = [
    DATASET_DIR / "Reviews.csv",
    DATASET_DIR / "processed_reviews.csv",
    DATASET_DIR / "preprocessed_reviews.csv",
]
CONFUSION_FILES = {
    "Logistic Regression": RESULTS_DIR / "logistic_regression_confusion_matrix.png",
    "Naive Bayes": RESULTS_DIR / "naive_bayes_confusion_matrix.png",
    "Linear SVM": RESULTS_DIR / "svm_confusion_matrix.png",
}

# --------------------------------------------------------------------------
# Design constants
# --------------------------------------------------------------------------
SENTS = ["Negative", "Neutral", "Positive"]
SENT_COLOR = {"Positive": "#07866F", "Neutral": "#B77912", "Negative": "#D74453"}
SENT_ICON = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
SENT_MEANING = {
    "Positive": "The model reads this review as expressing satisfaction or approval.",
    "Neutral": "The model reads this review as mixed, average, or mostly factual.",
    "Negative": "The model reads this review as expressing dissatisfaction or a problem.",
}
METRIC_COLOR = {
    "Accuracy": "#6546E8",
    "Precision": "#07866F",
    "Recall": "#B77912",
    "F1 Score": "#D74453",
}
METRICS = ["Accuracy", "Precision", "Recall", "F1 Score"]

NAV = [
    ("Overview", "home"),
    ("Analyze Reviews", "rate_review"),
    ("Model Studio", "analytics"),
    ("Review Explorer", "search"),
    ("Project Guide", "menu_book"),
]

HERO = {
    "Overview": (
        "Understand every review.",
        "Turn customer feedback into clear sentiment insights and practical next steps.",
    ),
    "Analyze Reviews": (
        "Discover the feeling behind the words.",
        "Submit a product review and see how the trained sentiment model classifies it.",
    ),
    "Model Studio": (
        "Measure model performance.",
        "Compare the saved evaluation scores of your machine-learning classifiers.",
    ),
    "Review Explorer": (
        "Explore the voice of customers.",
        "Search available review records and filter them by sentiment, rating, or text.",
    ),
    "Project Guide": (
        "Understand the intelligence pipeline.",
        "Explore how review text becomes a sentiment prediction.",
    ),
}

SAMPLES = {
    "Positive sample": (
        "This product is excellent and works perfectly. The quality is amazing "
        "and I am very happy with it."
    ),
    "Neutral sample": (
        "The product is okay. It works as expected, but the features and quality are average."
    ),
    "Negative sample": (
        "This product is terrible and completely useless. The quality is poor "
        "and I am very disappointed."
    ),
}

# Numeric label -> sentiment name. This is only used when the saved model
# returns numbers AND the mapping cannot be derived from the dataset.
# VERIFY this against your training script (the app tells the user when it
# is relying on this assumption).
DEFAULT_LABEL_MAP = {0: "Negative", 1: "Positive", 2: "Neutral"}

# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, .stApp, [class*="css"] {
  font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
}
.stApp { background: #F7F7FC; color: #17172E; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1440px; padding: 1.6rem 2.25rem 3rem 2.25rem; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
  background: #FFFFFF;
  border-right: 1px solid #E4E3EF;
}
[data-testid="stSidebar"][aria-expanded="true"] {
  width: 272px !important;
  min-width: 272px !important;
}
[data-testid="stSidebar"] .block-container,
[data-testid="stSidebarUserContent"] { padding-top: 1.2rem; }

.riq-brand { display:flex; align-items:center; gap:12px; margin: 4px 0 22px 0; }
.riq-brand-mark {
  width:46px; height:46px; border-radius:13px; background:#6546E8; color:#FFFFFF;
  display:flex; align-items:center; justify-content:center;
  font-size:26px; font-weight:900; line-height:1;
}
.riq-brand-title { font-size:22px; font-weight:800; color:#17172E; line-height:1.1; letter-spacing:-0.02em; }
.riq-brand-sub { font-size:10.5px; font-weight:750; color:#62647B; letter-spacing:0.12em; margin-top:3px; }
.riq-nav-head { font-size:11px; font-weight:800; color:#62647B; letter-spacing:0.14em; margin: 0 0 8px 4px; }

[data-testid="stSidebar"] .stButton > button {
  width: 100%;
  min-height: 46px;
  justify-content: flex-start;
  text-align: left;
  border-radius: 11px;
  border: 1px solid transparent;
  box-shadow: none;
  transition: background .15s ease, color .15s ease, border-color .15s ease;
}
[data-testid="stSidebar"] .stButton > button p { font-size: 15px; font-weight: 700; }
[data-testid="stSidebar"] .stButton > button[kind="secondary"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"] {
  background: #FFFFFF; color: #34334D;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] p,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"] p { color: #34334D; }
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"]:hover {
  background: #F0EDFF; border-color: #DDD5FF; color: #4328BA;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover p,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-secondary"]:hover p { color: #4328BA; }
[data-testid="stSidebar"] .stButton > button[kind="primary"],
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
  background: #6546E8; color: #FFFFFF; border-color: #6546E8;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] p,
[data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] p { color: #FFFFFF; }

.riq-sideinfo {
  background:#F0EDFF; border:1px solid #CFC5FF; border-radius:14px;
  padding:14px 16px; margin-top:26px; color:#17172E;
}
.riq-sideinfo .tag {
  display:inline-block; font-size:10.5px; font-weight:800; letter-spacing:0.12em;
  color:#4328BA; background:#FFFFFF; border:1px solid #CFC5FF; border-radius:999px;
  padding:3px 9px; margin-bottom:10px;
}
.riq-sideinfo div.line { font-size:13.5px; font-weight:500; color:#17172E; line-height:1.7; }

/* ---------- Hero ---------- */
.riq-hero {
  position:relative; overflow:hidden; background:#17172E; border-radius:22px;
  padding:36px 40px; min-height:180px; margin-bottom:28px;
}
.riq-hero .deco1 {
  position:absolute; right:-70px; top:-90px; width:320px; height:320px;
  border-radius:50%; background:#6546E8; opacity:0.55; z-index:0;
}
.riq-hero .deco2 {
  position:absolute; right:150px; bottom:-60px; width:130px; height:130px;
  border-radius:50%; background:#D9F76B; opacity:0.18; z-index:0;
}
.riq-hero .inner { position:relative; z-index:1; max-width:760px; }
.riq-hero .eyebrow {
  font-size:12px; font-weight:800; letter-spacing:0.16em; color:#D9F76B; margin-bottom:12px;
}
.riq-hero h1 {
  font-size:44px; font-weight:900; line-height:1.08; letter-spacing:-0.03em;
  color:#FFFFFF; margin:0 0 12px 0; padding:0;
}
.riq-hero p.desc { font-size:16px; line-height:1.6; color:#ECEBFA; margin:0; }

/* ---------- Typography blocks ---------- */
.riq-intro h2 { font-size:25px; font-weight:800; color:#17172E; margin:6px 0 4px 0; letter-spacing:-0.02em; padding:0; }
.riq-intro p { font-size:15px; color:#62647B; margin:0 0 16px 0; line-height:1.55; }
.riq-h3 { font-size:19px; font-weight:800; color:#17172E; margin:2px 0 4px 0; letter-spacing:-0.01em; }
.riq-sub { font-size:14px; color:#62647B; margin:0 0 12px 0; line-height:1.5; }
.riq-eyebrow { font-size:11.5px; font-weight:800; letter-spacing:0.14em; color:#6546E8; margin-bottom:6px; }

/* ---------- KPI strip ---------- */
.riq-kpis { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:16px; margin-bottom:26px; }
.riq-kpi {
  background:#FFFFFF; border:1px solid #E4E3EF; border-radius:17px;
  padding:18px 20px; min-height:148px;
}
.riq-kpi-label { font-size:12px; font-weight:800; letter-spacing:0.1em; text-transform:uppercase; color:#62647B; }
.riq-kpi-value { font-size:34px; font-weight:900; color:#17172E; margin:8px 0 6px 0; letter-spacing:-0.02em; line-height:1.1; }
.riq-kpi-note { font-size:13.5px; color:#62647B; line-height:1.45; }

/* ---------- Mix bars ---------- */
.riq-mix { margin: 18px 0 20px 0; }
.riq-mix-head { display:flex; justify-content:space-between; align-items:baseline; gap:10px; flex-wrap:wrap; margin-bottom:8px; }
.riq-mix-name { font-size:15px; font-weight:700; color:#17172E; }
.riq-mix-val { font-size:14px; color:#34334D; font-weight:500; }
.riq-mix-val b { font-size:17px; font-weight:800; color:#17172E; }
.riq-bar { background:#E9E8F1; border-radius:999px; height:12px; overflow:hidden; }
.riq-bar > div { height:12px; border-radius:999px; }

/* ---------- Insights ---------- */
.riq-insights { display:grid; grid-template-columns:1.25fr 1fr 1fr; gap:16px; margin-bottom:26px; }
.riq-insight { background:#FFFFFF; border:1px solid #E4E3EF; border-radius:16px; padding:18px 20px; }
.riq-insight .big { font-size:30px; font-weight:900; letter-spacing:-0.02em; line-height:1.1; }
.riq-insight .ttl { font-size:15px; font-weight:800; color:#17172E; margin:6px 0 6px 0; }
.riq-insight .txt { font-size:14px; color:#34334D; line-height:1.55; }

/* ---------- Quick actions ---------- */
.riq-qa-title { font-size:17px; font-weight:800; color:#17172E; margin:4px 0 4px 0; }
.riq-qa-desc { font-size:14px; color:#62647B; margin:0 0 10px 0; line-height:1.5; }
.riq-qa-icon {
  width:40px; height:40px; border-radius:11px; background:#F0EDFF; color:#4328BA;
  display:flex; align-items:center; justify-content:center; font-size:20px; margin-bottom:10px;
}

/* ---------- Result / review panels ---------- */
.riq-result {
  background:#FFFFFF; border:1px solid #E4E3EF; border-left-width:6px; border-radius:16px;
  padding:22px 24px; margin-bottom:12px;
}
.riq-result .lbl { font-size:11.5px; font-weight:800; letter-spacing:0.14em; color:#62647B; }
.riq-result .val { font-size:40px; font-weight:900; letter-spacing:-0.02em; margin:6px 0 8px 0; line-height:1.1; }
.riq-result .meaning { font-size:15.5px; color:#17172E; line-height:1.55; }
.riq-note { font-size:13.5px; color:#62647B; line-height:1.55; }
.riq-empty {
  background:#FFFFFF; border:1px dashed #CFC9E8; border-radius:16px; padding:26px 24px;
  color:#34334D; font-size:15px; line-height:1.6;
}
.riq-review-box {
  background:#FFFFFF; border:1px solid #E4E3EF; border-radius:14px; padding:18px 20px;
  font-size:15.5px; color:#17172E; line-height:1.65; word-wrap:break-word; overflow-wrap:anywhere;
}
.riq-chip {
  display:inline-block; font-size:13px; font-weight:700; border-radius:999px;
  padding:3px 12px; border:1px solid #E4E3EF; background:#FFFFFF; color:#17172E; margin-right:8px;
}

/* ---------- Workflow / status ---------- */
.riq-flow { display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:16px; margin: 6px 0 26px 0; }
.riq-step { background:#FFFFFF; border:1px solid #E4E3EF; border-radius:16px; padding:20px; position:relative; }
.riq-step .num {
  width:30px; height:30px; border-radius:50%; background:#6546E8; color:#FFFFFF;
  display:inline-flex; align-items:center; justify-content:center; font-size:14px; font-weight:800;
}
.riq-step .ico { float:right; font-size:26px; }
.riq-step .ttl { font-size:17px; font-weight:800; color:#17172E; margin:12px 0 6px 0; }
.riq-step .txt { font-size:14.5px; color:#34334D; line-height:1.55; }
.riq-stack { display:grid; grid-template-columns:repeat(auto-fit, minmax(240px, 1fr)); gap:16px; margin-bottom:26px; }
.riq-stack .box { background:#FFFFFF; border:1px solid #E4E3EF; border-radius:16px; padding:18px 20px; }
.riq-stack .box ul { margin:8px 0 0 0; padding-left:20px; }
.riq-stack .box li { font-size:14.5px; color:#17172E; line-height:1.7; }
.riq-status { width:100%; border-collapse:collapse; background:#FFFFFF; border:1px solid #E4E3EF; border-radius:14px; overflow:hidden; }
.riq-status td { padding:12px 16px; border-bottom:1px solid #E4E3EF; font-size:14.5px; color:#17172E; word-break:break-all; }
.riq-status tr:last-child td { border-bottom:none; }
.riq-ok { color:#07866F; font-weight:800; white-space:nowrap; }
.riq-miss { color:#D74453; font-weight:800; white-space:nowrap; }

/* ---------- Widgets ---------- */
[data-testid="stWidgetLabel"] p { color:#17172E; font-size:15px; font-weight:650; }
.stTextArea textarea {
  background:#FFFFFF; color:#17172E; font-size:16px; line-height:1.55;
}
.stTextArea [data-baseweb="textarea"], .stTextInput [data-baseweb="input"], .stSelectbox [data-baseweb="select"] > div {
  background:#FFFFFF; border-radius:12px; border:1px solid #D5D3E5;
}
.stTextArea [data-baseweb="textarea"]:focus-within,
.stTextInput [data-baseweb="input"]:focus-within,
.stSelectbox [data-baseweb="select"] > div:focus-within { border-color:#6546E8; box-shadow:0 0 0 2px #DDD5FF; }
.stTextInput input { color:#17172E; font-size:15px; background:#FFFFFF; }
.stTextArea textarea::placeholder, .stTextInput input::placeholder { color:#7A7C93; opacity:1; }

.stButton > button { min-height:46px; border-radius:11px; font-weight:700; transition: background .15s ease, color .15s ease; }
.stButton > button p { font-size:14.5px; font-weight:700; }
.stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"] {
  background:#6546E8; border:1px solid #6546E8; color:#FFFFFF;
}
.stButton > button[kind="primary"] p, .stButton > button[data-testid="stBaseButton-primary"] p { color:#FFFFFF; }
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="stBaseButton-primary"]:hover {
  background:#4E31C8; border-color:#4E31C8; color:#FFFFFF;
}
.stButton > button[kind="secondary"], .stButton > button[data-testid="stBaseButton-secondary"] {
  background:#FFFFFF; border:1px solid #D5D3E5; color:#17172E;
}
.stButton > button[kind="secondary"] p, .stButton > button[data-testid="stBaseButton-secondary"] p { color:#17172E; }
.stButton > button[kind="secondary"]:hover, .stButton > button[data-testid="stBaseButton-secondary"]:hover {
  background:#F0EDFF; color:#4328BA; border-color:#DDD5FF;
}
.stButton > button[kind="secondary"]:hover p, .stButton > button[data-testid="stBaseButton-secondary"]:hover p { color:#4328BA; }
.stButton > button:focus-visible { outline:3px solid #6546E8; outline-offset:2px; }
.main .stButton > button, [data-testid="stMain"] .stButton > button { width:100%; }

[data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px; }
[data-testid="stExpander"] summary p { font-size:15px; font-weight:650; color:#17172E; }
[data-testid="stCaptionContainer"] { color:#62647B; }

/* ---------- Responsive ---------- */
@media (max-width: 1100px) {
  .riq-kpis { grid-template-columns:repeat(2, minmax(0,1fr)); }
  .riq-insights { grid-template-columns:1fr; }
}
@media (max-width: 640px) {
  .block-container { padding: 1rem 1rem 2.5rem 1rem; }
  .riq-hero { padding:26px 22px; }
  .riq-hero h1 { font-size:31px; }
  .riq-kpis { grid-template-columns:1fr; }
  .riq-kpi-value { font-size:28px; }
  .riq-result .val { font-size:32px; }
}
</style>
"""


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def esc(text) -> str:
    """HTML-escape text and neutralise characters Markdown might treat specially."""
    return html.escape(str(text)).replace("$", "&#36;").replace("\n", "<br>")


def render(markup: str) -> None:
    """Render an HTML snippet safely (strips indentation so Markdown never
    mistakes it for a code block)."""
    flat = "".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


def panel():
    """A bordered container (falls back to a plain one on old Streamlit)."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def show_chart(chart) -> None:
    try:
        st.altair_chart(chart, width="stretch")
    except Exception:
        st.altair_chart(chart, use_container_width=True)


def styled_button(label, key, icon=None, kind="secondary", on_click=None, args=None) -> bool:
    kwargs = {"key": key, "type": kind, "on_click": on_click, "args": args}
    if icon:
        try:
            return st.button(label, icon=f":material/{icon}:", **kwargs)
        except Exception:
            pass
    return st.button(label, **kwargs)


def file_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def go(page: str) -> None:
    st.session_state.page = page


def set_sample(text: str) -> None:
    st.session_state.review_text = text


def fmt_int(n) -> str:
    return f"{int(n):,}"


# --------------------------------------------------------------------------
# Text cleaning (reuses the project's own function whenever possible)
# --------------------------------------------------------------------------
CLEAN_NAMES = [
    "clean_text", "preprocess_text", "preprocess", "clean_review", "clean_reviews",
    "text_preprocessing", "preprocess_review", "text_cleaning", "clean",
]


def _fallback_clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _inspect_module(path: Path):
    """Statically check a project module. Returns (safe_to_import, function_names).
    A module is considered unsafe if it runs heavy work (e.g. read_csv) at import time."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception:
        return False, []
    heavy = {"read_csv", "to_csv", "read_excel", "fit", "fit_transform", "dump", "train_test_split"}
    funcs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            funcs.append(node.name)
            continue
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef)):
            continue
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
        ):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                fn = sub.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if name in heavy:
                    return False, funcs
    return True, funcs


@st.cache_resource(show_spinner=False)
def get_cleaner(_mtimes):
    for fname in ("preprocess_text.py", "predict_sentiment.py"):
        path = APP_DIR / fname
        if not path.exists():
            continue
        safe, funcs = _inspect_module(path)
        if not safe:
            continue
        pick = next((n for n in CLEAN_NAMES if n in funcs), None)
        if not pick:
            continue
        try:
            if str(APP_DIR) not in sys.path:
                sys.path.insert(0, str(APP_DIR))
            spec = importlib.util.spec_from_file_location(f"riq_{path.stem}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn = getattr(module, pick)
            if isinstance(fn("This is a test review."), str):
                return fn, f"{fname} -> {pick}()", True
        except Exception:
            continue
    return _fallback_clean, "built-in basic cleaning (project cleaning function not found)", False


def cleaner():
    return get_cleaner(
        (file_mtime(APP_DIR / "preprocess_text.py"), file_mtime(APP_DIR / "predict_sentiment.py"))
    )


# --------------------------------------------------------------------------
# Model artifacts
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts(_m1, _m2):
    missing = [p.name for p in (SVM_PATH, VECTORIZER_PATH) if not p.exists()]
    if missing:
        return {"model": None, "vectorizer": None, "error": "Missing file(s): " + ", ".join(missing)}
    try:
        import joblib

        return {
            "model": joblib.load(SVM_PATH),
            "vectorizer": joblib.load(VECTORIZER_PATH),
            "error": None,
        }
    except Exception as exc:  # corrupted file, version mismatch, joblib missing...
        return {"model": None, "vectorizer": None, "error": f"Could not load model files: {exc}"}


def artifacts():
    return load_artifacts(file_mtime(SVM_PATH), file_mtime(VECTORIZER_PATH))


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
TEXT_CANDS = ["text", "review", "reviewtext", "content"]
SUMMARY_CANDS = ["summary"]
SCORE_CANDS = ["score", "rating"]
SENT_CANDS = ["sentiment", "label"]
PRODUCT_CANDS = ["productid", "product", "productname", "asin"]
LABEL_EXTRA = ["sentiment_label", "sentimentlabel", "sentiment_text", "label_name",
               "sentiment_encoded", "sentiment_id", "target"]


def _first(lowmap: dict, cands):
    for c in cands:
        if c in lowmap:
            return lowmap[c]
    return None


def _derive_label_map(raw: pd.DataFrame):
    """If the data holds both a numeric label and its text sentiment, learn the mapping."""
    text_col, num_col = None, None
    for col in raw.columns:
        s = raw[col].dropna().head(5000)
        if s.empty:
            continue
        if s.dtype == object and s.astype(str).str.strip().str.title().isin(SENTS).all():
            text_col = text_col or col
        elif pd.api.types.is_numeric_dtype(s) and s.nunique() <= 3 and col.lower() not in SCORE_CANDS:
            num_col = num_col or col
    if text_col is None or num_col is None:
        return None
    pair = raw[[num_col, text_col]].dropna()
    if pair.empty:
        return None
    pair[text_col] = pair[text_col].astype(str).str.strip().str.title()
    mapping = {}
    for num, grp in pair.groupby(num_col):
        top = grp[text_col].value_counts()
        if top.iloc[0] / len(grp) < 0.95:
            return None
        mapping[int(num)] = top.index[0]
    return mapping or None


def _normalize_sentiment(series: pd.Series, label_map: dict) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.map(label_map)
    s = series.astype(str).str.strip().str.title()
    return s.where(s.isin(SENTS))


@st.cache_data(show_spinner=False)
def load_dataset(_mtimes):
    """Pick the best dataset file and standardise it to the columns
    Review, Summary, Rating, Product, Sentiment (whichever exist)."""
    found, seen = [], {}
    for path in DATASET_FILES:
        if not path.exists():
            continue
        try:
            cols = list(pd.read_csv(path, nrows=0).columns)
        except Exception as exc:
            seen[path.name] = f"unreadable ({exc})"
            continue
        seen[path.name] = cols
        low = {str(c).strip().lower(): c for c in cols}
        text = _first(low, TEXT_CANDS) or _first(low, SUMMARY_CANDS)
        if text is None:
            continue
        sent = _first(low, SENT_CANDS)
        score = _first(low, SCORE_CANDS)
        tier = 0 if sent else (1 if score else 2)
        found.append((tier, path, low, text, sent, score))

    if not found:
        return {"df": None, "seen": seen,
                "reason": "No dataset CSV with a recognizable review-text column was found."}

    found.sort(key=lambda t: t[0])
    _, path, low, text_col, sent_col, score_col = found[0]
    summary_col = _first(low, SUMMARY_CANDS)
    product_col = _first(low, PRODUCT_CANDS)
    wanted = {text_col, sent_col, score_col, summary_col, product_col}
    wanted |= {low[c] for c in LABEL_EXTRA if c in low}
    wanted.discard(None)

    def read(enc):
        return pd.read_csv(path, usecols=lambda c: c in wanted, encoding=enc, low_memory=False)

    try:
        try:
            raw = read("utf-8")
        except UnicodeDecodeError:
            raw = read("latin-1")
    except Exception as exc:
        return {"df": None, "seen": seen, "reason": f"Could not read {path.name}: {exc}"}

    out = pd.DataFrame({"Review": raw[text_col].fillna("").astype(str)})
    if summary_col and summary_col != text_col:
        out["Summary"] = raw[summary_col].fillna("").astype(str)
    if product_col:
        out["Product"] = raw[product_col].astype(str)
    if score_col:
        out["Rating"] = pd.to_numeric(raw[score_col], errors="coerce")

    derived_map = _derive_label_map(raw)
    label_map = derived_map or DEFAULT_LABEL_MAP
    sentiment, source = None, None
    if sent_col:
        cand = _normalize_sentiment(raw[sent_col], label_map)
        if cand.notna().any():
            sentiment = cand
            if pd.api.types.is_numeric_dtype(raw[sent_col]):
                source = ("numeric labels decoded with a mapping learned from the dataset"
                          if derived_map else
                          "numeric labels decoded with an ASSUMED mapping (0=Negative, 1=Positive, 2=Neutral)")
            else:
                source = f"taken from the '{sent_col}' column"
    if sentiment is None and "Rating" in out.columns:
        r = out["Rating"]
        sentiment = pd.Series(
            np.select([r >= 4, r == 3, (r >= 1) & (r < 3)], ["Positive", "Neutral", "Negative"], default=""),
            index=out.index,
        ).replace("", np.nan)
        source = "derived from rating (4-5 Positive, 3 Neutral, 1-2 Negative)"
    if sentiment is not None:
        out["Sentiment"] = sentiment.astype(pd.CategoricalDtype(SENTS))

    return {"df": out, "seen": seen, "path": path.name, "sentiment_source": source,
            "label_map": derived_map, "reason": None}


def get_dataset():
    return load_dataset(tuple(file_mtime(p) for p in DATASET_FILES))


def sentiment_counts(df: pd.DataFrame):
    if "Sentiment" not in df.columns:
        return None
    vc = df["Sentiment"].value_counts()
    counts = {s: int(vc.get(s, 0)) for s in SENTS}
    return counts if sum(counts.values()) > 0 else None


# --------------------------------------------------------------------------
# Model metrics CSV
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_metrics(_mtime):
    if not METRICS_PATH.exists():
        return None, {}, "results/model_comparison.csv was not found."
    try:
        raw = pd.read_csv(METRICS_PATH)
    except Exception as exc:
        return None, {}, f"Could not read model_comparison.csv: {exc}"
    if raw.empty:
        return None, {}, "model_comparison.csv is empty."

    def norm(c):
        return re.sub(r"[^a-z0-9]", "", str(c).lower())

    cols = {norm(c): c for c in raw.columns}
    model_col = next((cols[k] for k in cols if k in ("model", "modelname", "classifier", "algorithm")), None)
    if model_col is None:
        first = raw.columns[0]
        if raw[first].dtype == object:
            model_col = first
        else:
            return None, {}, "No model-name column was found in model_comparison.csv."

    keys = {
        "Accuracy": ("accuracy", "acc"),
        "Precision": ("precision",),
        "Recall": ("recall",),
        "F1 Score": ("f1score", "f1", "fscore"),
    }
    used = {}
    for metric, prefixes in keys.items():
        for k, original in cols.items():
            if original != model_col and any(k.startswith(p) for p in prefixes):
                used[metric] = original
                break
    if not used:
        return None, {}, "No Accuracy / Precision / Recall / F1 columns were recognized."

    out = pd.DataFrame({"Model": raw[model_col].astype(str).str.strip()})
    for metric, original in used.items():
        vals = raw[original]
        if vals.dtype == object:
            vals = vals.astype(str).str.replace("%", "", regex=False).str.strip()
        out[metric] = pd.to_numeric(vals, errors="coerce")

    stacked = out[list(used)].stack()
    if stacked.empty:
        return None, {}, "The metric columns contain no numeric values."
    scale = "decimal fractions (x100 applied)" if stacked.max() <= 1.0 else "percentages (kept as is)"
    if stacked.max() <= 1.0:
        out[list(used)] = out[list(used)] * 100.0
    return out, {"columns": used, "scale": scale}, None


def get_metrics():
    return load_metrics(file_mtime(METRICS_PATH))


def best_metric(df, metric):
    if df is None or metric not in df.columns or df[metric].dropna().empty:
        return None
    idx = df[metric].idxmax()
    return float(df.loc[idx, metric]), str(df.loc[idx, "Model"])


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def decode_label(pred, model, label_map_source):
    """Convert whatever the model returns to Negative/Neutral/Positive."""
    if isinstance(pred, (str, np.str_)):
        name = str(pred).strip().title()
        if name in SENTS:
            return name, "labels returned by the model as text"
        raise ValueError(f"Unexpected text label from model: {pred!r}")
    try:
        key = int(pred)
    except Exception:
        raise ValueError(f"Unexpected label from model: {pred!r}")
    ds = get_dataset()
    learned = ds.get("label_map") if isinstance(ds, dict) else None
    mapping = learned or DEFAULT_LABEL_MAP
    if key not in mapping:
        raise ValueError(f"Model returned label {key}, which is not in the mapping {mapping}.")
    note = ("numeric label decoded with a mapping learned from the dataset" if learned
            else "numeric label decoded with the ASSUMED mapping 0=Negative, 1=Positive, 2=Neutral - "
                 "please verify it against your training script")
    return mapping[key], note


def predict_text(text: str) -> dict:
    art = artifacts()
    if art["error"]:
        raise RuntimeError(art["error"])
    clean_fn, clean_src, is_project = cleaner()
    cleaned = clean_fn(text)
    if not isinstance(cleaned, str) or not cleaned.strip():
        raise ValueError("After cleaning, no usable words remained in this review. "
                         "Try a longer or more descriptive review.")
    features = art["vectorizer"].transform([cleaned])
    pred = art["model"].predict(features)[0]
    label, label_note = decode_label(pred, art["model"], None)
    return {
        "label": label,
        "original": text,
        "cleaned": cleaned,
        "clean_source": clean_src,
        "clean_is_project": is_project,
        "label_note": label_note,
    }


# --------------------------------------------------------------------------
# Reusable UI pieces
# --------------------------------------------------------------------------
def hero(page: str) -> None:
    heading, desc = HERO[page]
    render(f"""
    <div class="riq-hero">
      <div class="deco1"></div><div class="deco2"></div>
      <div class="inner">
        <div class="eyebrow" style="color:#D9F76B;">REVIEWIQ / CUSTOMER REVIEW INTELLIGENCE</div>
        <h1 style="color:#FFFFFF;">{esc(heading)}</h1>
        <p class="desc" style="color:#ECEBFA;">{esc(desc)}</p>
      </div>
    </div>
    """)


def intro(title: str, desc: str) -> None:
    render(f'<div class="riq-intro"><h2>{esc(title)}</h2><p>{esc(desc)}</p></div>')


def kpi_card(label, value, note, color) -> str:
    return (
        f'<div class="riq-kpi" style="border-top:4px solid {color};">'
        f'<div class="riq-kpi-label">{esc(label)}</div>'
        f'<div class="riq-kpi-value">{esc(value)}</div>'
        f'<div class="riq-kpi-note">{esc(note)}</div></div>'
    )


def empty_state(title: str, body: str) -> None:
    render(f'<div class="riq-empty"><b>{esc(title)}</b><br>{esc(body)}</div>')


def sentiment_chart(counts: dict):
    data = pd.DataFrame({"Sentiment": SENTS, "Reviews": [counts[s] for s in SENTS]})
    base = alt.Chart(data).encode(x=alt.X("Sentiment:N", sort=SENTS, axis=alt.Axis(labelAngle=0, title=None)))
    bars = base.mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=90).encode(
        y=alt.Y("Reviews:Q", axis=alt.Axis(title="Number of reviews", format=",d")),
        color=alt.Color(
            "Sentiment:N",
            scale=alt.Scale(domain=SENTS, range=[SENT_COLOR[s] for s in SENTS]),
            legend=None,
        ),
        tooltip=[alt.Tooltip("Sentiment:N"), alt.Tooltip("Reviews:Q", format=",", title="Reviews")],
    )
    labels = base.mark_text(dy=-10, fontSize=14, fontWeight=700, color="#17172E").encode(
        y="Reviews:Q", text=alt.Text("Reviews:Q", format=",")
    )
    return (
        alt.layer(bars, labels)
        .properties(height=330, background="transparent")
        .configure_axis(labelFontSize=13, titleFontSize=13, labelColor="#34334D", titleColor="#34334D",
                        gridColor="#ECEBF5", domainColor="#D5D3E5")
        .configure_view(strokeWidth=0)
    )


def metrics_chart(df: pd.DataFrame):
    present = [m for m in METRICS if m in df.columns]
    long = df.melt(id_vars="Model", value_vars=present, var_name="Metric", value_name="Score").dropna()
    color = alt.Color(
        "Metric:N",
        sort=present,
        scale=alt.Scale(domain=present, range=[METRIC_COLOR[m] for m in present]),
        legend=alt.Legend(title="Metric", labelFontSize=13, titleFontSize=13, orient="bottom"),
    )
    y = alt.Y("Score:Q", scale=alt.Scale(domain=[0, 100]),
              axis=alt.Axis(title="Score (%)", labelExpr="datum.value + '%'"))
    tip = [alt.Tooltip("Model:N"), alt.Tooltip("Metric:N"), alt.Tooltip("Score:Q", format=".2f", title="Score (%)")]
    try:
        alt5 = int(alt.__version__.split(".")[0]) >= 5
    except Exception:
        alt5 = False
    if alt5:
        chart = alt.Chart(long).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Model:N", axis=alt.Axis(labelAngle=0, title=None, labelLimit=220)),
            xOffset=alt.XOffset("Metric:N", sort=present),
            y=y, color=color, tooltip=tip,
        ).properties(height=360, background="transparent")
    else:
        chart = alt.Chart(long).mark_bar().encode(
            x=alt.X("Metric:N", sort=present, axis=alt.Axis(labels=False, title=None, ticks=False)),
            y=y, color=color, tooltip=tip,
            column=alt.Column("Model:N", header=alt.Header(labelFontSize=13, title=None)),
        ).properties(width=90, height=330, background="transparent")
    return chart.configure_axis(labelFontSize=13, titleFontSize=13, labelColor="#34334D",
                                titleColor="#34334D", gridColor="#ECEBF5", domainColor="#D5D3E5"
                                ).configure_view(strokeWidth=0)


def result_panel(res: dict) -> None:
    label = res["label"]
    color = SENT_COLOR[label]
    render(f"""
    <div class="riq-result" style="border-left-color:{color};">
      <div class="lbl">MODEL PREDICTION</div>
      <div class="val" style="color:{color};">{SENT_ICON[label]} {esc(label)}</div>
      <div class="meaning">{esc(SENT_MEANING[label])}</div>
    </div>
    """)
    with st.expander("Original review", expanded=False):
        render(f'<div class="riq-review-box">{esc(res["original"])}</div>')
    with st.expander("Cleaned model input", expanded=False):
        render(f'<div class="riq-review-box">{esc(res["cleaned"])}</div>')
        st.caption(f"Cleaning source: {res['clean_source']}")
    if not res["clean_is_project"]:
        st.warning("The project's own cleaning function could not be loaded, so basic fallback "
                   "cleaning was used. Results may differ from your training pipeline.")
    st.caption(f"Label decoding: {res['label_note']}.")
    render('<div class="riq-note">This prediction comes from a statistical model and may be incorrect. '
           'Use it as a guide, not as a guaranteed judgement.</div>')


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_overview() -> None:
    hero("Overview")
    ds = get_dataset()
    metrics_df, _, metrics_err = get_metrics()
    intro("Your review snapshot", "The essential numbers first, with more detail just below.")

    df = ds["df"]
    counts = sentiment_counts(df) if df is not None else None
    best_acc = best_metric(metrics_df, "Accuracy")

    if df is None:
        empty_state("Dataset statistics cannot be calculated.",
                    ds.get("reason", "No dataset is available.") +
                    " Place Reviews.csv (or a processed CSV) inside the dataset folder.")
    elif counts is None:
        empty_state("Sentiment statistics cannot be calculated.",
                    f"{ds['path']} loaded, but no recognizable sentiment, label, or rating column was found. "
                    f"Columns present: {', '.join(map(str, ds['seen'].get(ds['path'], [])))}.")
    else:
        labeled = sum(counts.values())
        pct = {s: counts[s] / labeled * 100 for s in SENTS}
        acc_val = f"{best_acc[0]:.2f}%" if best_acc else "Not available"
        acc_note = best_acc[1] if best_acc else (metrics_err or "Accuracy column not found")
        render('<div class="riq-kpis">' + "".join([
            kpi_card("Total Reviews", fmt_int(len(df)), f"{fmt_int(labeled)} records carry a sentiment label", "#6546E8"),
            kpi_card("Positive Reviews", fmt_int(counts["Positive"]), f"{pct['Positive']:.2f}% of labeled reviews", SENT_COLOR["Positive"]),
            kpi_card("Negative Reviews", fmt_int(counts["Negative"]), f"{pct['Negative']:.2f}% of labeled reviews", SENT_COLOR["Negative"]),
            kpi_card("Best Model Accuracy", acc_val, acc_note, "#D9F76B"),
        ]) + "</div>")
        st.caption(f"Source: dataset/{ds['path']} - sentiment {ds['sentiment_source']}.")

        left, right = st.columns([1.4, 1], gap="large")
        with left:
            with panel():
                render('<div class="riq-h3">Sentiment landscape</div>'
                       '<div class="riq-sub">Compare the number of positive, neutral, and negative reviews.</div>')
                show_chart(sentiment_chart(counts))
                st.caption(f"Chart represents {fmt_int(labeled)} labeled reviews.")
        with right:
            with panel():
                rows = ""
                for s in ["Positive", "Neutral", "Negative"]:
                    rows += (
                        f'<div class="riq-mix"><div class="riq-mix-head">'
                        f'<span class="riq-mix-name">{SENT_ICON[s]} {s}</span>'
                        f'<span class="riq-mix-val"><b>{pct[s]:.2f}%</b> &middot; {fmt_int(counts[s])} reviews</span></div>'
                        f'<div class="riq-bar" role="progressbar" aria-label="{s} share" aria-valuemin="0" '
                        f'aria-valuemax="100" aria-valuenow="{pct[s]:.2f}">'
                        f'<div style="width:{pct[s]:.2f}%;background:{SENT_COLOR[s]};"></div></div></div>'
                    )
                render('<div class="riq-h3">Customer sentiment mix</div>'
                       '<div class="riq-sub">Share of labeled reviews in each sentiment class.</div>' + rows)

        intro("What the data tells us", "Summaries calculated from the currently loaded dataset.")
        render(f"""
        <div class="riq-insights">
          <div class="riq-insight" style="border-top:4px solid {SENT_COLOR['Positive']};">
            <div class="big" style="color:{SENT_COLOR['Positive']};">{pct['Positive']:.2f}%</div>
            <div class="ttl">Positive feedback</div>
            <div class="txt">Of the labeled reviews, this share is positive. Positive reviews can help identify features customers appreciate.</div>
          </div>
          <div class="riq-insight" style="border-top:4px solid {SENT_COLOR['Neutral']};">
            <div class="big" style="color:{SENT_COLOR['Neutral']};">{pct['Neutral']:.2f}%</div>
            <div class="ttl">Neutral feedback</div>
            <div class="txt">These reviews may describe average, mixed, or factual experiences.</div>
          </div>
          <div class="riq-insight" style="border-top:4px solid {SENT_COLOR['Negative']};">
            <div class="big" style="color:{SENT_COLOR['Negative']};">{pct['Negative']:.2f}%</div>
            <div class="ttl">Negative feedback</div>
            <div class="txt">Negative feedback may help identify issues that deserve investigation.</div>
          </div>
        </div>
        """)

    intro("Quick actions", "Jump straight into the part of ReviewIQ you need.")
    actions = [
        ("rate_review", "✍️", "Analyze a review", "Test your own product review with the trained model.", "Analyze Reviews"),
        ("analytics", "📊", "Model Studio", "Compare accuracy, precision, recall and F1 score.", "Model Studio"),
        ("search", "🔎", "Review Explorer", "Browse and filter the review dataset.", "Review Explorer"),
    ]
    cols = st.columns(3, gap="medium")
    for col, (icon, emoji, title, desc, target) in zip(cols, actions):
        with col:
            with panel():
                render(f'<div class="riq-qa-icon">{emoji}</div><div class="riq-qa-title">{esc(title)}</div>'
                       f'<div class="riq-qa-desc">{esc(desc)}</div>')
                styled_button(f"Open {title}", f"qa_{target}", icon="arrow_forward", kind="primary",
                              on_click=go, args=(target,))


def page_analyze() -> None:
    hero("Analyze Reviews")
    art = artifacts()
    if art["error"]:
        st.error(f"The sentiment model is not ready. {art['error']}")

    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        render('<div class="riq-h3">Test a customer review</div>'
               '<div class="riq-sub">Paste a product review below or try one of the sample inputs.</div>')
        st.text_area(
            "Customer review text",
            key="review_text",
            height=220,
            placeholder="Example: The battery lasts all day and the screen is gorgeous...",
        )
        text = st.session_state.get("review_text", "")
        st.caption(f"{len(text.split())} words · {len(text)} characters")

        analyze = styled_button("Analyze Sentiment", "analyze_btn", kind="primary")

        render('<div class="riq-note" style="margin:14px 0 8px 0;">Or load a sample review:</div>')
        c1, c2, c3 = st.columns(3)
        for col, (name, sample) in zip((c1, c2, c3), SAMPLES.items()):
            with col:
                styled_button(name, f"sample_{name}", on_click=set_sample, args=(sample,))

        if analyze:
            if not text.strip():
                st.session_state.result = None
                st.session_state.result_error = "Please enter a review before analyzing."
            else:
                try:
                    with st.spinner("Analyzing sentiment..."):
                        st.session_state.result = predict_text(text)
                        st.session_state.result_error = None
                except Exception as exc:
                    st.session_state.result = None
                    st.session_state.result_error = f"Prediction failed: {exc}"

    with right:
        if st.session_state.get("result_error"):
            st.error(st.session_state.result_error)
        res = st.session_state.get("result")
        if res:
            result_panel(res)
        elif not st.session_state.get("result_error"):
            empty_state("No prediction yet.",
                        "Enter a review and press Analyze Sentiment. The result, the original text and the "
                        "cleaned model input will appear here.")


def page_models() -> None:
    hero("Model Studio")
    df, info, err = get_metrics()
    if df is None:
        st.warning(f"Model metrics cannot be shown. {err}")
        empty_state("Evaluation data unavailable.",
                    "Place results/model_comparison.csv in the project and reload the page.")
    else:
        intro("Best saved scores", "Highest value found for each metric in results/model_comparison.csv.")
        cards = []
        for metric in METRICS:
            best = best_metric(df, metric)
            if best:
                cards.append(kpi_card(metric, f"{best[0]:.2f}%", f"Best: {best[1]}", METRIC_COLOR[metric]))
            else:
                cards.append(kpi_card(metric, "Not available", "Column not found in the CSV", METRIC_COLOR[metric]))
        render('<div class="riq-kpis">' + "".join(cards) + "</div>")

        with panel():
            render('<div class="riq-h3">Model comparison</div>'
                   '<div class="riq-sub">All saved metrics for every evaluated model.</div>')
            show_chart(metrics_chart(df))
            st.caption(f"Chart represents {len(df)} models. Values read as {info['scale']}.")

        intro("Evaluation table", "Scores formatted as percentages with two decimal places.")
        shown = df.copy()
        for metric in METRICS:
            if metric in shown.columns:
                shown[metric] = shown[metric].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}%")
        st.dataframe(shown, hide_index=True)
        used = ", ".join(f"{m} = '{c}'" for m, c in info["columns"].items())
        st.caption(f"CSV columns used: {used}. If your CSV mixes macro, weighted or overall averages, "
                   "check which one these columns represent.")

    intro("Confusion matrices", "How each classifier's predictions compare with the true labels.")
    cols = st.columns(3, gap="medium")
    for col, (name, path) in zip(cols, CONFUSION_FILES.items()):
        with col:
            with panel():
                render(f'<div class="riq-h3">{esc(name)}</div>')
                if path.exists():
                    st.image(str(path), caption=f"{name} confusion matrix")
                else:
                    st.info(f"Image not found: results/{path.name}")

    with st.expander("What do these metrics mean?"):
        st.markdown(
            "- **Accuracy** - the share of all reviews the model classified correctly.\n"
            "- **Precision** - of the reviews the model labeled with a class, how many truly belong to it.\n"
            "- **Recall** - of the reviews that truly belong to a class, how many the model found.\n"
            "- **F1 score** - a single balanced score combining precision and recall.\n\n"
            "A *macro* average treats the three classes equally, while a *weighted* average gives larger "
            "classes more influence. Because Positive reviews dominate the dataset, the two can differ a lot."
        )


def page_explorer() -> None:
    hero("Review Explorer")
    ds = get_dataset()
    df = ds["df"]
    if df is None:
        st.warning(ds.get("reason", "No dataset is available."))
        if ds["seen"]:
            st.write("Files and columns found:")
            for name, cols in ds["seen"].items():
                st.write(f"**{name}**: {cols if isinstance(cols, str) else ', '.join(map(str, cols))}")
        else:
            st.info("Place Reviews.csv inside the dataset folder.")
        return

    st.caption(f"Dataset: dataset/{ds['path']} · {fmt_int(len(df))} records")
    has_sent = "Sentiment" in df.columns
    has_rating = "Rating" in df.columns

    c1, c2, c3, c4 = st.columns([1, 1, 2, 1], gap="medium")
    with c1:
        sent_sel = st.selectbox("Sentiment", ["All"] + SENTS, disabled=not has_sent,
                                help=None if has_sent else "No sentiment information in this dataset.")
    with c2:
        ratings = []
        if has_rating:
            vals = sorted(df["Rating"].dropna().unique().tolist())
            ratings = [int(v) if float(v).is_integer() else v for v in vals]
        rating_sel = st.selectbox("Rating", ["All"] + ratings, disabled=not has_rating,
                                  help=None if has_rating else "No rating column in this dataset.")
    with c3:
        query = st.text_input("Search review text", placeholder="Type a word or phrase...").strip()
    with c4:
        limit = st.selectbox("Rows to display", [25, 50, 100, 200], index=1)

    with st.spinner("Filtering reviews..."):
        mask = np.ones(len(df), dtype=bool)
        if has_sent and sent_sel != "All":
            mask &= (df["Sentiment"] == sent_sel).to_numpy()
        if has_rating and rating_sel != "All":
            mask &= (df["Rating"] == rating_sel).to_numpy()
        if query:
            mask &= df["Review"].str.contains(query, case=False, regex=False, na=False).to_numpy()
        positions = np.flatnonzero(mask)

    total = len(positions)
    shown_pos = positions[:limit]
    st.markdown(f"**{fmt_int(total)}** matching records · showing {fmt_int(len(shown_pos))}")
    if total == 0:
        empty_state("No reviews match these filters.", "Try clearing the search phrase or choosing 'All'.")
        return

    sub = df.iloc[shown_pos]
    table = pd.DataFrame({"Row": shown_pos + 1})
    snippet = sub["Review"].reset_index(drop=True)
    table["Review"] = snippet.where(snippet.str.len() <= 160, snippet.str.slice(0, 160) + "…")
    if has_rating:
        table["Rating"] = sub["Rating"].to_numpy()
    if "Product" in sub.columns:
        table["Product"] = sub["Product"].to_numpy()
    if has_sent:
        table["Dataset sentiment"] = sub["Sentiment"].astype(str).replace("nan", "—").to_numpy()
    st.dataframe(table, hide_index=True)

    intro("Inspect a review", "Pick a row from the table above to read it in full.")
    choice = st.selectbox(
        "Select a review",
        options=list(range(len(shown_pos))),
        format_func=lambda i: f"Row {int(shown_pos[i]) + 1:,} — {str(snippet.iloc[i])[:70]}",
    )
    record = sub.iloc[choice]
    chips = ""
    if has_rating and pd.notna(record["Rating"]):
        chips += f'<span class="riq-chip">Rating: {esc(record["Rating"])}</span>'
    if has_sent and pd.notna(record["Sentiment"]):
        chips += (f'<span class="riq-chip" style="color:{SENT_COLOR[str(record["Sentiment"])]};">'
                  f'Dataset sentiment (original label): {esc(record["Sentiment"])}</span>')
    if "Product" in sub.columns:
        chips += f'<span class="riq-chip">Product: {esc(record["Product"])}</span>'
    if "Summary" in sub.columns and str(record["Summary"]).strip():
        render(f'<div class="riq-eyebrow">SUMMARY</div><div class="riq-note" style="margin-bottom:10px;">{esc(record["Summary"])}</div>')
    if chips:
        render(f'<div style="margin-bottom:12px;">{chips}</div>')
    render(f'<div class="riq-review-box">{esc(record["Review"])}</div>')

    st.write("")
    if styled_button("Run the model on this review", "predict_selected"):
        try:
            with st.spinner("Analyzing sentiment..."):
                res = predict_text(str(record["Review"]))
            color = SENT_COLOR[res["label"]]
            render(f'<div class="riq-result" style="border-left-color:{color};">'
                   f'<div class="lbl">NEW MODEL PREDICTION (not the dataset label)</div>'
                   f'<div class="val" style="color:{color};">{SENT_ICON[res["label"]]} {esc(res["label"])}</div></div>')
            st.caption(f"Label decoding: {res['label_note']}.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


def page_guide() -> None:
    hero("Project Guide")
    intro("Project objective",
          "ReviewIQ classifies e-commerce product reviews into Negative, Neutral, and Positive "
          "categories using Natural Language Processing and machine learning.")

    intro("The four-stage pipeline", "From raw text to a sentiment label.")
    steps = [
        ("📝", "Review Input", "A customer review is entered as free text."),
        ("🧹", "Text Cleaning", "The review is normalised with the same cleaning function used during training."),
        ("🔢", "TF-IDF Feature Extraction", "The saved TF-IDF vectorizer turns the cleaned text into weighted numeric features."),
        ("🎯", "Linear SVM Prediction", "The trained Linear SVM assigns Negative, Neutral or Positive."),
    ]
    render('<div class="riq-flow">' + "".join(
        f'<div class="riq-step"><span class="num">{i}</span><span class="ico">{ico}</span>'
        f'<div class="ttl">{esc(t)}</div><div class="txt">{esc(d)}</div></div>'
        for i, (ico, t, d) in enumerate(steps, start=1)) + "</div>")

    art = artifacts()
    ngram = getattr(art["vectorizer"], "ngram_range", None) if art["vectorizer"] is not None else None
    ngram_txt = (f"TF-IDF n-gram range: {tuple(ngram)}" if ngram is not None
                 else "TF-IDF n-gram range: unavailable (vectorizer not loaded)")
    intro("Technology stack", "Only what this project actually uses.")
    render(f"""
    <div class="riq-stack">
      <div class="box"><div class="riq-h3">Application</div><ul><li>Python</li><li>Streamlit</li><li>Pandas</li></ul></div>
      <div class="box"><div class="riq-h3">Text processing</div><ul><li>Text cleaning</li><li>TF-IDF vectorization</li><li>{esc(ngram_txt)}</li></ul></div>
      <div class="box"><div class="riq-h3">Machine learning</div><ul><li>Logistic Regression</li><li>Naive Bayes</li><li>Linear SVM (main model)</li></ul></div>
    </div>
    """)

    intro("Artifact status", "Which project files the app can find right now.")
    ds = get_dataset()
    items = [
        ("models/svm_model.joblib", SVM_PATH.exists()),
        ("models/tfidf_vectorizer.joblib", VECTORIZER_PATH.exists()),
        ("results/model_comparison.csv", METRICS_PATH.exists()),
        (f"dataset/{ds['path']}" if ds["df"] is not None else "dataset CSV (recognized review data)", ds["df"] is not None),
    ]
    rows = "".join(
        f'<tr><td>{esc(name)}</td><td>' +
        ('<span class="riq-ok">✓ Available</span>' if ok else '<span class="riq-miss">✕ Missing</span>') +
        "</td></tr>"
        for name, ok in items
    )
    render(f'<table class="riq-status">{rows}</table>')
    _, src, is_project = cleaner()
    st.caption(f"Cleaning function in use: {src}.")


# --------------------------------------------------------------------------
# Sidebar + routing
# --------------------------------------------------------------------------
def sidebar() -> None:
    with st.sidebar:
        render("""
        <div class="riq-brand">
          <div class="riq-brand-mark">R</div>
          <div><div class="riq-brand-title">ReviewIQ</div>
          <div class="riq-brand-sub">CUSTOMER INTELLIGENCE</div></div>
        </div>
        <div class="riq-nav-head">WORKSPACE</div>
        """)
        for name, icon in NAV:
            active = st.session_state.page == name
            styled_button(name, f"nav_{name}", icon=icon, kind="primary" if active else "secondary",
                          on_click=go, args=(name,))
        render("""
        <div class="riq-sideinfo">
          <span class="tag">3-CLASS SENTIMENT</span>
          <div class="line">Natural Language Processing</div>
          <div class="line">TF-IDF feature extraction</div>
          <div class="line">Linear SVM classification</div>
          <div class="line">Python + Streamlit</div>
        </div>
        """)


PAGES = {
    "Overview": page_overview,
    "Analyze Reviews": page_analyze,
    "Model Studio": page_models,
    "Review Explorer": page_explorer,
    "Project Guide": page_guide,
}


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    if "page" not in st.session_state or st.session_state.page not in PAGES:
        st.session_state.page = "Overview"
    if "review_text" not in st.session_state:
        st.session_state.review_text = ""
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("result_error", None)

    sidebar()
    PAGES[st.session_state.page]()


main()