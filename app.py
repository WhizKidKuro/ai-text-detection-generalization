import os
import pandas as pd, streamlit as st
from lab import data, experiments as ex, plots
from lab.models import make, stylo, FEATURES

st.set_page_config(page_title="AI-Text Detector Lab", page_icon="🔎", layout="wide")
st.markdown("<style>.block-container{max-width:1100px;padding-top:2rem}h1{font-weight:700}</style>", unsafe_allow_html=True)
st.title("🔎 AI-Text Detector Lab")
st.caption("Do AI-text detectors survive new domains, and do they unfairly flag non-native writers?")

with st.sidebar:
    SAVED = os.path.exists("results/matrix.csv")
    mode = st.radio("Results", ["Saved results", "Run new experiments"]) if SAVED else "Run new experiments"
    st.header("Data")
    src = st.radio("Source", ["Demo (synthetic, instant)", "HC3 (real, needs internet)"])
    up = st.file_uploader("Essays CSV (text, group)", type="csv", help="group = native / non-native; human-written only")
    seeds = st.slider("Random seeds", 1, 5, 3)
    cap = st.slider("Max questions per domain", 100, 800, 300, 50)
    go = st.button("▶ Run experiments", type="primary", use_container_width=True)
    st.info("Demo data is synthetic: it only checks that the pipeline works. Use HC3 for real findings.")

demo = src.startswith("Demo")

def load_saved():
    out = {}
    for k in ["matrix", "ablation", "fairness", "errors"]:
        try: out[k] = pd.read_csv(f"results/{k}.csv")
        except Exception: out[k] = pd.DataFrame()
    return out

@st.cache_data(show_spinner="Loading data…")
def load(demo, cap): return data.load_demo() if demo else data.load_hc3(cap)

@st.cache_data(show_spinner="Training and testing detectors…")
def run(demo, cap, seeds, ess):
    df = load(demo, cap)
    if ess is None and demo: ess = data.load_demo_essays()
    return ex.run_all(df, data.load_essays(ess) if ess is not None else None, tuple(range(seeds)))

@st.cache_resource(show_spinner="Training detectors for live demo…")
def live(demo, cap):
    df = load(demo, cap); return {d: make(d).fit(df.text, df.label) for d in "AB"}

t1, t2, t3, t4, t5 = st.tabs(["✍️ Try it", "🌐 Transfer (RQ1)", "⚖️ Fairness (RQ2)", "🧪 Ablation & errors", "ℹ️ About"])

with t1:
    txt = st.text_area("Paste any text (30+ words works best)", height=180, placeholder="Paste an essay, answer, or paragraph…")
    if st.button("Analyze") and txt.strip():
        m = live(demo, cap); c = st.columns(2)
        for col, d, n in zip(c, "AB", ["Detector A · word patterns", "Detector B · writing style"]):
            p = m[d].predict_proba([txt])[0, 1]
            col.metric(n, f"{p:.0%} AI-like"); col.progress(float(p))
        st.subheader("Style fingerprint")
        st.dataframe(pd.DataFrame({"feature": FEATURES, "value": stylo([txt])[0].round(3)}), hide_index=True)
        st.warning("A score is a statistical guess, not proof. Never use it as the only evidence against a person.")

if mode == "Saved results":
    res = load_saved()
else:
    res = run(demo, cap, seeds, pd.read_csv(up) if up else None) if go or "ran" in st.session_state else None
    if go: st.session_state["ran"] = True
for tab in (t2, t3, t4):
    if res is None:
        with tab: st.info("Click **Run experiments** in the sidebar.")
if res is not None:
    M = res["matrix"]
    with t2:
        det = st.radio("Detector", ["A", "B", "L"], horizontal=True, format_func=lambda d: plots.NAMES[d])
        metric = st.selectbox("Metric", ["auroc", "acc", "f1"])
        st.pyplot(plots.heat(M, det, metric))
        S = M[M.det == det].groupby(["train", "test"])[metric].mean().reset_index()
        diag, off = S[S.train == S.test][metric].mean(), S[S.train != S.test][metric].mean()
        a, b, c = st.columns(3); a.metric("In-domain (diagonal)", f"{diag:.3f}")
        b.metric("Unseen domains", f"{off:.3f}"); c.metric("Average drop", f"{diag - off:.3f}")
        st.dataframe(ex.summary(M).query("det == @det").round(3), hide_index=True)
    with t3:
        if len(res["fairness"]):
            st.pyplot(plots.fpr_bar(res["fairness"])); st.dataframe(res["fairness"].round(3), hide_index=True)
            st.caption("Non-overlapping intervals support H2. The gap between groups matters more than the absolute level (genre shift hits both).")
        else: st.info("Upload an essays CSV to run RQ2.")
    with t4:
        st.subheader("E4 · What does Detector B rely on?")
        st.dataframe(res["ablation"].groupby("removed")[["auroc", "acc"]].mean().round(3))
        st.subheader("Error analysis")
        st.dataframe(res["errors"], hide_index=True)
with t5:
    st.markdown("Two classic detectors (**A** TF-IDF + logistic regression, **B** stylometric features + random forest) are trained on one domain and tested on others (RQ1), then run on human essays by native vs non-native writers (RQ2). See `docs/PROJECT_REPORT.md`.")
