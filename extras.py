import argparse, os, textwrap, numpy as np, pandas as pd
from lab import data, experiments as ex, models

ap = argparse.ArgumentParser()
ap.add_argument("--essays", required=True); ap.add_argument("--source", default="hc3", choices=["hc3", "demo"])
ap.add_argument("--cap", type=int, default=800); ap.add_argument("--seeds", type=int, default=3)
a = ap.parse_args(); out = []
def say(*x):
    s = " ".join(map(str, x)); print(s); out.append(s)

df = data.load_hc3(a.cap) if a.source == "hc3" else data.load_demo()
ess = data.load_essays(a.essays); ess["words"] = ess.text.str.split().str.len()

if os.path.exists("results/matrix.csv"):
    say("== RQ1: in-domain vs unseen AUROC (mean over seeds, std of drop) ==")
    M = pd.read_csv("results/matrix.csv")
    for det in "ABL":
        r = pd.DataFrame([(g[g.train == g.test].auroc.mean(), g[g.train != g.test].auroc.mean())
                          for _, g in M[M.det == det].groupby("seed")], columns=["in_domain", "unseen"])
        r["drop"] = r.in_domain - r.unseen
        say(det, r.mean().round(3).to_dict(), "std of drop:", round(r["drop"].std(), 3))

def match(e, b):
    e = e.assign(bin=b)
    n = e.groupby(["bin", "group"], observed=True).size().unstack(fill_value=0).min(axis=1)
    p = [g.sample(int(n[k]), random_state=0) for (k, _), g in e.groupby(["bin", "group"], observed=True) if n[k] > 0]
    return pd.concat(p) if p else e.iloc[:0]
matched = match(ess, pd.cut(ess.words, [150, 200, 250, 300, 400, 500, 600], right=False))
if len(matched) == 0: matched = match(ess, pd.qcut(ess.words, 5, duplicates="drop"))

rng = np.random.RandomState(0)
def gap_ci(e):
    x, y = e[e.group == "non-native"].f.values, e[e.group == "native"].f.values
    d = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(2000)]
    return round(float(x.mean() - y.mean()), 3), np.percentile(d, [2.5, 97.5]).round(3).tolist()

say("\n== RQ2: share of human essays flagged AI (gap = non-native minus native, 95% CI) ==")
for det in "AB":
    f05, f01 = [], []
    for s in range(a.seeds):
        tr, te = ex.split(df, s); m = models.make(det, seed=s).fit(tr.text, tr.label)
        thr = np.quantile(m.predict_proba(list(te[te.label == 0].text))[:, 1], .99)   # 1% FPR on held-out HC3 human text
        p = m.predict_proba(list(ess.text))[:, 1]; f05.append(p > .5); f01.append(p > thr)
    for name, f in [("0.5 cutoff", f05), ("1%-FPR cutoff", f01)]:
        e = ess.assign(f=np.mean(f, axis=0))
        for sub, d in [("all", e), ("length-matched", e.loc[matched.index])]:
            say(det, name, sub, len(d), d.groupby("group").f.mean().round(3).to_dict(), "gap:", gap_ci(d))

say("\n== Detector A strongest words ==")
m = models.make("A").fit(df.text, df.label)
w = np.array(m[0].get_feature_names_out()); o = np.argsort(m[-1].coef_[0])
say("HUMAN-leaning:", textwrap.fill(", ".join(w[o[:20]]), 90)); say("AI-leaning:", textwrap.fill(", ".join(w[o[-20:]]), 90))
os.makedirs("results", exist_ok=True); open("results/extras.txt", "w").write("\n".join(out))
