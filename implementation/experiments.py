"""E1 in-domain, E2 cross-domain, E3 writer-group fairness, E4 ablation, plus error analysis."""
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from .models import make, GROUPS

def split(df, seed, test=0.2):
    """Per domain, hold out 20% of QUESTIONS so a question never sits in both train and test."""
    tr, te = [], []
    for _, g in df.groupby("domain"):
        q = np.array(g.qid.unique()); np.random.RandomState(seed).shuffle(q); k = max(1, int(len(q) * test))
        te.append(g[g.qid.isin(q[:k])]); tr.append(g[~g.qid.isin(q[:k])])
    return pd.concat(tr), pd.concat(te)

def score(m, X, y):
    p = m.predict_proba(list(X))[:, 1]
    return roc_auc_score(y, p), accuracy_score(y, p > .5), f1_score(y, p > .5)

def boot(x, n=1000, seed=0):
    r = np.random.RandomState(seed)
    return np.percentile([r.choice(x, len(x)).mean() for _ in range(n)], [2.5, 97.5])

def run_all(df, essays=None, seeds=(0, 1, 2), progress=lambda *_: None):
    M, A, flags = [], [], {}
    doms = sorted(df.domain.unique())
    for si, s in enumerate(seeds):
        tr, te = split(df, s)
        for det in "ABL":
            for d in doms:
                g = tr[tr.domain == d]; m = make(det, seed=s).fit(g.text, g.label)
                for t in doms:
                    h = te[te.domain == t]; M.append((det, d, t, s, *score(m, h.text, h.label)))
        for gname, idx in [("none (full model)", ())] + list(GROUPS.items()):
            m = make("B", drop=idx, seed=s).fit(tr.text, tr.label)
            A.append((gname, s, *score(m, te.text, te.label)))
        if essays is not None:
            for det in "AB":
                m = make(det, seed=s).fit(tr.text, tr.label)
                flags.setdefault(det, []).append(m.predict_proba(list(essays.text))[:, 1] > .5)
        progress((si + 1) / len(seeds))
    M = pd.DataFrame(M, columns=["det", "train", "test", "seed", "auroc", "acc", "f1"])
    A = pd.DataFrame(A, columns=["removed", "seed", "auroc", "acc", "f1"])
    F = pd.DataFrame(); E = pd.DataFrame()
    if essays is not None:
        rows = []
        for det, fl in flags.items():
            fl = np.array(fl).T
            for grp in sorted(essays.group.unique()):
                x = fl[(essays.group == grp).values].mean(1); lo, hi = boot(x)
                rows.append((det, grp, x.mean(), lo, hi, len(x)))
        F = pd.DataFrame(rows, columns=["det", "group", "fpr", "lo", "hi", "n"])
        E = essays[flags["A"][0]].groupby("group").head(5).assign(kind="human essay called AI (detector A)")[["kind", "group", "text"]]
    d = M[(M.det == "A") & (M.train != M.test)].groupby(["train", "test"]).auroc.mean().idxmin()
    tr, te = split(df, seeds[0]); g = tr[tr.domain == d[0]]; h = te[te.domain == d[1]]
    m = make("A", seed=seeds[0]).fit(g.text, g.label); h = h.assign(p=m.predict_proba(list(h.text))[:, 1] > .5)
    err = pd.concat([h[(h.label == 0) & h.p].head(10).assign(kind=f"human called AI ({d[0]}→{d[1]})"),
                     h[(h.label == 1) & ~h.p].head(10).assign(kind=f"AI called human ({d[0]}→{d[1]})")])
    err = err.assign(group="")[["kind", "group", "text"]]
    E = pd.concat([E, err], ignore_index=True)
    return {"matrix": M, "ablation": A, "fairness": F, "errors": E}

def summary(M, cols=("auroc", "acc", "f1")):
    g = M.groupby(["det", "train", "test"])[list(cols)]
    return g.mean().join(g.std(), rsuffix="_std").reset_index()
