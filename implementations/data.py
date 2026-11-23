"""Data loaders. Real: HC3 (human vs ChatGPT) + essay CSV."""
import json, random
import re
import pandas as pd

HC3_DOMAINS = ["reddit_eli5", "finance", "medicine", "open_qa", "wiki_csai"]
MIN_WORDS = 30

def clean(t):
    """Remove dataset formatting artifacts that could act as shortcuts: literal \\n sequences in ChatGPT
    answers, and spaces before punctuation in human answers; collapse whitespace."""
    t = str(t).replace("\\n", " ")
    t = re.sub(r"\burl_\d+\b", " ", t)
    t = re.sub(r"\s+([,.;:!?)\]])", r"\1", t)
    t = re.sub(r"([(\[])\s+", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()

def _ok(t):
    return isinstance(t, str) and len(t.split()) >= MIN_WORDS

def load_hc3(cap=800, seed=0):
    """One human + one ChatGPT answer per question, so every domain is balanced.
    Rows share a qid, which lets us split by question (no leakage)."""
    from huggingface_hub import hf_hub_download
    parts = []
    for d in HC3_DOMAINS:
        path = hf_hub_download("Hello-SimpleAI/HC3", f"{d}.jsonl", repo_type="dataset")
        rows = []
        for i, line in enumerate(open(path, encoding="utf-8")):
            r = json.loads(line)
            h = [t for t in map(clean, r["human_answers"]) if _ok(t)]
            a = [t for t in map(clean, r["chatgpt_answers"]) if _ok(t)]
            if h and a:
                rows += [(d, f"{d}-{i}", h[0], 0), (d, f"{d}-{i}", a[0], 1)]
        g = pd.DataFrame(rows, columns=["domain", "qid", "text", "label"])
        keep = g.qid.drop_duplicates().sample(min(cap, g.qid.nunique()), random_state=seed)
        parts.append(g[g.qid.isin(keep)])
    return pd.concat(parts, ignore_index=True)

def load_essays(path_or_df):
    """CSV/DataFrame with columns: text, group ('native' or 'non-native'). Human-written only."""
    df = path_or_df if isinstance(path_or_df, pd.DataFrame) else pd.read_csv(path_or_df)
    df = df[["text", "group"]].dropna()
    df["group"] = df.group.str.lower().str.strip()
    df["text"] = df.text.map(clean)
    return df[df.text.map(_ok)].reset_index(drop=True)

# ---------- synthetic demo (NOT real findings) ----------
_POOLS = {"cooking": "salt pan heat oil sauce boil recipe oven flour".split(),
          "sports": "team game score coach season player match goal train".split(),
          "health": "sleep doctor diet stress pain habit body rest dose".split()}
_HUM = "honestly lol i think maybe kinda just so well anyway yeah the a it".split()
_AI = "furthermore overall essential comprehensive various factors approach significant important".split()

def _txt(R, pool, ai):
    out = []
    for _ in range(R.randint(4, 6)):
        k = R.randint(14, 20) if ai else R.randint(8, 18)
        w = [R.choice(pool if R.random() < .5 else (_AI if ai else _HUM)) for _ in range(k)]
        if ai: w[0] = R.choice(["Furthermore,", "Moreover,", "Additionally,"])
        out.append(" ".join(w).capitalize() + ".")
    return " ".join(out)

def load_demo(n=250, seed=0):
    R = random.Random(seed); rows = []
    for d, pool in _POOLS.items():
        for i in range(n):
            rows += [(d, f"{d}-{i}", _txt(R, pool, False), 0), (d, f"{d}-{i}", _txt(R, pool, True), 1)]
    return pd.DataFrame(rows, columns=["domain", "qid", "text", "label"])

def load_demo_essays(n=120, seed=1):
    R = random.Random(seed); pool = _POOLS["cooking"]; rows = []
    for _ in range(n):
        rows.append((_txt(R, pool, False), "native"))
        rows.append((" ".join(_txt(R, pool, False).split()[:40]) + " " + _txt(R, pool, False), "non-native"))
    return pd.DataFrame(rows, columns=["text", "group"])
