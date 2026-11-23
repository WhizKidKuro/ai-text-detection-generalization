"""Train Detectors A, B, L on all HC3 domains and export them to model.js for the offline browser demo.
Run inside the project folder (Colab or local):  python export_model.py --source hc3 --cap 800"""
import argparse, base64, json, numpy as np
from lab import data, models
ap = argparse.ArgumentParser(); ap.add_argument("--source", default="hc3"); ap.add_argument("--cap", type=int, default=800)
ap.add_argument("--out", default="model.js"); ap.add_argument("--ref"); a = ap.parse_args()
df = data.load_hc3(a.cap) if a.source == "hc3" else data.load_demo()
X, y = list(df.text), df.label.values
A, B, L = (models.make(k, seed=0).fit(X, y) for k in "ABL")
b64 = lambda arr: base64.b64encode(arr.tobytes()).decode()
tf, lrA, rf, lrL = A.steps[0][1], A.steps[-1][1], B.steps[-1][1], L.steps[-1][1]
feat, thr, lef, rig, prob, offs = [], [], [], [], [], [0]
for est in rf.estimators_:
    t = est.tree_; leaf = t.children_left == -1; v = t.value[:, 0, :]; p = v[:, 1] / v.sum(1)
    feat.append(np.where(leaf, -1, t.feature).astype(np.int8)); thr.append(t.threshold.astype(np.float32))
    lef.append(np.where(leaf, 65535, t.children_left).astype(np.uint16)); rig.append(np.where(leaf, 65535, t.children_right).astype(np.uint16))
    prob.append(np.round(p * 255).astype(np.uint8)); offs.append(offs[-1] + t.node_count)
cat = np.concatenate; F = models.stylo(X)
M = {"meta": {"source": a.source, "texts": len(X), "cap": a.cap},
     "A": {"terms": sorted(tf.vocabulary_, key=tf.vocabulary_.get), "idf": [round(float(v), 5) for v in tf.idf_], "coef": [round(float(v), 5) for v in lrA.coef_[0]], "b": float(lrA.intercept_[0])},
     "B": {"feat": b64(cat(feat)), "thr": b64(cat(thr)), "left": b64(cat(lef)), "right": b64(cat(rig)), "prob": b64(cat(prob)), "offs": offs,
           "mean0": F[y == 0].mean(0).round(4).tolist(), "mean1": F[y == 1].mean(0).round(4).tolist()},
     "L": {"coef": float(lrL.coef_[0][0]), "b": float(lrL.intercept_[0])}}
open(a.out, "w").write("window.DETECTOR_MODEL=" + json.dumps(M, separators=(",", ":")) + ";")
if a.ref:
    S = ["Honestly I think the new bus route is a good idea. It saves me time in the morning but the stops are too far apart, and my friends agree.", "Public transportation plays a crucial role in modern urban planning. It reduces traffic congestion, lowers carbon emissions, and provides affordable mobility for residents.", data.load_demo().text.iloc[3], data.load_demo().text.iloc[8], "short text here"]
    json.dump([{"t": s, "A": float(A.predict_proba([data.clean(s)])[0, 1]), "B": float(B.predict_proba([data.clean(s)])[0, 1]), "L": float(L.predict_proba([data.clean(s)])[0, 1])} for s in S], open(a.ref, "w"))
print("saved", a.out, f"({len(X)} training texts, {len(offs) - 1} trees, {offs[-1]} nodes)")
