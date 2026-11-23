
import argparse, os
from implementation import data, experiments as ex, plots

ap = argparse.ArgumentParser()
ap.add_argument("--source", default="demo", choices=["demo", "hc3"])
ap.add_argument("--essays", help="CSV with columns text,group (native / non-native)")
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--cap", type=int, default=800)
ap.add_argument("--out", default="results")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
df = data.load_demo() if a.source == "demo" else data.load_hc3(a.cap)
ess = data.load_essays(a.essays) if a.essays else (data.load_demo_essays() if a.source == "demo" else None)
res = ex.run_all(df, ess, tuple(range(a.seeds)), progress=lambda p: print(f"{p:.0%} done"))
for k, v in res.items(): v.to_csv(f"{a.out}/{k}.csv", index=False)
ex.summary(res["matrix"]).to_csv(f"{a.out}/matrix_summary.csv", index=False)
for det in "AB": plots.heat(res["matrix"], det).savefig(f"{a.out}/heatmap_{det}.png", dpi=150)
if len(res["fairness"]): plots.fpr_bar(res["fairness"]).savefig(f"{a.out}/fpr_bar.png", dpi=150)
print(res["fairness"].round(3).to_string() if len(res["fairness"]) else "no essays given: RQ2 skipped")
print(f"Saved to {a.out}/")
