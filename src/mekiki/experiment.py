"""python -m mekiki.experiment -> results/report.md, results/results.json"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .estimators import Estimators
from .shifts import FEATURE_SHIFTS, label_shift

REPS = 5


def models(seed):
    return {"logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "MLP": make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=400, random_state=seed)),
            "random forest": RandomForestClassifier(100, random_state=seed)}


def main():
    out = Path("results"); out.mkdir(exist_ok=True)
    X, y = load_digits(return_X_y=True)
    rows = []   # dict per (model, family, severity, rep)
    for split_seed in range(3):
        rng = np.random.default_rng(split_seed)
        idx = rng.permutation(len(y))
        tr, va, te = idx[:900], idx[900:1200], idx[1200:]
        for mname, m in models(split_seed).items():
            m.fit(X[tr], y[tr])
            est = Estimators(m.predict_proba(X[va]), y[va])
            conds = [("no shift", 0, X[te], y[te])]
            for fam, (fn, sevs) in FEATURE_SHIFTS.items():
                for sev in sevs:
                    for r in range(REPS if fam in ("gaussian noise", "pixel dropout") else 1):
                        conds.append((fam, sev, fn(X[te], np.random.default_rng(1000 * split_seed + r), sev), y[te]))
            for r in range(REPS):
                Xs, ys = label_shift(X[te], y[te], np.random.default_rng(r + 7 * split_seed))
                conds.append(("label shift (90% digits 0-4)", 0.9, Xs, ys))
            for fam, sev, Xt, yt in conds:
                P = m.predict_proba(Xt)
                true = float((P.argmax(1) == yt).mean())
                for ename, e in est.estimate(P).items():
                    rows.append(dict(model=mname, family=fam, sev=sev, est=ename, true=true, pred=e, err=abs(e - true), seed=split_seed))
    (out / "results.json").write_text(json.dumps(rows))
    fams = ["no shift"] + list(FEATURE_SHIFTS) + ["label shift (90% digits 0-4)"]
    ests = list(dict.fromkeys(r["est"] for r in rows))
    L = ["# mekiki results\n",
         f"Digits (8x8): 900 train, 300 source-validation, 597 target images; 3 random splits; 3 models; shifts applied to the target images only. "
         "Estimators use only source validation data plus unlabeled target predictions.\n",
         "## Mean absolute error of the accuracy estimate (accuracy points), by shift family (all models, severities, repeats pooled)\n",
         "| shift family | true accuracy range | " + " | ".join(ests) + " |", "|---|---|" + "---|" * len(ests)]
    for f in fams:
        sub = [r for r in rows if r["family"] == f and r["est"] == ests[0]]
        lo, hi = min(r["true"] for r in sub), max(r["true"] for r in sub)
        L.append(f"| {f} | {lo:.0%}-{hi:.0%} | " + " | ".join(
            f"{100 * np.mean([r['err'] for r in rows if r['family'] == f and r['est'] == e]):.1f}" for e in ests) + " |")
    L.append("| **all shifts pooled** | | " + " | ".join(
        f"**{100 * np.mean([r['err'] for r in rows if r['est'] == e]):.1f}**" for e in ests) + " |")
    L.append("\n## Per model, mean absolute error over all conditions (accuracy points)\n")
    L.append("| model | " + " | ".join(ests) + " |"); L.append("|---|" + "---|" * len(ests))
    for mname in ["logistic regression", "MLP", "random forest"]:
        L.append(f"| {mname} | " + " | ".join(
            f"{100 * np.mean([r['err'] for r in rows if r['model'] == mname and r['est'] == e]):.1f}" for e in ests) + " |")
    L.append("\n## Do the estimates at least rank conditions correctly? (Spearman between estimated and true accuracy, per model)\n")
    L.append("| model | " + " | ".join(ests[1:]) + " |"); L.append("|---|" + "---|" * (len(ests) - 1))
    for mname in ["logistic regression", "MLP", "random forest"]:
        cells = []
        for e in ests[1:]:
            sub = [r for r in rows if r["model"] == mname and r["est"] == e]
            cells.append(f"{spearmanr([r['true'] for r in sub], [r['pred'] for r in sub]).statistic:.2f}")
        L.append(f"| {mname} | " + " | ".join(cells) + " |")
    L.append("\n## Worst single misses (largest |estimate - truth|)\n")
    worst = sorted([r for r in rows if r["est"] != ests[0]], key=lambda r: -r["err"])[:6]
    L.append("| estimator | model | shift | true acc | estimated |"); L.append("|---|---|---|---|---|")
    for r in worst:
        L.append(f"| {r['est']} | {r['model']} | {r['family']} {r['sev']} | {r['true']:.0%} | {r['pred']:.0%} |")
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
