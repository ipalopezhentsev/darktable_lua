"""Learn the per-frame taste RESIDUAL (spec 03, Phase 2 — milestone 2: the
analytical-only ridge FLOOR). Loads `residual_dataset.json` (from
`build_residual_dataset.py`) and, per target param, trains a regularized Ridge to
predict `GT - analytical` from analytical-only features, validated
LEAVE-ONE-ROLL-OUT (whole roll held out — frames within a roll correlate, so
random CV would leak; same discipline as the constants cross-validation).

The honest question: does the ridge beat the **analytical baseline** (= predict
ZERO residual) on rolls it never saw? If yes, image metrics carry some of the
taste; if not (spec's hypothesis — CLAUDE.md found ~0 corr of gamma with pixel
metrics), that itself is the result: semantic LLM labels are required (milestone 3).
This is RECORD-ONLY / a measurement — it deploys nothing.

Guardrails (per spec): heavy regularization with alpha chosen by NESTED LORO on the
training rolls only (no leak); the predicted residual is CLIPPED to the training
residual range (bounded output); the model falls back to analytical (zero) where it
has no signal.

Run:  conda run -n autocrop python auto_negadoctor/tests/learn_residual.py
      [--data residual_dataset.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
except Exception as e:                                 # pragma: no cover
    print(f"scikit-learn required ({e}); it is in environment.yml.")
    raise

TESTS_DIR = Path(__file__).resolve().parent
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)     # ridge strength grid
MIN_ROLLS = 2                                         # need >=2 rolls for LORO


def _xy(rows, target, feature_names):
    """Feature matrix X + target vector y over the rows that carry `target`."""
    sel = [r for r in rows if target in r["targets"]]
    X = np.array([[r["features"][f] for f in feature_names] for r in sel],
                 dtype=float)
    y = np.array([r["targets"][target] for r in sel], dtype=float)
    rolls = [r["roll"] for r in sel]
    return X, y, rolls


def _fit_predict(Xtr, ytr, Xte, alpha):
    """Standardize on TRAIN only, fit Ridge, predict — with the predicted residual
    CLIPPED to the train residual range (bounded output guardrail)."""
    sc = StandardScaler().fit(Xtr)
    model = Ridge(alpha=alpha).fit(sc.transform(Xtr), ytr)
    pred = model.predict(sc.transform(Xte))
    return np.clip(pred, ytr.min(), ytr.max()), model, sc


def _pick_alpha(Xtr, ytr, rolls_tr):
    """Choose alpha by NESTED leave-one-roll-out over the TRAINING rolls only (so
    the outer held-out roll never informs alpha). Falls back to a strong default
    when <2 training rolls. Returns (alpha, inner_mae)."""
    uniq = sorted(set(rolls_tr))
    if len(uniq) < 2:
        return 100.0, None
    rolls_tr = np.array(rolls_tr)
    best = (None, np.inf)
    for a in ALPHAS:
        errs = []
        for h in uniq:
            tr, te = rolls_tr != h, rolls_tr == h
            if tr.sum() == 0 or te.sum() == 0:
                continue
            pred, _, _ = _fit_predict(Xtr[tr], ytr[tr], Xtr[te], a)
            errs.append(np.abs(pred - ytr[te]))
        if errs:
            mae = float(np.concatenate(errs).mean())
            if mae < best[1]:
                best = (a, mae)
    return best if best[0] is not None else (100.0, None)


def evaluate_target(rows, target, feature_names):
    """Outer leave-one-roll-out for one target. Returns a result dict or None if
    too few rolls carry the target."""
    X, y, rolls = _xy(rows, target, feature_names)
    uniq = sorted(set(rolls))
    if len(uniq) < MIN_ROLLS or len(y) < 8:
        return None
    rolls_arr = np.array(rolls)

    pred_all, mean_all, true_all, fold_rolls, alphas = [], [], [], [], []
    coef_sum = np.zeros(len(feature_names))
    for h in uniq:
        tr, te = rolls_arr != h, rolls_arr == h
        if tr.sum() == 0 or te.sum() == 0:
            continue
        alpha, _ = _pick_alpha(X[tr], y[tr], rolls_arr[tr].tolist())
        pred, model, _ = _fit_predict(X[tr], y[tr], X[te], alpha)
        # RECENTER baseline: predict the TRAIN rolls' mean residual (a pure
        # constant offset, NO features) for the held-out roll. This is what a
        # re-centered preset CONSTANT would achieve — the control that tells
        # whether ridge's edge over analytical is just "the default is off" vs
        # genuine per-frame prediction.
        tr_mean = float(y[tr].mean())
        pred_all.append(pred)
        mean_all.append(np.full(int(te.sum()), tr_mean))
        true_all.append(y[te])
        fold_rolls.append((h, float(np.abs(y[te]).mean()),
                           float(np.abs(np.full(int(te.sum()), tr_mean)
                                        - y[te]).mean()),
                           float(np.abs(pred - y[te]).mean())))
        alphas.append(alpha)
        coef_sum += np.abs(model.coef_)

    pred_all = np.concatenate(pred_all)
    mean_all = np.concatenate(mean_all)
    true_all = np.concatenate(true_all)

    def _mae(p):
        return float(np.abs(p - true_all).mean())

    def _rmse(p):
        return float(np.sqrt(np.mean((p - true_all) ** 2)))

    zero = np.zeros_like(true_all)
    base_mae, base_rmse = _mae(zero), _rmse(zero)             # analytical (predict 0)
    mean_mae, mean_rmse = _mae(mean_all), _rmse(mean_all)     # recentered constant
    ridge_mae, ridge_rmse = _mae(pred_all), _rmse(pred_all)   # ridge (features)
    skill = 1.0 - ridge_rmse / base_rmse if base_rmse > 0 else 0.0
    recenter_skill = 1.0 - mean_rmse / base_rmse if base_rmse > 0 else 0.0
    ridge_vs_mean = 1.0 - ridge_rmse / mean_rmse if mean_rmse > 0 else 0.0

    # Classify the source of any gain (the whole point of the recenter control).
    if mean_mae < base_mae * 0.90:
        if ridge_mae < mean_mae * 0.95:
            verdict = "FEATURES help (per-frame signal beyond a constant)"
        else:
            verdict = "CONSTANT mis-centered (recenter the preset; features add ~0)"
    elif ridge_mae < base_mae * 0.95:
        verdict = "FEATURES help (per-frame signal)"
    else:
        verdict = "NO signal (neither recenter nor features beat analytical)"

    coef_rank = sorted(zip(feature_names, (coef_sum / max(1, len(uniq))).tolist()),
                       key=lambda t: -abs(t[1]))
    return {
        "target": target, "n_frames": int(len(true_all)),
        "n_rolls": len(uniq), "rolls": uniq,
        "mean_residual": float(true_all.mean()),   # the recenter amount (GT - analytical)
        "baseline_mae": base_mae, "recenter_mae": mean_mae, "ridge_mae": ridge_mae,
        "baseline_rmse": base_rmse, "recenter_rmse": mean_rmse,
        "ridge_rmse": ridge_rmse,
        "skill_rmse": skill, "recenter_skill_rmse": recenter_skill,
        "ridge_vs_recenter_skill": ridge_vs_mean,
        "verdict": verdict,
        "beats_analytical": ridge_mae < base_mae,
        # "features help" = ridge beats BOTH analytical AND the recenter control,
        # i.e. genuine per-frame signal (not just a mis-centered constant).
        "features_help": verdict.startswith("FEATURES"),
        "alphas_per_fold": alphas,
        "per_fold": [{"held_out": h, "baseline_mae": b, "recenter_mae": m,
                      "ridge_mae": r} for h, b, m, r in fold_rolls],
        "top_features": [{"feature": f, "abs_coef": c} for f, c in coef_rank[:8]],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(TESTS_DIR / "residual_dataset.json"))
    args = ap.parse_args()
    path = Path(args.data)
    if not path.is_file():
        print(f"{path} not found — run build_residual_dataset.py first.")
        return 1
    data = json.loads(path.read_text())
    rows, feats = data["rows"], data["feature_names"]
    print(f"Loaded {len(rows)} rows, {len(feats)} features, rolls "
          f"{', '.join(data['rolls'])} (built {data.get('created')}).")
    print("Leave-one-roll-out residual learning (ridge, analytical-only features).\n")

    results = []
    for target in data["target_names"]:
        res = evaluate_target(rows, target, feats)
        if res is None:
            print(f"== {target}: too few rolls/frames carry this target — skip.")
            continue
        results.append(res)
        print(f"== {target}  ({res['n_frames']} frames, {res['n_rolls']} rolls)  "
              f"mean GT-analytical residual = {res['mean_residual']:+.4f}")
        print(f"   held-out MAE: analytical(0) {res['baseline_mae']:.4f}  ->  "
              f"recenter {res['recenter_mae']:.4f}  ->  ridge {res['ridge_mae']:.4f}")
        print(f"   RMSE skill vs analytical: recenter {res['recenter_skill_rmse']:+.3f}"
              f" | ridge {res['skill_rmse']:+.3f}   (ridge vs recenter "
              f"{res['ridge_vs_recenter_skill']:+.3f})")
        print(f"   -> {res['verdict']}")
        print("   per held-out roll (analytical -> recenter -> ridge MAE):")
        for f in res["per_fold"]:
            print(f"     {f['held_out']}: {f['baseline_mae']:.4f} -> "
                  f"{f['recenter_mae']:.4f} -> {f['ridge_mae']:.4f}")
        print("   top features (|coef|, standardized): "
              + ", ".join(f"{t['feature']}={t['abs_coef']:.3f}"
                          for t in res["top_features"][:5]))
        print()

    if results:
        feats_help = [r for r in results if r["verdict"].startswith("FEATURES")]
        recenter = [r for r in results if r["verdict"].startswith("CONSTANT")]
        print("=" * 64)
        print("VERDICT (milestone-2 floor):")
        print(f"  features add per-frame lift beyond a constant: "
              f"{len(feats_help)}/{len(results)} target(s)"
              + (f" [{', '.join(r['target'] for r in feats_help)}]"
                 if feats_help else ""))
        if recenter:
            print("  CONSTANT mis-centered (fix in the constants/LORO track, not a "
                  "learned model):")
            for r in recenter:
                print(f"    - {r['target']}: GT sits {r['mean_residual']:+.4f} from "
                      f"analytical on average (recenter skill "
                      f"{r['recenter_skill_rmse']:+.3f})")
        if not feats_help:
            print("  -> analytical image metrics do NOT predict per-frame taste on")
            print("     unseen rolls (spec's hypothesis confirmed). Any apparent win")
            print("     is a mis-centered CONSTANT. Next: fold those into the constants")
            print("     track, and pursue milestone 3 (gemma3:12b semantic labels —")
            print("     needs labeling the other 3 rolls) for the per-frame residual.")
        else:
            print("  -> validate the picture-EMD effect on held-out rolls before")
            print("     wiring a 'learned' variant.")
        out = TESTS_DIR / "residual_learning_report.json"
        out.write_text(json.dumps({"data": str(path), "results": results}, indent=2))
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
