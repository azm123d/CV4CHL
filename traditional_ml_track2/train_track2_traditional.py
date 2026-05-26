import argparse
import os
from collections import Counter

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from track2_features import ID_TO_LABEL, build_patient_examples, build_side_examples, read_pkl
from track2_evgs_features import Track1EVGSExtractor, build_patient_evgs_examples, build_side_evgs_examples


ALPHA_GRID = [0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0, 1.25]
SIDE_FUSION_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
GATE_DECIMALS = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Traditional ML model selection for CV4CHL Track 2.")
    parser.add_argument("--train-pkl", default="dataset/train_dataset_track2_all.pkl")
    parser.add_argument("--out", default="traditional_ml_track2/track2_traditional_model.joblib")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--track1-config", default="configs/gait/MB_ft_gait_track1.yaml")
    parser.add_argument("--track1-model", default="checkpoint/gait1/best.pth")
    parser.add_argument("--no-evgs", action="store_true", help="Disable Track 1 EVGS prediction features.")
    return parser.parse_args()


def balanced_pipeline(clf, scale=True):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


def make_candidates(seed: int):
    candidates = []

    for c in [0.3, 1.0, 3.0, 10.0]:
        candidates.append((
            f"seq_svm_c{c}",
            "sequence",
            balanced_pipeline(SVC(C=c, gamma="scale", probability=True, class_weight="balanced", random_state=seed)),
        ))
    for c in [0.1, 0.3, 1.0, 3.0]:
        candidates.append((
            f"seq_logreg_c{c}",
            "sequence",
            balanced_pipeline(LogisticRegression(C=c, max_iter=5000, class_weight="balanced", random_state=seed)),
        ))
    for k in [1, 3, 5, 7, 11]:
        candidates.append((
            f"seq_knn{k}",
            "sequence",
            balanced_pipeline(KNeighborsClassifier(n_neighbors=k, weights="distance")),
        ))
    candidates.extend([
        ("seq_lda_shrinkage", "sequence", balanced_pipeline(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))),
        ("seq_gaussian_nb", "sequence", balanced_pipeline(GaussianNB(var_smoothing=1e-8))),
    ])
    for max_features in ["sqrt", "log2"]:
        for min_leaf in [1, 2, 4]:
            candidates.append((
                f"seq_extra_trees_{max_features}_leaf{min_leaf}",
                "sequence",
                balanced_pipeline(ExtraTreesClassifier(
                    n_estimators=700,
                    max_features=max_features,
                    min_samples_leaf=min_leaf,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ), scale=False),
            ))
    for min_leaf in [1, 2, 4]:
        candidates.append((
            f"seq_random_forest_leaf{min_leaf}",
            "sequence",
            balanced_pipeline(RandomForestClassifier(
                n_estimators=600,
                max_features="sqrt",
                min_samples_leaf=min_leaf,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ), scale=False),
        ))

    for c in [1.0, 3.0, 10.0, 30.0]:
        candidates.append((
            f"patient_svm_c{c}",
            "patient",
            balanced_pipeline(SVC(C=c, gamma="scale", probability=True, class_weight="balanced", random_state=seed)),
        ))
    for c in [0.1, 0.3, 1.0]:
        candidates.append((
            f"patient_logreg_c{c}",
            "patient",
            balanced_pipeline(LogisticRegression(C=c, max_iter=5000, class_weight="balanced", random_state=seed)),
        ))
    candidates.extend([
        ("patient_lda_shrinkage", "patient", balanced_pipeline(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))),
        ("patient_extra_trees", "patient", balanced_pipeline(ExtraTreesClassifier(
            n_estimators=700,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ), scale=False)),
        ("patient_random_forest", "patient", balanced_pipeline(RandomForestClassifier(
            n_estimators=700,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        ), scale=False)),
    ])

    return candidates


def predict_proba_5(model, x):
    raw_prob = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)
    aligned = np.zeros((len(x), 5), dtype=np.float32)
    aligned[:, classes] = raw_prob
    return aligned


def prior_weights_from_counts(class_counts, alpha: float):
    counts = np.asarray([class_counts.get(i, 0) for i in range(5)], dtype=np.float32)
    counts = np.maximum(counts, 1.0)
    weights = (counts.max() / counts) ** alpha
    return weights / weights.mean()


def apply_prior_adjustment(probs, class_counts, alpha: float):
    weights = prior_weights_from_counts(class_counts, alpha)
    adjusted = probs * weights[None, :]
    return adjusted / np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-8)


def rounded_gate_value(value: float) -> float:
    return float(np.round(value, GATE_DECIMALS))


def apply_group_side_fusion(probs, groups, weight: float):
    if weight <= 0:
        return probs

    fused = probs.copy()
    for group_id in np.unique(groups):
        idx = np.where(groups == group_id)[0]
        if len(idx) < 2:
            continue
        group_mean = probs[idx].mean(axis=0, keepdims=True)
        fused[idx] = (1.0 - weight) * probs[idx] + weight * group_mean
    return fused


def metric_dict(y_true, probs):
    pred = probs.argmax(axis=1)
    acc = accuracy_score(y_true, pred)
    macro_f1 = f1_score(y_true, pred, labels=list(range(5)), average="macro", zero_division=0)
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "balanced_acc": balanced_accuracy_score(y_true, pred),
        "score": (acc + macro_f1) / 2.0,
    }


def evaluate_candidate(name, model, x, y, groups, seed, cv_splits):
    unique_groups = np.unique(groups)
    n_splits = max(2, min(cv_splits, len(unique_groups)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_prob = np.zeros((len(y), 5), dtype=np.float32)

    for train_idx, val_idx in splitter.split(x, y, groups):
        fold_model = clone(model)
        fold_model.fit(x[train_idx], y[train_idx])
        oof_prob[val_idx] = predict_proba_5(fold_model, x[val_idx])

    class_counts = dict(Counter(y))
    best = None
    for alpha in ALPHA_GRID:
        adjusted = apply_prior_adjustment(oof_prob, class_counts, alpha)
        for fusion_weight in SIDE_FUSION_GRID:
            fused = apply_group_side_fusion(adjusted, groups, fusion_weight)
            metrics = metric_dict(y, fused)
            if best is None or metrics["score"] > best["metrics"]["score"]:
                best = {
                    "name": name,
                    "alpha": alpha,
                    "side_fusion_weight": fusion_weight,
                    "metrics": metrics,
                    "oof_prob": fused,
                }
    return best


def print_final_report(y, probs, title):
    pred = probs.argmax(axis=1)
    metrics = metric_dict(y, probs)
    print(f"\n{title}")
    print(
        f"  acc={metrics['acc']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
        f"balanced_acc={metrics['balanced_acc']:.4f}, score={metrics['score']:.4f}"
    )
    print("  confusion matrix rows=true cols=pred labels=[WNL,type1,type2,type3,type4]")
    print(confusion_matrix(y, pred, labels=list(range(5))))
    print(classification_report(y, pred, labels=list(range(5)), target_names=[ID_TO_LABEL[i] for i in range(5)], zero_division=0))


def main():
    args = parse_args()
    np.random.seed(args.seed)

    dataset = read_pkl(args.train_pkl)
    seq_x, seq_y, seq_groups, _, seq_feature_names = build_side_examples(dataset)
    patient_x, patient_y, patient_groups, _, patient_feature_names = build_patient_examples(dataset)
    use_evgs = not args.no_evgs

    if use_evgs:
        print("Extracting Track 1 EVGS prediction features...")
        evgs_extractor = Track1EVGSExtractor(args.track1_config, args.track1_model)
        seq_evgs = build_side_evgs_examples(dataset, evgs_extractor)
        patient_evgs = build_patient_evgs_examples(dataset, evgs_extractor)
        wnl_evgs_totals = seq_evgs[seq_y == 0, 34]
        type1_evgs_totals = seq_evgs[seq_y == 1, 34]
        type3_evgs_totals = seq_evgs[seq_y == 3, 34]
        wnl_gate = {
            "enabled": len(wnl_evgs_totals) > 0,
            "threshold": rounded_gate_value(np.percentile(wnl_evgs_totals, 80)) if len(wnl_evgs_totals) else 0.0,
            "percentile": 80,
            "exempt_labels": [1, 2],
            "exempt_confidence": 0.8,
        }
        mild_type1_gate = {
            "enabled": len(type1_evgs_totals) > 0 and len(type3_evgs_totals) > 0,
            "evgs_threshold": rounded_gate_value(
                (
                    np.percentile(type1_evgs_totals, 50)
                    + np.percentile(type3_evgs_totals, 25)
                ) / 2.0
            ) if len(type1_evgs_totals) and len(type3_evgs_totals) else 0.0,
            "type1_probability_floor": 0.2,
            "from_labels": [3],
        }
        if wnl_gate["enabled"]:
            print(
                "  WNL EVGS gate threshold="
                f"{wnl_gate['threshold']:.1f} from train WNL p{wnl_gate['percentile']}"
            )
        if mild_type1_gate["enabled"]:
            print(
                "  Mild type1 gate threshold="
                f"{mild_type1_gate['evgs_threshold']:.1f} from train type1 p50/type3 p25"
            )
        seq_x = np.concatenate([seq_x, seq_evgs], axis=1)
        patient_x = np.concatenate([patient_x, patient_evgs], axis=1)
        seq_feature_names = seq_feature_names + [f"evgs_{i}" for i in range(seq_evgs.shape[1])]
        patient_feature_names = patient_feature_names + [f"patient_evgs_{i}" for i in range(patient_evgs.shape[1])]
    else:
        wnl_gate = {"enabled": False}
        mild_type1_gate = {"enabled": False}

    print(f"Loaded {args.train_pkl}")
    print(f"  sequences={len(dataset)}")
    print(f"  sequence side examples={len(seq_y)}, features={seq_x.shape[1]}, groups={len(np.unique(seq_groups))}")
    print(f"  patient side examples={len(patient_y)}, features={patient_x.shape[1]}, groups={len(np.unique(patient_groups))}")
    print(f"  use_evgs_features={use_evgs}")
    print("  sequence class counts:", {ID_TO_LABEL[k]: v for k, v in sorted(Counter(seq_y).items())})
    print("  patient class counts:", {ID_TO_LABEL[k]: v for k, v in sorted(Counter(patient_y).items())})

    results = []
    candidates = make_candidates(args.seed)
    print(f"\nSearching {len(candidates)} candidate models with group-aware CV...")
    for idx, (name, level, model) in enumerate(candidates, start=1):
        if level == "sequence":
            x, y, groups = seq_x, seq_y, seq_groups
        else:
            x, y, groups = patient_x, patient_y, patient_groups
        result = evaluate_candidate(name, model, x, y, groups, args.seed, args.cv_splits)
        result.update({"level": level, "model": model})
        results.append(result)
        m = result["metrics"]
        print(
            f"  [{idx:02d}/{len(candidates)}] {name:<32} level={level:<8} "
            f"alpha={result['alpha']:<4} fusion={result['side_fusion_weight']:<4} "
            f"score={m['score']:.4f} acc={m['acc']:.4f} macro_f1={m['macro_f1']:.4f}"
        )

    results.sort(key=lambda item: item["metrics"]["score"], reverse=True)
    print("\nTop candidates")
    for rank, item in enumerate(results[:8], start=1):
        m = item["metrics"]
        print(
            f"  {rank}. {item['name']} ({item['level']}): "
            f"score={m['score']:.4f}, acc={m['acc']:.4f}, macro_f1={m['macro_f1']:.4f}, "
            f"alpha={item['alpha']}, fusion={item['side_fusion_weight']}"
        )

    best = results[0]

    if best["level"] == "sequence":
        final_x, final_y, feature_names = seq_x, seq_y, seq_feature_names
    else:
        final_x, final_y, feature_names = patient_x, patient_y, patient_feature_names

    final_model = clone(best["model"])
    final_model.fit(final_x, final_y)

    patient_results = [item for item in results if item["level"] == "patient"]
    aux_patient = max(patient_results, key=lambda item: item["metrics"]["score"]) if patient_results else None
    aux_patient_model = None
    if aux_patient is not None:
        aux_patient_model = clone(aux_patient["model"])
        aux_patient_model.fit(patient_x, patient_y)

    bundle = {
        "model": final_model,
        "model_name": best["name"],
        "feature_level": best["level"],
        "prior_alpha": best["alpha"],
        "side_fusion_weight": best["side_fusion_weight"],
        "feature_names": feature_names,
        "id_to_label": ID_TO_LABEL,
        "class_counts": dict(Counter(final_y)),
        "use_evgs": use_evgs,
        "track1_config": args.track1_config,
        "track1_model": args.track1_model,
        "wnl_gate": wnl_gate,
        "mild_type1_gate": mild_type1_gate,
        "type2_over_type4_gate": {
            "enabled": aux_patient_model is not None,
            "aux_confidence_floor": 0.8,
        },
        "aux_patient_model": aux_patient_model,
        "aux_patient_model_name": aux_patient["name"] if aux_patient is not None else None,
        "aux_patient_prior_alpha": aux_patient["alpha"] if aux_patient is not None else 0.0,
        "aux_patient_side_fusion_weight": aux_patient["side_fusion_weight"] if aux_patient is not None else 0.0,
        "aux_patient_class_counts": dict(Counter(patient_y)),
        "cv_results": [
            {
                "name": item["name"],
                "level": item["level"],
                "alpha": item["alpha"],
                "side_fusion_weight": item["side_fusion_weight"],
                "metrics": item["metrics"],
            }
            for item in results
        ],
        "train_pkl": args.train_pkl,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"\nSaved best model to {args.out}")


if __name__ == "__main__":
    main()
