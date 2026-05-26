import argparse
import glob
import os
import re

import joblib
import numpy as np

from track2_features import build_test_patient_aggregates, build_test_patient_features, read_pkl
from track2_evgs_features import Track1EVGSExtractor, build_test_patient_evgs, build_test_side_evgs


def parse_args():
    parser = argparse.ArgumentParser(description="Print Track 2 test predictions from the traditional ML model.")
    parser.add_argument("--model", default="traditional_ml_track2/track2_traditional_model.joblib")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--pattern", default="test_track2_*.pkl")
    parser.add_argument("--show-proba", action="store_true")
    parser.add_argument("--force-same-limb", action="store_true", help="Optional clinical prior: force both limbs to the same subtype.")
    return parser.parse_args()


def natural_key(path: str):
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", name)]


def predict_proba_5(model, x):
    raw_prob = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)
    aligned = np.zeros((len(x), 5), dtype=np.float32)
    aligned[:, classes] = raw_prob
    return aligned


def adjust_probabilities(probs, alpha, counts_dict):
    counts = np.asarray([counts_dict.get(i, counts_dict.get(np.int64(i), 1)) for i in range(5)], dtype=np.float32)
    counts = np.maximum(counts, 1.0)
    weights = (counts.max() / counts) ** alpha
    weights = weights / weights.mean()
    adjusted = probs * weights[None, :]
    return adjusted / np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-8)


def apply_prior_adjustment(bundle, probs):
    return adjust_probabilities(
        probs,
        float(bundle.get("prior_alpha", 0.0)),
        bundle.get("class_counts", {}),
    )


def apply_side_fusion_weight(probs_by_side, weight: float):
    if weight <= 0:
        return probs_by_side

    patient_prob = (probs_by_side[0] + probs_by_side[1]) / 2.0
    return {
        side: (1.0 - weight) * probs_by_side[side] + weight * patient_prob
        for side in (0, 1)
    }


def apply_side_fusion(bundle, probs_by_side):
    return apply_side_fusion_weight(probs_by_side, float(bundle.get("side_fusion_weight", 0.0)))


def side_probabilities(bundle, dataset, evgs_extractor=None):
    level = bundle["feature_level"]
    model = bundle["model"]
    evgs_totals = None
    if level == "sequence":
        side_features, _ = build_test_patient_features(dataset)
        if bundle.get("use_evgs", False):
            if evgs_extractor is None:
                raise ValueError("EVGS features are enabled in the model bundle, but no extractor was provided.")
            side_evgs = build_test_side_evgs(dataset, evgs_extractor)
            evgs_totals = {side: float(side_evgs[side][:, 34].mean()) for side in (0, 1)}
            side_features = {
                side: np.concatenate([side_features[side], side_evgs[side]], axis=1)
                for side in (0, 1)
            }
        probs = {
            side: predict_proba_5(model, side_features[side]).mean(axis=0, keepdims=True)
            for side in (0, 1)
        }
    elif level == "patient":
        side_features, _ = build_test_patient_aggregates(dataset)
        if bundle.get("use_evgs", False):
            if evgs_extractor is None:
                raise ValueError("EVGS features are enabled in the model bundle, but no extractor was provided.")
            side_evgs = build_test_patient_evgs(dataset, evgs_extractor)
            evgs_totals = {side: float(side_evgs[side][0, 34]) for side in (0, 1)}
            side_features = {
                side: np.concatenate([side_features[side], side_evgs[side]], axis=1)
                for side in (0, 1)
            }
        probs = {
            side: predict_proba_5(model, side_features[side])
            for side in (0, 1)
        }
    else:
        raise ValueError(f"Unknown feature level: {level}")

    probs_by_side = {side: apply_prior_adjustment(bundle, prob)[0] for side, prob in probs.items()}
    return apply_side_fusion(bundle, probs_by_side), evgs_totals


def aux_patient_probabilities(bundle, dataset, evgs_extractor=None):
    model = bundle.get("aux_patient_model")
    if model is None:
        return None

    side_features, _ = build_test_patient_aggregates(dataset)
    if bundle.get("use_evgs", False):
        if evgs_extractor is None:
            raise ValueError("EVGS features are enabled in the model bundle, but no extractor was provided.")
        side_evgs = build_test_patient_evgs(dataset, evgs_extractor)
        side_features = {
            side: np.concatenate([side_features[side], side_evgs[side]], axis=1)
            for side in (0, 1)
        }

    probs_by_side = {}
    for side in (0, 1):
        probs = predict_proba_5(model, side_features[side])
        adjusted = adjust_probabilities(
            probs,
            float(bundle.get("aux_patient_prior_alpha", 0.0)),
            bundle.get("aux_patient_class_counts", {}),
        )
        probs_by_side[side] = adjusted[0]

    return apply_side_fusion_weight(
        probs_by_side,
        float(bundle.get("aux_patient_side_fusion_weight", 0.0)),
    )


def apply_type2_over_type4_gate(bundle, pred_ids, aux_probs_by_side):
    gate = bundle.get("type2_over_type4_gate", {})
    if not gate.get("enabled", False) or aux_probs_by_side is None:
        return pred_ids, False

    patient_pred = max(set(pred_ids.values()), key=list(pred_ids.values()).count)
    if patient_pred != 4:
        return pred_ids, False

    aux_patient_prob = (aux_probs_by_side[0] + aux_probs_by_side[1]) / 2.0
    aux_top = int(aux_patient_prob.argmax())
    aux_conf = float(aux_patient_prob[aux_top])
    if aux_top == 2 and aux_conf >= float(gate.get("aux_confidence_floor", 1.0)):
        return {0: 2, 1: 2}, True

    return pred_ids, False


def apply_mild_type1_gate(bundle, probs_by_side, pred_ids, evgs_totals):
    gate = bundle.get("mild_type1_gate", {})
    if not gate.get("enabled", False) or evgs_totals is None:
        return pred_ids, False

    patient_pred = max(set(pred_ids.values()), key=list(pred_ids.values()).count)
    if patient_pred not in set(int(x) for x in gate.get("from_labels", [])):
        return pred_ids, False

    if max(evgs_totals.values()) > float(gate.get("evgs_threshold", 0.0)):
        return pred_ids, False

    patient_prob = (probs_by_side[0] + probs_by_side[1]) / 2.0
    if float(patient_prob[1]) >= float(gate.get("type1_probability_floor", 1.0)):
        return {0: 1, 1: 1}, True

    return pred_ids, False


def apply_wnl_gate(bundle, probs_by_side, pred_ids, evgs_totals):
    gate = bundle.get("wnl_gate", {})
    if not gate.get("enabled", False) or evgs_totals is None:
        return pred_ids, False

    threshold = float(gate.get("threshold", 0.0))
    if max(evgs_totals.values()) > threshold:
        return pred_ids, False

    patient_prob = (probs_by_side[0] + probs_by_side[1]) / 2.0
    top_label = int(patient_prob.argmax())
    top_conf = float(patient_prob[top_label])
    exempt_labels = set(int(x) for x in gate.get("exempt_labels", []))
    exempt_confidence = float(gate.get("exempt_confidence", 1.0))
    if top_label in exempt_labels and top_conf >= exempt_confidence:
        return pred_ids, False

    return {0: 0, 1: 0}, True


def main():
    args = parse_args()
    bundle = joblib.load(args.model)
    id_to_label = bundle["id_to_label"]

    test_paths = sorted(glob.glob(os.path.join(args.dataset_dir, args.pattern)), key=natural_key)
    if not test_paths:
        raise FileNotFoundError(f"No test pkl matched {os.path.join(args.dataset_dir, args.pattern)}")

    print(f"Loaded model: {args.model}")
    print(
        f"Selected model: {bundle['model_name']} ({bundle['feature_level']}), "
        f"prior_alpha={bundle['prior_alpha']}, side_fusion={bundle.get('side_fusion_weight', 0.0)}"
    )
    print(f"use_evgs_features={bundle.get('use_evgs', False)}")
    print(f"Predicting {len(test_paths)} Track 2 test files from {args.dataset_dir}\n")

    evgs_extractor = None
    if bundle.get("use_evgs", False):
        evgs_extractor = Track1EVGSExtractor(bundle["track1_config"], bundle["track1_model"])

    for path in test_paths:
        dataset = read_pkl(path)
        probs_by_side, evgs_totals = side_probabilities(bundle, dataset, evgs_extractor)
        aux_probs_by_side = aux_patient_probabilities(bundle, dataset, evgs_extractor)
        pred_ids = {side: int(probs_by_side[side].argmax()) for side in (0, 1)}
        notes = []

        pred_ids, used_type2_gate = apply_type2_over_type4_gate(bundle, pred_ids, aux_probs_by_side)
        if used_type2_gate:
            notes.append("TYPE2_AUX_GATE")

        pred_ids, used_mild_type1_gate = apply_mild_type1_gate(bundle, probs_by_side, pred_ids, evgs_totals)
        if used_mild_type1_gate:
            notes.append("MILD_TYPE1_GATE")

        pred_ids, used_wnl_gate = apply_wnl_gate(bundle, probs_by_side, pred_ids, evgs_totals)
        if used_wnl_gate:
            notes.append("WNL_GATE")

        if args.force_same_limb and pred_ids[0] != pred_ids[1]:
            patient_prob = (probs_by_side[0] + probs_by_side[1]) / 2.0
            shared_pred = int(patient_prob.argmax())
            pred_ids = {0: shared_pred, 1: shared_pred}

        base = os.path.splitext(os.path.basename(path))[0]
        left_label = id_to_label[pred_ids[0]]
        right_label = id_to_label[pred_ids[1]]
        print(f"{base}: Left_gait_subtype={left_label}, Right_gait_subtype={right_label}")

        if args.show_proba:
            left_prob = ",".join(f"{id_to_label[i]}:{probs_by_side[0][i]:.3f}" for i in range(5))
            right_prob = ",".join(f"{id_to_label[i]}:{probs_by_side[1][i]:.3f}" for i in range(5))
            print(f"  left={left_prob} | right={right_prob}")
            if evgs_totals is not None:
                gate_note = f" {'/'.join(notes)}" if notes else ""
                print(f"  evgs_total_left={evgs_totals[0]:.3f}, evgs_total_right={evgs_totals[1]:.3f}{gate_note}")


if __name__ == "__main__":
    main()
