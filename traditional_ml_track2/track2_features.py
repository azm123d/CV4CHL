import os
import pickle
import re
from typing import Dict, Iterable, List, Tuple

import numpy as np


LABEL_MAP = {
    "WNL": 0,
    "type1": 1,
    "type2": 2,
    "type3": 3,
    "type4": 4,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

LEFT_LEG = [6, 8, 10, 12, 13, 14]
RIGHT_LEG = [7, 9, 11, 15, 16, 17]
LEFT_SIDE = {"shoulder": 0, "hip": 6, "knee": 8, "ankle": 10, "foot": [12, 13, 14]}
RIGHT_SIDE = {"shoulder": 1, "hip": 7, "knee": 9, "ankle": 11, "foot": [15, 16, 17]}
VIEWS = ("left", "right", "forward", "backward")


def read_pkl(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def parse_patient_id(name: str) -> int:
    match = re.match(r"^(\d+)", os.path.basename(name))
    return int(match.group(1)) if match else -1


def parse_view(name: str) -> str:
    lower = os.path.basename(name).lower()
    for view in VIEWS:
        if view in lower:
            return view
    return "unknown"


def _interp_1d(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=True)
    idx = np.arange(values.shape[0], dtype=np.float32)
    if valid.sum() == 0:
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    if valid.sum() == 1:
        values[:] = values[valid][0]
        return values
    values[~valid] = np.interp(idx[~valid], idx[valid], values[valid])
    return values


def interpolate_object_track(motion: np.ndarray, obj_ids: np.ndarray) -> np.ndarray:
    """Keep the dominant person id and interpolate missing/off-track frames."""
    if obj_ids is None or len(obj_ids) != len(motion):
        return motion

    valid_ids = obj_ids[obj_ids >= 0]
    if valid_ids.size == 0:
        return motion

    ids, counts = np.unique(valid_ids, return_counts=True)
    dominant_id = ids[np.argmax(counts)]
    valid_frames = obj_ids == dominant_id
    if valid_frames.sum() == 0 or valid_frames.sum() == len(motion):
        return motion

    cleaned = motion.astype(np.float32, copy=True)
    for joint in range(cleaned.shape[1]):
        for channel in range(cleaned.shape[2]):
            cleaned[:, joint, channel] = _interp_1d(cleaned[:, joint, channel], valid_frames)
    return cleaned


def interpolate_bad_values(motion: np.ndarray) -> np.ndarray:
    cleaned = motion.astype(np.float32, copy=True)
    for joint in range(cleaned.shape[1]):
        for channel in range(cleaned.shape[2]):
            series = cleaned[:, joint, channel]
            valid = np.isfinite(series)
            cleaned[:, joint, channel] = _interp_1d(series, valid)
    return cleaned


def crop_scale(motion: np.ndarray, scale_range: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    """Same normalization style as lib/data/utils_data.py, kept local for this ML baseline."""
    result = motion.astype(np.float32, copy=True)
    valid_coords = motion[motion[..., 2] != 0][:, :2]
    if len(valid_coords) < 4:
        return np.zeros_like(motion, dtype=np.float32)

    xmin, ymin = valid_coords.min(axis=0)
    xmax, ymax = valid_coords.max(axis=0)
    ratio = np.random.uniform(low=scale_range[0], high=scale_range[1])
    scale = max(xmax - xmin, ymax - ymin) * ratio
    if scale == 0:
        return np.zeros_like(motion, dtype=np.float32)

    xs = (xmin + xmax - scale) / 2.0
    ys = (ymin + ymax - scale) / 2.0
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2.0
    result = np.clip(result, -1.0, 1.0)
    return result.astype(np.float32)


def clean_motion(record: Dict, scale_range: Tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    keypoints = np.asarray(record["keypoints"], dtype=np.float32)
    scores = np.asarray(record.get("keypoint_scores", np.ones(keypoints.shape[:2])), dtype=np.float32)
    if scores.ndim == 2:
        scores = scores[..., None]

    motion = np.concatenate([keypoints, scores], axis=-1)
    motion = interpolate_object_track(motion, np.asarray(record.get("obj_ids", [])))
    motion = interpolate_bad_values(motion)
    return crop_scale(motion, scale_range=scale_range)


def _safe_stats(values: np.ndarray, prefix: str, names: List[str], out: List[float]) -> None:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        arr = np.zeros(1, dtype=np.float32)

    q10, q25, q50, q75, q90 = np.percentile(arr, [10, 25, 50, 75, 90])
    stats = [
        arr.mean(),
        arr.std(),
        arr.min(),
        arr.max(),
        arr.max() - arr.min(),
        q10,
        q25,
        q50,
        q75,
        q90,
        q75 - q25,
    ]
    stat_names = ["mean", "std", "min", "max", "range", "p10", "p25", "p50", "p75", "p90", "iqr"]
    out.extend(float(x) for x in stats)
    names.extend([f"{prefix}_{name}" for name in stat_names])


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1)
    cos = np.sum(ba * bc, axis=-1) / np.maximum(denom, 1e-6)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a - b, axis=-1)


def _speed(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.zeros((1, points.shape[1]), dtype=np.float32)
    return np.linalg.norm(np.diff(points, axis=0), axis=-1)


def _side_spec(side: int):
    return (LEFT_SIDE, RIGHT_SIDE, LEFT_LEG, RIGHT_LEG) if side == 0 else (RIGHT_SIDE, LEFT_SIDE, RIGHT_LEG, LEFT_LEG)


def extract_side_features(motion: np.ndarray, side: int, view: str = "unknown") -> Tuple[np.ndarray, List[str]]:
    """Extract compact, side-canonical gait features for one limb."""
    if side not in (0, 1):
        raise ValueError("side must be 0 for left or 1 for right")

    work = motion.astype(np.float32, copy=True)
    if side == 1:
        work[..., 0] *= -1.0

    side_spec, opp_spec, side_leg, opp_leg = _side_spec(side)
    xy = work[..., :2]
    conf = motion[..., 2]
    pelvis = (xy[:, LEFT_SIDE["hip"]] + xy[:, RIGHT_SIDE["hip"]]) / 2.0

    names: List[str] = []
    feats: List[float] = []

    view_vec = [1.0 if view == v else 0.0 for v in VIEWS]
    feats.extend(view_vec)
    names.extend([f"view_{v}" for v in VIEWS])

    rel_side = xy[:, side_leg] - pelvis[:, None, :]
    rel_opp = xy[:, opp_leg] - pelvis[:, None, :]

    for local_idx, joint_idx in enumerate(side_leg):
        _safe_stats(rel_side[:, local_idx, 0], f"j{joint_idx}_rel_x", names, feats)
        _safe_stats(rel_side[:, local_idx, 1], f"j{joint_idx}_rel_y", names, feats)
        _safe_stats(conf[:, joint_idx], f"j{joint_idx}_conf", names, feats)

    _safe_stats(rel_side[..., 0], "side_all_rel_x", names, feats)
    _safe_stats(rel_side[..., 1], "side_all_rel_y", names, feats)
    _safe_stats(conf[:, side_leg], "side_all_conf", names, feats)
    _safe_stats(conf.mean(axis=1), "global_conf", names, feats)

    hip = xy[:, side_spec["hip"]]
    knee = xy[:, side_spec["knee"]]
    ankle = xy[:, side_spec["ankle"]]
    shoulder = xy[:, side_spec["shoulder"]]
    foot = xy[:, side_spec["foot"]].mean(axis=1)

    opp_hip = xy[:, opp_spec["hip"]]
    opp_knee = xy[:, opp_spec["knee"]]
    opp_ankle = xy[:, opp_spec["ankle"]]
    opp_shoulder = xy[:, opp_spec["shoulder"]]
    opp_foot = xy[:, opp_spec["foot"]].mean(axis=1)

    knee_angle = _angle(hip, knee, ankle)
    hip_angle = _angle(shoulder, hip, knee)
    ankle_angle = _angle(knee, ankle, foot)
    opp_knee_angle = _angle(opp_hip, opp_knee, opp_ankle)
    opp_hip_angle = _angle(opp_shoulder, opp_hip, opp_knee)
    opp_ankle_angle = _angle(opp_knee, opp_ankle, opp_foot)

    for prefix, series in [
        ("knee_angle", knee_angle),
        ("hip_angle", hip_angle),
        ("ankle_angle", ankle_angle),
        ("knee_angle_delta", np.diff(knee_angle) if len(knee_angle) > 1 else np.zeros(1)),
        ("hip_angle_delta", np.diff(hip_angle) if len(hip_angle) > 1 else np.zeros(1)),
        ("ankle_angle_delta", np.diff(ankle_angle) if len(ankle_angle) > 1 else np.zeros(1)),
    ]:
        _safe_stats(series, prefix, names, feats)

    distances = {
        "upper_leg": _dist(hip, knee),
        "lower_leg": _dist(knee, ankle),
        "ankle_to_foot": _dist(ankle, foot),
        "shoulder_to_hip": _dist(shoulder, hip),
        "foot_spread": _dist(xy[:, side_spec["foot"][0]], xy[:, side_spec["foot"][-1]]),
    }
    opp_distances = {
        "upper_leg": _dist(opp_hip, opp_knee),
        "lower_leg": _dist(opp_knee, opp_ankle),
        "ankle_to_foot": _dist(opp_ankle, opp_foot),
        "shoulder_to_hip": _dist(opp_shoulder, opp_hip),
        "foot_spread": _dist(xy[:, opp_spec["foot"][0]], xy[:, opp_spec["foot"][-1]]),
    }

    for prefix, series in distances.items():
        _safe_stats(series, prefix, names, feats)
        _safe_stats(series - opp_distances[prefix], f"{prefix}_asym_signed", names, feats)
        _safe_stats(np.abs(series - opp_distances[prefix]), f"{prefix}_asym_abs", names, feats)

    for prefix, series in [
        ("knee_angle_asym_signed", knee_angle - opp_knee_angle),
        ("hip_angle_asym_signed", hip_angle - opp_hip_angle),
        ("ankle_angle_asym_signed", ankle_angle - opp_ankle_angle),
        ("knee_angle_asym_abs", np.abs(knee_angle - opp_knee_angle)),
        ("hip_angle_asym_abs", np.abs(hip_angle - opp_hip_angle)),
        ("ankle_angle_asym_abs", np.abs(ankle_angle - opp_ankle_angle)),
    ]:
        _safe_stats(series, prefix, names, feats)

    side_speed = _speed(xy[:, side_leg])
    opp_speed = _speed(xy[:, opp_leg])
    _safe_stats(side_speed, "side_joint_speed", names, feats)
    _safe_stats(side_speed - opp_speed, "joint_speed_asym_signed", names, feats)
    _safe_stats(np.abs(side_speed - opp_speed), "joint_speed_asym_abs", names, feats)

    # Coarse temporal bins preserve a little phase information without exploding dimensionality.
    bins = np.array_split(np.arange(len(motion)), 6)
    for i, idx in enumerate(bins):
        if len(idx) == 0:
            idx = np.array([0])
        feats.extend([
            float(knee_angle[idx].mean()),
            float(hip_angle[idx].mean()),
            float(ankle_angle[idx].mean()),
            float(distances["upper_leg"][idx].mean()),
            float(distances["lower_leg"][idx].mean()),
        ])
        names.extend([
            f"bin{i}_knee_angle_mean",
            f"bin{i}_hip_angle_mean",
            f"bin{i}_ankle_angle_mean",
            f"bin{i}_upper_leg_mean",
            f"bin{i}_lower_leg_mean",
        ])

    arr = np.asarray(feats, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr, names


def build_side_examples(dataset: Dict, scale_range: Tuple[float, float] = (1.0, 1.0)):
    x_rows, y_rows, groups, meta = [], [], [], []
    feature_names = None

    for seq_name, record in sorted(dataset.items()):
        if "label" not in record:
            continue
        motion = clean_motion(record, scale_range=scale_range)
        view = parse_view(seq_name)
        labels = [
            record["label"]["left"]["gait_subtype"],
            record["label"]["right"]["gait_subtype"],
        ]
        for side, label in enumerate(labels):
            feats, names = extract_side_features(motion, side=side, view=view)
            if feature_names is None:
                feature_names = names
            x_rows.append(feats)
            y_rows.append(LABEL_MAP[label])
            groups.append(parse_patient_id(seq_name))
            meta.append({"sequence": seq_name, "side": "left" if side == 0 else "right", "label": label, "view": view})

    return np.vstack(x_rows), np.asarray(y_rows), np.asarray(groups), meta, feature_names or []


def aggregate_feature_rows(rows: Iterable[np.ndarray]) -> np.ndarray:
    stacked = np.vstack(list(rows))
    return np.concatenate([
        stacked.mean(axis=0),
        stacked.std(axis=0),
        stacked.min(axis=0),
        stacked.max(axis=0),
        np.median(stacked, axis=0),
    ]).astype(np.float32)


def build_patient_examples(dataset: Dict, scale_range: Tuple[float, float] = (1.0, 1.0)):
    by_patient_side = {}
    labels = {}
    feature_names = None

    for seq_name, record in sorted(dataset.items()):
        if "label" not in record:
            continue
        patient_id = parse_patient_id(seq_name)
        motion = clean_motion(record, scale_range=scale_range)
        view = parse_view(seq_name)
        side_labels = [
            record["label"]["left"]["gait_subtype"],
            record["label"]["right"]["gait_subtype"],
        ]
        for side, label in enumerate(side_labels):
            feats, names = extract_side_features(motion, side=side, view=view)
            if feature_names is None:
                feature_names = names
            key = (patient_id, side)
            by_patient_side.setdefault(key, []).append(feats)
            labels[key] = LABEL_MAP[label]

    x_rows, y_rows, groups, meta = [], [], [], []
    for (patient_id, side), rows in sorted(by_patient_side.items()):
        x_rows.append(aggregate_feature_rows(rows))
        y_rows.append(labels[(patient_id, side)])
        groups.append(patient_id)
        meta.append({"patient_id": patient_id, "side": "left" if side == 0 else "right"})

    agg_feature_names = []
    for prefix in ("mean", "std", "min", "max", "median"):
        agg_feature_names.extend([f"patient_{prefix}_{name}" for name in (feature_names or [])])

    return np.vstack(x_rows), np.asarray(y_rows), np.asarray(groups), meta, agg_feature_names


def build_test_patient_features(dataset: Dict, scale_range: Tuple[float, float] = (1.0, 1.0)):
    by_side = {0: [], 1: []}
    meta = []
    for seq_name, record in sorted(dataset.items()):
        motion = clean_motion(record, scale_range=scale_range)
        view = parse_view(seq_name)
        for side in (0, 1):
            feats, _ = extract_side_features(motion, side=side, view=view)
            by_side[side].append(feats)
            meta.append({"sequence": seq_name, "side": "left" if side == 0 else "right", "view": view})
    return {side: np.vstack(rows) if rows else np.zeros((0, 1), dtype=np.float32) for side, rows in by_side.items()}, meta


def build_test_patient_aggregates(dataset: Dict, scale_range: Tuple[float, float] = (1.0, 1.0)):
    side_features, meta = build_test_patient_features(dataset, scale_range=scale_range)
    aggregated = {
        side: aggregate_feature_rows(rows) if len(rows) else np.zeros(1, dtype=np.float32)
        for side, rows in side_features.items()
    }
    return {side: feat[None, :] for side, feat in aggregated.items()}, meta
