import os
import sys
import hashlib
from typing import Dict

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lib.model.model_gait import GaitNet_1
from lib.utils.learning import load_backbone
from lib.utils.tools import get_config

from track2_features import aggregate_feature_rows, clean_motion, parse_patient_id


class Track1EVGSExtractor:
    def __init__(
        self,
        config_path: str = "configs/gait/MB_ft_gait_track1.yaml",
        checkpoint_path: str = "checkpoint/gait1/best.pth",
        device: str = None,
        cache_dir: str = "traditional_ml_track2/evgs_cache",
    ):
        self.args = get_config(config_path)
        self.batch_frames = int(self.args.maxlen)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        checkpoint_stamp = os.path.getmtime(checkpoint_path) if os.path.exists(checkpoint_path) else 0
        cache_signature = f"{config_path}|{checkpoint_path}|{checkpoint_stamp}|{self.batch_frames}"
        self.cache_prefix = hashlib.md5(cache_signature.encode("utf-8")).hexdigest()[:12]

        backbone = load_backbone(self.args)
        self.model = GaitNet_1(
            backbone=backbone,
            num_joints=self.args.num_joints,
            dim_rep=self.args.dim_rep,
            num_classes=self.args.EVGS_classes,
            dropout_ratio=self.args.dropout_ratio,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    def split_motion(self, motion: np.ndarray):
        total_frames = len(motion)
        batch_frames = self.batch_frames
        clips = []

        num_splits = total_frames // batch_frames
        remainder = total_frames % batch_frames
        for split_idx in range(num_splits):
            start = split_idx * batch_frames
            clips.append(motion[start:start + batch_frames])

        if remainder > batch_frames // 4:
            leftover = motion[-remainder:]
            clips.append(self._bounce_pad(leftover, batch_frames))

        if not clips and total_frames > 0:
            clips.append(self._bounce_pad(motion, batch_frames))

        if not clips:
            clips.append(np.zeros((batch_frames, 18, 3), dtype=np.float32))
        return np.stack(clips).astype(np.float32)

    @staticmethod
    def _bounce_pad(sequence: np.ndarray, target_len: int):
        if len(sequence) >= target_len:
            return sequence[:target_len]
        reversed_sequence = sequence[::-1]
        pieces = []
        current_len = 0
        forward = True
        while current_len < target_len:
            piece = sequence if forward else reversed_sequence
            pieces.append(piece)
            current_len += len(piece)
            forward = not forward
        return np.concatenate(pieces, axis=0)[:target_len]

    def _cache_path(self, cache_key: str):
        if not self.cache_dir or cache_key is None:
            return None
        key_digest = hashlib.md5(str(cache_key).encode("utf-8")).hexdigest()
        return os.path.join(self.cache_dir, f"{self.cache_prefix}_{key_digest}.npy")

    @torch.no_grad()
    def predict_record(self, record: Dict, cache_key: str = None) -> np.ndarray:
        cache_path = self._cache_path(cache_key)
        if cache_path and os.path.exists(cache_path):
            return np.load(cache_path).astype(np.float32)

        motion = clean_motion(record, scale_range=(1.0, 1.0))
        clips = self.split_motion(motion)
        tensor = torch.tensor(clips, dtype=torch.float32, device=self.device)
        logits = self.model(tensor)
        probs = torch.sigmoid(logits).mean(dim=0).detach().cpu().numpy()
        if cache_path:
            np.save(cache_path, probs.astype(np.float32))
        return probs.astype(np.float32)


def side_evgs_features(evgs_probs: np.ndarray, side: int) -> np.ndarray:
    own = evgs_probs[side]
    other = evgs_probs[1 - side]
    own_total = np.asarray([own.sum(), own.mean(), own.std(), own.max(), own.min()], dtype=np.float32)
    other_total = np.asarray([other.sum(), other.mean(), other.std(), other.max(), other.min()], dtype=np.float32)
    diff = own - other
    diff_stats = np.asarray([diff.mean(), diff.std(), np.abs(diff).mean(), np.abs(diff).max()], dtype=np.float32)
    patient_total = np.asarray([evgs_probs.sum(), evgs_probs.mean(), evgs_probs.std()], dtype=np.float32)
    return np.concatenate([own, other, own_total, other_total, diff_stats, patient_total]).astype(np.float32)


def build_side_evgs_examples(dataset: Dict, extractor: Track1EVGSExtractor) -> np.ndarray:
    rows = []
    for seq_name, record in sorted(dataset.items()):
        if "label" not in record:
            continue
        evgs_probs = extractor.predict_record(record, cache_key=seq_name)
        for side in (0, 1):
            rows.append(side_evgs_features(evgs_probs, side))
    return np.vstack(rows).astype(np.float32)


def build_patient_evgs_examples(dataset: Dict, extractor: Track1EVGSExtractor) -> np.ndarray:
    by_patient_side = {}
    for seq_name, record in sorted(dataset.items()):
        if "label" not in record:
            continue
        patient_id = parse_patient_id(seq_name)
        evgs_probs = extractor.predict_record(record, cache_key=seq_name)
        for side in (0, 1):
            by_patient_side.setdefault((patient_id, side), []).append(side_evgs_features(evgs_probs, side))

    rows = []
    for key in sorted(by_patient_side):
        rows.append(aggregate_feature_rows(by_patient_side[key]))
    return np.vstack(rows).astype(np.float32)


def build_test_side_evgs(dataset: Dict, extractor: Track1EVGSExtractor):
    by_side = {0: [], 1: []}
    for seq_name, record in sorted(dataset.items()):
        evgs_probs = extractor.predict_record(record, cache_key=seq_name)
        for side in (0, 1):
            by_side[side].append(side_evgs_features(evgs_probs, side))
    return {side: np.vstack(rows).astype(np.float32) for side, rows in by_side.items()}


def build_test_patient_evgs(dataset: Dict, extractor: Track1EVGSExtractor):
    side_rows = build_test_side_evgs(dataset, extractor)
    return {
        side: aggregate_feature_rows(rows)[None, :].astype(np.float32)
        for side, rows in side_rows.items()
    }
