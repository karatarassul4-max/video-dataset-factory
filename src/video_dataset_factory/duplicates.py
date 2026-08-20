from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from video_dataset_factory.schema import ClipRecord


@dataclass(frozen=True)
class DuplicatePair:
    clip_id: str
    duplicate_of: str
    distance: int


def frame_perceptual_hash(frame: np.ndarray, hash_size: int = 8) -> str:
    if hash_size < 2:
        raise ValueError("hash_size must be at least 2")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    median = float(np.median(resized))
    bits = (resized >= median).astype(np.uint8).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    width = (hash_size * hash_size + 3) // 4
    return f"{value:0{width}x}"


def clip_perceptual_hash(frames: list[np.ndarray], hash_size: int = 8) -> str | None:
    if not frames:
        return None

    frame_hashes = [int(frame_perceptual_hash(frame, hash_size), 16) for frame in frames]
    bit_count = hash_size * hash_size
    consensus = 0
    for bit_index in reversed(range(bit_count)):
        votes = sum((hash_value >> bit_index) & 1 for hash_value in frame_hashes)
        consensus = (consensus << 1) | int(votes >= (len(frame_hashes) / 2))
    width = (bit_count + 3) // 4
    return f"{consensus:0{width}x}"


def hamming_distance(left_hash: str, right_hash: str) -> int:
    if len(left_hash) != len(right_hash):
        raise ValueError("hashes must have the same length")
    return (int(left_hash, 16) ^ int(right_hash, 16)).bit_count()


def find_duplicate_pairs(records: list[ClipRecord], threshold: int = 6) -> list[DuplicatePair]:
    anchors: list[ClipRecord] = []
    pairs: list[DuplicatePair] = []

    for record in records:
        if not record.perceptual_hash:
            anchors.append(record)
            continue

        duplicate = _find_nearest_anchor(record, anchors, threshold)
        if duplicate is None:
            anchors.append(record)
            continue

        duplicate_of, distance = duplicate
        pairs.append(
            DuplicatePair(
                clip_id=record.clip_id,
                duplicate_of=duplicate_of.clip_id,
                distance=distance,
            )
        )

    return pairs


def mark_duplicates(records: list[ClipRecord], threshold: int = 6) -> list[ClipRecord]:
    pair_by_clip = {pair.clip_id: pair for pair in find_duplicate_pairs(records, threshold)}
    marked: list[ClipRecord] = []

    for record in records:
        pair = pair_by_clip.get(record.clip_id)
        if pair is None:
            marked.append(record)
            continue

        reasons = list(record.reject_reasons)
        if "near_duplicate" not in reasons:
            reasons.append("near_duplicate")
        marked.append(
            record.model_copy(
                update={
                    "keep": False,
                    "duplicate_of": pair.duplicate_of,
                    "reject_reasons": reasons,
                }
            )
        )

    return marked


def _find_nearest_anchor(
    record: ClipRecord,
    anchors: list[ClipRecord],
    threshold: int,
) -> tuple[ClipRecord, int] | None:
    best: tuple[ClipRecord, int] | None = None
    for anchor in anchors:
        if not anchor.perceptual_hash or not record.perceptual_hash:
            continue
        distance = hamming_distance(record.perceptual_hash, anchor.perceptual_hash)
        if distance <= threshold and (best is None or distance < best[1]):
            best = (anchor, distance)
    return best
