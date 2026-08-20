import numpy as np

from video_dataset_factory.duplicates import (
    clip_perceptual_hash,
    find_duplicate_pairs,
    frame_perceptual_hash,
    hamming_distance,
    mark_duplicates,
)
from video_dataset_factory.schema import ClipRecord


def _record(clip_id, perceptual_hash, keep=True):
    return ClipRecord(
        clip_id=clip_id,
        source_path=f"{clip_id}.mp4",
        duration_sec=2.0,
        fps=24.0,
        width=512,
        height=512,
        frame_count=48,
        perceptual_hash=perceptual_hash,
        keep=keep,
        reject_reasons=[],
    )


def test_frame_perceptual_hash_is_stable_for_same_frame():
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    frame[:, 16:] = 255

    assert frame_perceptual_hash(frame) == frame_perceptual_hash(frame.copy())


def test_clip_perceptual_hash_handles_empty_frames():
    assert clip_perceptual_hash([]) is None


def test_hamming_distance_counts_changed_bits():
    assert hamming_distance("00", "03") == 2


def test_find_duplicate_pairs_uses_first_matching_anchor():
    records = [
        _record("anchor", "0000000000000000"),
        _record("near", "0000000000000003"),
        _record("far", "ffffffffffffffff"),
    ]

    pairs = find_duplicate_pairs(records, threshold=2)

    assert len(pairs) == 1
    assert pairs[0].clip_id == "near"
    assert pairs[0].duplicate_of == "anchor"
    assert pairs[0].distance == 2


def test_mark_duplicates_rejects_near_duplicate_records():
    records = [
        _record("anchor", "0000000000000000"),
        _record("near", "0000000000000001"),
    ]

    marked = mark_duplicates(records, threshold=1)

    assert marked[0].keep is True
    assert marked[1].keep is False
    assert marked[1].duplicate_of == "anchor"
    assert marked[1].reject_reasons == ["near_duplicate"]
