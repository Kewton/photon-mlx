from __future__ import annotations

from baseline_reporag.photon_pipeline import _merge_protected_and_photon_indices


def test_merge_protected_and_photon_indices_keeps_ranked_top_n() -> None:
    merged = _merge_protected_and_photon_indices(
        ranked_chunk_ids=[f"chunk_{i}" for i in range(16)],
        chunk_ids_for_scoring=[f"chunk_{i}" for i in range(16)],
        photon_indices=[10, 11, 12, 13],
        protected_top_n=4,
    )

    assert merged == [0, 1, 2, 3, 10, 11, 12, 13]


def test_merge_protected_and_photon_indices_deduplicates_overlap() -> None:
    merged = _merge_protected_and_photon_indices(
        ranked_chunk_ids=["chunk_0", "chunk_1", "chunk_2", "chunk_3"],
        chunk_ids_for_scoring=[f"chunk_{i}" for i in range(6)],
        photon_indices=[2, 3, 4, 5],
        protected_top_n=4,
    )

    assert merged == [0, 1, 2, 3, 4, 5]
