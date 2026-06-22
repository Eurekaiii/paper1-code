"""
AeroMoDE Expert Similarity & Candidate Construction
=====================================================
Cosine similarity between expert FFN weights → substitutable sets A(r)
→ candidate expert set ℰ_cand.

Reference: aeromde_main.tex  Section IV-C  (Eqs. 9–11).
"""

from typing import List, Dict, Set, Tuple, Optional
import numpy as np

from .models import Expert


# ---------------------------------------------------------------------------
# Cosine similarity  Sim(r, e)  — Eq. (9)
# ---------------------------------------------------------------------------
def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Sim(r, e) = ⟨vec(W_r), vec(W_e)⟩ / (||·||₂ · ||·||₂)."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compute_similarity_matrix(
    experts: List[Expert],
) -> Dict[Tuple[int, int], float]:
    """Compute pairwise cosine similarity for all experts that have weight vectors.

    Returns dict  (id_a, id_b) → Sim(a, b).
    """
    sim: Dict[Tuple[int, int], float] = {}
    for i, ea in enumerate(experts):
        if ea.weight_vector is None:
            continue
        for j, eb in enumerate(experts):
            if eb.weight_vector is None:
                continue
            if i == j:
                sim[(ea.id, eb.id)] = 1.0
            else:
                val = cosine_similarity(ea.weight_vector, eb.weight_vector)
                sim[(ea.id, eb.id)] = val
    return sim


# ---------------------------------------------------------------------------
# Substitutable set  A(r)  — Eq. (10)
# ---------------------------------------------------------------------------
def build_substitutable_sets(
    similarity: Dict[Tuple[int, int], float],
    required_experts: Set[int],
    xi: float,
) -> Dict[int, Set[int]]:
    """A(r) = {r} ∪ {e | Sim(r, e) ≥ ξ}.

    Parameters
    ----------
    similarity : dict  (r, e) → Sim(r, e)
    required_experts : set  ℰ_req
    xi : float  similarity threshold

    Returns
    -------
    A : dict  A[r] = {r, e1, e2, ...}
    """
    # Collect all expert ids that appear in the similarity matrix
    all_experts: Set[int] = set()
    for (a, b) in similarity:
        all_experts.add(a)
        all_experts.add(b)

    A: Dict[int, Set[int]] = {}
    for r in required_experts:
        A[r] = {r}
        for e in all_experts:
            if e == r:
                continue
            s = similarity.get((r, e), 0.0)
            if s >= xi:
                A[r].add(e)
    return A


# ---------------------------------------------------------------------------
# Candidate expert set  ℰ_cand  — Eq. (11)
# ---------------------------------------------------------------------------
def build_candidate_set(
    substitutable_sets: Dict[int, Set[int]],
) -> Set[int]:
    """ℰ_cand = ⋃_{r ∈ ℰ_req}  A(r)."""
    cand: Set[int] = set()
    for A_r in substitutable_sets.values():
        cand.update(A_r)
    return cand


# ---------------------------------------------------------------------------
# Strict substitute set  Ā(e) = A(e) \ {e}  (used in redundancy)
# ---------------------------------------------------------------------------
def strict_substitute_set(
    e: int,
    substitutable_sets: Dict[int, Set[int]],
) -> Set[int]:
    """Ā(e) = A(e) \ {e}."""
    return substitutable_sets.get(e, {e}) - {e}
