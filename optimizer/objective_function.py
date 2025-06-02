

def compute_objective(summary: dict, alpha: float = 1.0) -> float:
    """
    Computes the separation quality score for a given LC-MS summary.

    Score = min distance between different compound_ids
            - alpha * max distance within same compound_id

    Parameters:
        summary (dict): summary dictionary from decoding
        alpha (float): penalty weight for intra-compound peak spread

    Returns:
        float: computed objective score
    """
    import numpy as np
    from collections import defaultdict

    peaks = summary.get("peaks", [])
    compound_groups = defaultdict(list)
    centers = []

    for peak in peaks:
        compound_id = peak.get("compound_id")
        if compound_id is None:
            continue
        start, end = peak["time_range"]
        center = (start + end) / 2
        compound_groups[compound_id].append(center)
        centers.append((compound_id, center))

    # Compute max intra-compound distance
    d_intra = 0.0
    for group in compound_groups.values():
        if len(group) >= 2:
            distances = [abs(a - b) for i, a in enumerate(group) for b in group[i+1:]]
            if distances:
                d_intra = max(d_intra, max(distances))

    # Compute min inter-compound distance
    d_inter = float("inf")
    for i in range(len(centers)):
        id1, c1 = centers[i]
        for j in range(i + 1, len(centers)):
            id2, c2 = centers[j]
            if id1 != id2:
                d_inter = min(d_inter, abs(c1 - c2))

    if d_inter == float("inf"):
        return -1e6  # penalize if only one compound detected
    return d_inter - alpha * d_intra