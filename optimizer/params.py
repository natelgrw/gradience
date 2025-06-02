def encode_gradient_dict_to_vector(gradient_dict: dict) -> list[float]:
    """
    Converts a gradient dictionary into a flat parameter vector.
    Input gradient_dict format:
        {
            "points": [(t1, A1), ..., (t5, A5)],
            "flow_rate": float,
            "injection_volume": float
        }
    """
    vec = []
    for t, A in gradient_dict["points"]:
        vec.extend([t, A])
    vec.append(gradient_dict["flow_rate"])
    vec.append(gradient_dict["injection_volume"])
    return vec

def decode_gradient_vector_to_dict(vector: list[float]) -> dict:
    """
    Converts a flat parameter vector into a structured gradient dictionary.
    Output:
        {
            "points": [(t1, A1), ..., (t5, A5)],
            "flow_rate": float,
            "injection_volume": float
        }
    """
    assert len(vector) == 12
    points = [(vector[i], vector[i+1]) for i in range(0, 10, 2)]
    return {
        "points": points,
        "flow_rate": vector[10],
        "injection_volume": vector[11]
    }

def get_search_bounds():
    """
    Returns parameter bounds for optimization.
    """
    time_bounds = [(0.0, 12.0)] * 5
    A_bounds = [(0.0, 100.0)] * 5
    flow_bound = [(0.1, 2.0)]
    inj_bound = [(1.0, 50.0)]
    return time_bounds + A_bounds + flow_bound + inj_bound
