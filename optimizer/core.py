import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from optimizer.params import decode_gradient_vector_to_dict, get_search_bounds
from optimizer.objective_function import compute_objective
from decoding.peak_decoder import run_peak_decoding
from decoding.assign_compounds import assign_compounds


def evaluate_gradient(vector: torch.Tensor, alpha: float = 1.0) -> float:
    gradient_dict = decode_gradient_vector_to_dict(vector.tolist())
    summary = run_peak_decoding(gradient_dict)
    assigned = assign_compounds(summary)
    return compute_objective(assigned, alpha)


def run_gradient_optimization(alpha: float = 1.0, n_init: int = 10, n_iter: int = 10):
    bounds = torch.tensor(get_search_bounds(), dtype=torch.float).T

    # Sobol initialization
    sobol = torch.quasirandom.SobolEngine(dimension=12, scramble=True)
    train_X = bounds[0] + (bounds[1] - bounds[0]) * sobol.draw(n_init)
    train_Y = torch.tensor([[evaluate_gradient(x, alpha)] for x in train_X])

    # Fit surrogate model
    model = SingleTaskGP(train_X, train_Y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    for _ in range(n_iter):
        EI = ExpectedImprovement(model=model, best_f=train_Y.max(), maximize=True)
        candidate, _ = optimize_acqf(
            acq_function=EI,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=32,
        )
        new_y = torch.tensor([[evaluate_gradient(candidate[0], alpha)]])
        train_X = torch.cat([train_X, candidate])
        train_Y = torch.cat([train_Y, new_y])
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

    best_idx = train_Y.argmax()
    best_vector = train_X[best_idx].tolist()
    best_gradient = decode_gradient_vector_to_dict(best_vector)
    return best_gradient, train_Y[best_idx].item()
