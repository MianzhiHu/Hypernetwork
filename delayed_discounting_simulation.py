"""Standalone DD simulation, maximum-likelihood fitting, and evaluation.

Example::

    model = DelayedDiscounting("dd_hyperbolic")
    data = model.simulate(20, 50, 10, 30, 5, 100, 10, seed=42)
    estimates = model.fit(data, num_iterations=20, seed=42)
    metrics = model.evaluate(estimates, data)

Only NumPy, pandas, and SciPy are required; no original model code is imported.
Use an ``if __name__ == "__main__":`` guard when fitting with multiple
processes on Windows. Fit and evaluate also accept the original participant
dictionary format: {participant_id: {column: trial_values}}.
"""

from concurrent.futures import ProcessPoolExecutor
from numbers import Integral
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import truncnorm
MODEL_BOUNDS = {
    "dd_exponential": [(0.01, 20.0), (0.0, 1.0)],
    "dd_hyperbolic": [(0.01, 20.0), (0.0, 1.0)],
    "dd_hyperboloid": [(0.01, 20.0), (0.0, 1.0), (0.0, 5.0)],
}


def _subjective_value(model_type, params, large_amount, later_delay):
    k = params[1]
    if model_type == "dd_exponential":
        return large_amount * np.exp(-k * later_delay)
    if model_type == "dd_hyperbolic":
        return large_amount / (1 + k * later_delay)
    return large_amount / (1 + k * later_delay) ** params[2]


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _trial_arrays(large_amount, later_delay, small_amount, choice):
    arrays = tuple(np.asarray(value, dtype=float) for value in
                   (large_amount, later_delay, small_amount, choice))
    if any(a.ndim != 1 or a.size == 0 for a in arrays):
        raise ValueError("Trial columns must be nonempty one-dimensional arrays")
    if len({len(a) for a in arrays}) != 1 or not all(np.isfinite(a).all() for a in arrays):
        raise ValueError("Trial columns must have equal lengths and finite values")
    large, delay, small, choices = arrays
    if np.any(large < 0) or np.any(small < 0) or np.any(delay < 0):
        raise ValueError("Amounts and delays must be nonnegative (unscaled dollars/days)")
    if not np.isin(choices, [0, 1]).all():
        raise ValueError("choice must be 0 (smaller-sooner) or 1 (larger-later)")
    return arrays


def _participant_data(data):
    if isinstance(data, pd.DataFrame):
        id_column = "worker_id" if "worker_id" in data else "participant_id"
        if id_column not in data or data.empty or data[id_column].isna().any():
            raise ValueError("Data must have nonmissing worker_id or participant_id values and trials")
        groups = list(data.groupby(id_column, sort=False))
    elif isinstance(data, dict) and data:
        groups = list(data.items())
    else:
        raise ValueError("data must be a nonempty DataFrame or participant dictionary")
    columns = ("large_amount", "later_delay", "small_amount", "choice")
    validated = []
    for participant, trials in groups:
        try:
            arrays = _trial_arrays(*(trials[col] for col in columns))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid trials for participant {participant!r}: {exc}") from exc
        validated.append((participant, arrays))
    return validated


def _nll(params, model_type, arrays):
    large, delay, small, choice = arrays
    delta = params[0] * (_subjective_value(model_type, params, large, delay) - small)
    # logaddexp avoids clipping and remains stable for near-deterministic choices.
    return float(np.logaddexp(0, np.where(choice == 1, -delta, delta)).sum())


def _fit_dd_participant(job):
    participant, arrays, model_type, num_iterations, seed = job
    bounds = np.asarray(MODEL_BOUNDS[model_type])
    rng = np.random.default_rng(seed)
    best = None
    n_converged = 0
    last_message = ""
    for _ in range(num_iterations):
        initial = rng.uniform(bounds[:, 0], bounds[:, 1])
        result = minimize(_nll, initial, args=(model_type, arrays),
                          bounds=bounds, method="L-BFGS-B", options={"maxiter": 10000})
        last_message = str(result.message)
        if result.success and np.isfinite(result.fun):
            n_converged += 1
            if best is None or result.fun < best.fun:
                best = result
    if best is None:
        raise RuntimeError(f"No converged fit for participant {participant!r}: {last_message}")
    n = len(arrays[3])
    n_params = len(bounds)
    return {
        "participant_id": participant, "model_type": model_type,
        "best_parameters": best.x.copy(), "best_nll": float(best.fun),
        "AIC": 2 * n_params + 2 * best.fun,
        "BIC": n_params * np.log(n) + 2 * best.fun,
        "n_trials": n, "t": best.x[0], "k": best.x[1],
        "s": best.x[2] if n_params == 3 else np.nan,
        "n_converged": n_converged, "success": True,
    }


class DelayedDiscounting:
    """Standalone model using softmax inverse temperature t, discount k, and s.

    Fits use bounded L-BFGS-B with independent uniform random restarts, selecting
    the lowest NLL among converged runs. Bounds: t in [0.01, 20], k in [0, 1],
    and hyperboloid s in [0, 5]. Inputs retain dollars and days. All supplied
    trials are used; remove missing/nonresponse trials before fitting.
    """

    input_columns = ["large_amount", "later_delay", "small_amount", "choice"]

    def __init__(self, model_type="dd_hyperbolic"):
        if model_type not in MODEL_BOUNDS:
            raise ValueError("Use dd_exponential, dd_hyperbolic, or dd_hyperboloid")
        self.model_type = model_type
        self.num_params = len(MODEL_BOUNDS[model_type])
        self.results = None
        self.parameters = {}

    def _validate_parameters(self, params):
        params = np.asarray(params, dtype=float)
        bounds = np.asarray(MODEL_BOUNDS[self.model_type])
        if (params.shape != (self.num_params,) or not np.isfinite(params).all()
                or np.any(params < bounds[:, 0]) or np.any(params > bounds[:, 1])):
            raise ValueError(f"Expected {self.num_params} finite parameters within MODEL_BOUNDS")
        return params

    def simulate(self, lower_amount, higher_amount_mean, higher_amount_sd,
                 t_mean, t_sd, n_trials_per_agent, n_agent, *,
                 agent_parameters=None, seed=None):
        """See simulate_dd; t_mean/t_sd specify delay in days."""
        return simulate_dd(lower_amount, higher_amount_mean, higher_amount_sd,
                           t_mean, t_sd, n_trials_per_agent, n_agent,
                           model_type=self.model_type,
                           agent_parameters=agent_parameters, seed=seed)

    def negative_log_likelihood(self, params, large_amount, later_delay, small_amount, choice):
        params = self._validate_parameters(params)
        arrays = _trial_arrays(large_amount, later_delay, small_amount, choice)
        return _nll(params, self.model_type, arrays)

    def fit(self, data, num_iterations=20, max_workers=1, *, seed=None):
        """Fit each participant; return estimates, NLL, AIC, BIC, and diagnostics.

        max_workers=1 runs serially; larger values use worker processes, and
        None uses the executor default. A seed gives identical starting points
        in serial and parallel runs. Failed runs are discarded; if all starts
        fail for any participant, raise without replacing prior fit results.
        """
        _positive_integer(num_iterations, "num_iterations")
        if max_workers is not None:
            _positive_integer(max_workers, "max_workers")
        groups = _participant_data(data)
        seeds = np.random.SeedSequence(seed).spawn(len(groups))
        jobs = [(participant, arrays, self.model_type, num_iterations, child_seed)
                for (participant, arrays), child_seed in zip(groups, seeds)]
        if max_workers == 1:
            rows = [_fit_dd_participant(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                rows = list(executor.map(_fit_dd_participant, jobs))
        results = pd.DataFrame(rows)
        self.parameters = {row["participant_id"]: row["best_parameters"].copy() for row in rows}
        self.results = results
        return results

    def evaluate(self, params, data):
        """Evaluate participant trials without refitting; return per-participant metrics.

        params can be the fit result DataFrame, a dictionary of participant
        parameter vectors, a DataFrame indexed by participant with t/k[/s]
        columns, or one parameter vector shared by all participants.
        Pass None to use this instance's last fitted parameters.
        """
        groups = _participant_data(data)
        if params is None:
            if not self.parameters:
                raise ValueError("Fit the model first or supply parameters")
            params = self.parameters
        if isinstance(params, pd.DataFrame):
            if "model_type" in params and not params.model_type.eq(self.model_type).all():
                raise ValueError("Parameter model_type does not match this model")
            if "participant_id" in params:
                params = params.set_index("participant_id")
            if not params.index.is_unique:
                raise ValueError("Parameter participant IDs must be unique")
            if "best_parameters" in params:
                params = params.best_parameters.to_dict()
            else:
                names = ["t", "k", "s"][:self.num_params]
                params = {key: row[names].to_numpy(dtype=float) for key, row in params.iterrows()}
        rows = []
        for participant, arrays in groups:
            if isinstance(params, dict):
                if participant not in params:
                    raise ValueError(f"Missing parameters for participant {participant!r}")
                vector = params[participant]
            else:
                vector = params
            vector = self._validate_parameters(vector)
            large, delay, small, choice = arrays
            delta = vector[0] * (_subjective_value(self.model_type, vector, large, delay) - small)
            nll = _nll(vector, self.model_type, arrays)
            n = len(choice)
            correct = int(np.sum((delta > 0) == choice))
            rows.append({"participant_id": participant, "model_type": self.model_type,
                         "total_nll": nll, "mean_nll": nll / n,
                         "accuracy": correct / n, "n_trials": n, "n_correct": correct})
        return pd.DataFrame(rows)


def fit_dd(data, model_type="dd_hyperbolic", num_iterations=20, max_workers=1, *, seed=None):
    """Convenience function returning the same estimates as DelayedDiscounting.fit."""
    return DelayedDiscounting(model_type).fit(data, num_iterations, max_workers, seed=seed)


def simulate_dd(
    lower_amount,
    higher_amount_mean,
    higher_amount_sd,
    t_mean,
    t_sd,
    n_trials_per_agent,
    n_agent,
    *,
    model_type="dd_hyperbolic",
    agent_parameters=None,
    seed=None,
):
    """Return one row per simulated trial in a pandas DataFrame.

    The seven required inputs specify a fixed immediate reward (dollars),
    the normal distribution of larger rewards (dollars), the normal
    distribution of delays (days), and the number of trials and agents.
    Here ``t_mean``/``t_sd`` refer to DELAY, not the model's inverse
    temperature parameter ``t``.

    Rewards are sampled independently per trial from a normal truncated
    strictly above ``lower_amount``; delays are truncated at zero. Means
    and SDs describe the underlying normals, so realized moments can differ
    after truncation. Zero SD gives a constant value and requires a valid
    mean. The immediate reward is delivered at delay zero.

    model_type : str
        dd_exponential, dd_hyperbolic (default), or dd_hyperboloid.
    agent_parameters : array-like, optional
        One parameter vector shared by all agents, or an (n_agent, n_params)
        array. Order is [t, k], or [t, k, s] for hyperboloid, as in the
        parameterization. Values must lie within MODEL_BOUNDS. If omitted,
        parameters are drawn independently and uniformly from MODEL_BOUNDS
        once per agent and held fixed across that agent's trials. These are
        synthetic parameters, not estimates from participant data.
    seed : int, optional
        Seed for a local NumPy generator (does not change global RNG state).

    Returns
    -------
    pandas.DataFrame
        worker_id and trial are one-based. small_amount, large_amount,
        later_delay, and choice can be passed through
        ``dict_generator_cognitive(df, task='dd')`` to the existing DD fit.
        choice: 0 = smaller-sooner; 1 = larger-later. Also includes
        subjective_value, p_larger_later, model_type, and generating t/k/s
        (s is NaN for models without it). Choice probability is
        expit(t * (subjective_value - small_amount)), matching the existing
        two-option softmax exactly.

    Example
    -------
    >>> trials = simulate_dd(20, 50, 10, 30, 5, 100, 10, seed=42)
    """
    for name, value in (("n_trials_per_agent", n_trials_per_agent), ("n_agent", n_agent)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    values = {}
    for name, value in (("lower_amount", lower_amount),
                        ("higher_amount_mean", higher_amount_mean),
                        ("higher_amount_sd", higher_amount_sd),
                        ("t_mean", t_mean), ("t_sd", t_sd)):
        if not np.isscalar(value) or isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a finite number")
        try:
            values[name] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite number") from exc
        if not np.isfinite(values[name]):
            raise ValueError(f"{name} must be a finite number")
    lower_amount = values["lower_amount"]
    higher_amount_mean = values["higher_amount_mean"]
    higher_amount_sd = values["higher_amount_sd"]
    t_mean, t_sd = values["t_mean"], values["t_sd"]
    if lower_amount < 0 or higher_amount_sd < 0 or t_sd < 0:
        raise ValueError("lower_amount and both SDs must be nonnegative")
    if higher_amount_sd == 0 and higher_amount_mean <= lower_amount:
        raise ValueError("With zero SD, higher_amount_mean must exceed lower_amount")
    if t_sd == 0 and t_mean < 0:
        raise ValueError("With zero SD, t_mean must be nonnegative")

    if model_type not in MODEL_BOUNDS:
        raise ValueError("model_type must be dd_exponential, dd_hyperbolic, or dd_hyperboloid")
    bounds = np.asarray(MODEL_BOUNDS[model_type], dtype=float)
    rng = np.random.default_rng(seed)
    if agent_parameters is None:
        parameters = rng.uniform(bounds[:, 0], bounds[:, 1],
                                 size=(n_agent, len(bounds)))
    else:
        parameters = np.asarray(agent_parameters, dtype=float)
        if parameters.shape == (len(bounds),):
            parameters = np.tile(parameters, (n_agent, 1))
        if parameters.shape != (n_agent, len(bounds)):
            raise ValueError(f"agent_parameters must have shape ({len(bounds)},) "
                             f"or ({n_agent}, {len(bounds)})")
        if (not np.isfinite(parameters).all()
                or np.any(parameters < bounds[:, 0])
                or np.any(parameters > bounds[:, 1])):
            raise ValueError("agent_parameters must be finite and within MODEL_BOUNDS")

    def draw(mean, sd, minimum):
        if sd == 0:
            return np.full(n_trials_per_agent, mean)
        samples = truncnorm.rvs((minimum - mean) / sd, np.inf, loc=mean,
                               scale=sd, size=n_trials_per_agent, random_state=rng)
        return np.maximum(samples, minimum)

    frames = []
    for agent, params in enumerate(parameters, start=1):
        inverse_temperature, k = params[:2]
        s = params[2] if model_type == "dd_hyperboloid" else np.nan
        large_amount = draw(higher_amount_mean, higher_amount_sd,
                            np.nextafter(lower_amount, np.inf))
        later_delay = draw(t_mean, t_sd, 0.0)
        subjective_value = _subjective_value(model_type, params, large_amount, later_delay)
        probability = expit(inverse_temperature * (subjective_value - lower_amount))
        frames.append(pd.DataFrame({
            "worker_id": agent,
            "trial": np.arange(1, n_trials_per_agent + 1),
            "small_amount": lower_amount,
            "large_amount": large_amount,
            "later_delay": later_delay,
            "choice": rng.binomial(1, probability),
            "subjective_value": subjective_value,
            "p_larger_later": probability,
            "model_type": model_type,
            "t": inverse_temperature,
            "k": k,
            "s": s,
        }))
    return pd.concat(frames, ignore_index=True)
